import copy
import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.block import DFL
from ultralytics.utils.tal import dist2bbox, make_anchors

class GSBranch(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        self.pw = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(c2, c2, k, padding=k // 2, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(self.pw(x))

class PConvBranch(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        assert c2 % 4 == 0, f"PConvBranch: c2={c2} "
        mid = c2 // 4
        pads    = [(0, 0, k - 1, 0), (k - 1, 0, 0, 0), (0, 0, 0, k - 1), (0, k - 1, 0, 0)]
        kernels = [(k, 1),           (1, k),            (k, 1),            (1, k)          ]
        self.pad      = nn.ModuleList([nn.ZeroPad2d(p) for p in pads])
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c1, mid, ki, padding=0, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(inplace=True),
            ) for ki in kernels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(p(x)) for p, b in zip(self.pad, self.branches)]
        return self.fuse(torch.cat(outs, dim=1))


class DualBranchHead(nn.Module):
   
    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        assert c2 % 4 == 0, f"DualBranchHead: c2={c2} "
        c_half = c2 // 2
        self.gs_branch   = GSBranch(c1, c_half, k=k)
        self.pconv_branch = PConvBranch(c1, c_half, k=k)
        self.fuse = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gs  = self.gs_branch(x)
        pc  = self.pconv_branch(x)
        return self.fuse(torch.cat([gs, pc], dim=1))

class Detect(nn.Module):
    dynamic = False
    export  = False
    format  = None
    end2end = False
    max_det = 300
    shape   = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy  = False
    xyxy    = False

    def __init__(self, nc: int = 80, ch: Tuple = ()):
        super().__init__()
        self.nc      = nc
        self.nl      = len(ch)
        self.reg_max = 16
        self.no      = nc + self.reg_max * 4
        self.stride  = torch.zeros(self.nl)

        c2_raw = max(16, ch[0] // 4, self.reg_max * 4) if ch else 64
        c2     = (c2_raw + 3) // 4 * 4  

        c3_raw = max(ch[0], min(self.nc, 100)) if ch else 80
        c3     = (c3_raw + 3) // 4 * 4  
        
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                DualBranchHead(x, c2, k=3),    
                nn.Conv2d(c2, 4 * self.reg_max, 1), 
            )
            for x in ch
        )

        if self.legacy:
            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    Conv(x, c3, 3),
                    Conv(c3, c3, 3),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        else:
            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    DualBranchHead(x, c3, k=3),  
                    nn.Conv2d(c3, self.nc, 1),   
                )
                for x in ch
            )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple]:
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x: List[torch.Tensor]) -> Union[dict, Tuple]:
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]),
                        self.one2one_cv3[i](x_detach[i])), 1)
            for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x: List[torch.Tensor]) -> torch.Tensor:
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (
                t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h   = shape[2]
            grid_w   = shape[3]
            grid_size = torch.tensor(
                [grid_w, grid_h, grid_w, grid_h], device=box.device
            ).reshape(1, 4, 1)
            norm  = self.strides / (self.stride[0] * grid_size)
            dbox  = self.decode_bboxes(
                self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2]
            )
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        if self.export and self.format == "imx":
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):
                a[-1].bias.data[:] = 1.0
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

    def decode_bboxes(
        self,
        bboxes: torch.Tensor,
        anchors: torch.Tensor,
        xywh: bool = True,
    ) -> torch.Tensor:
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not (self.end2end or self.xyxy),
            dim=1,
        )

    @staticmethod
    def postprocess(
        preds: torch.Tensor,
        max_det: int,
        nc: int = 80,
    ) -> torch.Tensor:
        batch_size, anchors, _ = preds.shape
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes  = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]
        return torch.cat(
            [boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()],
            dim=-1,
        )

