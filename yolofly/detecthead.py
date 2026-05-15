import copy
import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.block import DFL
from ultralytics.utils.tal import dist2bbox, make_anchors

class GSBranch(nn.Module):
    """
    Group-Shuffle 空间分支：点卷积压缩 → 深度卷积提取空间特征。
    输出通道 = c2（c2 必须为偶数）。
    """
    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        # 逐点压缩
        self.pw = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        # 深度卷积（空间聚合）
        self.dw = nn.Sequential(
            nn.Conv2d(c2, c2, k, padding=k // 2, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(self.pw(x))


class PConvBranch(nn.Module):
    """
    Pinwheel 方向分支：4 方向条形核并行，捕获水平/垂直方向特征。
    输出通道 = c2（c2 必须为 4 的倍数）。
    """
    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        assert c2 % 4 == 0, f"PConvBranch: c2={c2} 必须是 4 的倍数"
        mid = c2 // 4
        # 4 方向：上/左/下/右 各一路，padding 让输出 HW 与输入一致
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
        # 4 路 cat 后融合
        self.fuse = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(p(x)) for p, b in zip(self.pad, self.branches)]
        return self.fuse(torch.cat(outs, dim=1))


class DualBranchHead(nn.Module):
    """
    双分支宽化模块（DetectPC 的核心构件）。

    结构：
        GSBranch(c1 → c_half)  ─┐
                                  cat → Conv2d(c2, c2, 1, BN, SiLU)  → out(c2)
        PConvBranch(c1 → c_half)─┘

    输出通道 = c2，深度仅 1 层（替代原版连续 2 层 3×3 Conv）。

    要求：
        c2 必须满足 c2 % 4 == 0（PConvBranch 约束）
        c_half = c2 // 2 必须 >= 4
    """
    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        assert c2 % 4 == 0, f"DualBranchHead: c2={c2} 必须是 4 的倍数"
        c_half = c2 // 2
        self.gs_branch   = GSBranch(c1, c_half, k=k)
        self.pconv_branch = PConvBranch(c1, c_half, k=k)
        # 融合压缩（将 c2 通道的 cat 结果保持为 c2）
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
    """
    DetectPC — PConv 双分支宽化检测头，接口与官方 Detect 完全兼容。

    cv2（边框回归头）结构（深度=2）：
        DualBranchHead(x → c2)  →  Conv2d(c2 → 4*reg_max, 1)

    cv3（分类头）结构（深度=2）：
        DualBranchHead(x → c3)  →  Conv2d(c3 → nc, 1)

    对比官方 Detect（深度=3）：
        Conv(x,c,3) → Conv(c,c,3) → Conv2d(c,out,1)
        ↓↓↓  本版  ↓↓↓
        DualBranch(x→c)         → Conv2d(c,out,1)   ← 少1层，宽2倍分支

    Attributes:
        dynamic  (bool)  : 强制重建网格
        export   (bool)  : 导出模式
        format   (str)   : 导出格式
        end2end  (bool)  : 端到端检测模式
        max_det  (int)   : 最大检测数
        legacy   (bool)  : 兼容旧版 v3/v5/v8/v9
        xyxy     (bool)  : 输出格式 xyxy 或 xywh
    """

    # 类属性，与官方 Detect 保持一致
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
        """
        Args:
            nc (int): 类别数。
            ch (tuple): 各特征层通道数，来自 backbone/neck。
        """
        super().__init__()
        self.nc      = nc
        self.nl      = len(ch)
        self.reg_max = 16
        self.no      = nc + self.reg_max * 4
        self.stride  = torch.zeros(self.nl)

        # ── 计算内部通道，与官方 Detect 相同公式，向上对齐到 4 的倍数 ──
        c2_raw = max(16, ch[0] // 4, self.reg_max * 4) if ch else 64
        c2     = (c2_raw + 3) // 4 * 4   # 向上对齐到 4 的倍数（PConv 要求）

        c3_raw = max(ch[0], min(self.nc, 100)) if ch else 80
        c3     = (c3_raw + 3) // 4 * 4   # 同上

        # ── 边框回归头 cv2：双分支(深度1) + 预测层(深度1) = 总深度 2 ──
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                DualBranchHead(x, c2, k=3),          # 宽化：GSConv ‖ PConv
                nn.Conv2d(c2, 4 * self.reg_max, 1),  # 预测层
            )
            for x in ch
        )

        # ── 分类头 cv3：双分支(深度1) + 预测层(深度1) = 总深度 2 ──
        #    legacy=True 时回退到官方简单结构（兼容旧权重）
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
                    DualBranchHead(x, c3, k=3),   # 宽化：GSConv ‖ PConv
                    nn.Conv2d(c3, self.nc, 1),     # 预测层
                )
                for x in ch
            )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # end2end 模式（对应 YOLOv10 one2one 分支）
        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    # ─────────────────── forward（与官方 Detect 完全一致）────────────────────

    def forward(self, x: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple]:
        """拼接边框回归和分类预测，返回结果。"""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x: List[torch.Tensor]) -> Union[dict, Tuple]:
        """端到端前向（YOLOv10 one2one 模式）。"""
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
        """解码多尺度特征图，输出边框 + 分类概率。"""
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

    # ─────────────────── 工具方法（与官方 Detect 完全一致）──────────────────

    def bias_init(self):
        """初始化检测头偏置，需要 stride 已计算完毕。"""
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
        """从预测分布解码边框坐标。"""
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
        """后处理：Top-K 筛选，输出 [x,y,w,h, conf, cls]。"""
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

