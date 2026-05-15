import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv




class DCA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv_h3 = nn.Conv2d(c, c, (1,3), padding=(0,1), groups=c, bias=False)
        self.conv_w3 = nn.Conv2d(c, c, (3,1), padding=(1,0), groups=c, bias=False)
        self.conv_h5 = nn.Conv2d(c, c, (1,5), padding=(0,2), groups=c, bias=False)
        self.conv_w5 = nn.Conv2d(c, c, (5,1), padding=(2,0), groups=c, bias=False)
        self.pw      = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        h    = self.conv_h3(x) + self.conv_h5(x)
        w    = self.conv_w3(x) + self.conv_w5(x)
        attn = torch.sigmoid(self.pw(h + w))
        return x * attn

class RFAConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, dilations=(1, 2, 3)):
        super().__init__()
        self.n = len(dilations)
        self.c2 = c2

        # 每个 dilation 一个 DWConv（深度可分离，参数少）
        self.dw_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c1, c1, k, stride=s,
                          padding=d * (k // 2), dilation=d,
                          groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                nn.SiLU(inplace=True),
            ) for d in dilations
        ])

        # 感受野注意力：对 n 路特征预测 softmax 权重
        # 输入：n 路 cat → n*c1 通道，输出：n 个标量权重图
        self.attn = nn.Sequential(
            nn.Conv2d(c1 * self.n, self.n, 1, bias=False),  # n*c1 → n
            nn.Softmax(dim=1),                               # 在 n 维做 softmax
        )

        # 1x1 逐点卷积完成通道变换 c1 → c2
        self.pw = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        # 各感受野分支
        feats = [dw(x) for dw in self.dw_convs]          # n × (B, c1, H, W)

        # 预测每个位置各感受野的权重
        feat_cat = torch.cat(feats, dim=1)                 # (B, n*c1, H, W)
        weights  = self.attn(feat_cat)                     # (B, n, H, W)

        # 加权求和
        out = sum(weights[:, i:i+1, :, :] * feats[i]
                  for i in range(self.n))                  # (B, c1, H, W)

        return self.pw(out)                                # (B, c2, H, W)

class Bottleneck_RFA_DCA(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, dilations=(1,2,3)):
        super().__init__()
        c_ = int(c2 * e)

        # branch_std：标准卷积路（保留局部规则特征作为基准）
        self.cv1_std = Conv(c1, c_, 1, 1)
        self.cv2_std = Conv(c_, c_, 3, 1)

        # branch_rfa：DCA 方向感知 → RFAConv 多尺度感受野
        self.cv1_rfa = Conv(c1, c_, 1, 1)
        self.dca     = DCA(c_)
        self.rfa     = RFAConv(c_, c_, k=3, s=1, dilations=dilations)

        # branch_res：原始特征直连（梯度高速公路）
        self.res_conv = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        # 三路融合：3*c_ → c2
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * c_, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 标准路
        out_std = self.cv2_std(self.cv1_std(x))

        # RFA 路：先方向感知，再多尺度感受野加权
        y       = self.cv1_rfa(x)
        y       = self.dca(y)        # 方向注意力引导后续采样方向
        out_rfa = self.rfa(y)        # 多尺度感受野自适应加权

        # 残差直连路
        out_res = self.res_conv(x)

        # 三路融合
        out = self.fuse(torch.cat([out_std, out_rfa, out_res], dim=1))

        return x + out if self.add else out

class C2f_DCN_DCA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c   = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m   = nn.ModuleList(
            Bottleneck_RFA_DCA(self.c, self.c, shortcut, e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

