import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
# ─────────────────────────────────────────────────────────────────────
# SAGate — 出口空间注意力门控
# ─────────────────────────────────────────────────────────────────────
class SAGate(nn.Module):
    """
    空间注意力门控（Spatial Attention Gate）。

    来源：CBAM 空间分支的轻量改进版。
    用 avg + max 双池化联合描述每个位置的特征强度，
    再用大感受野 Conv7x7 建模空间依赖，生成 [0,1] 空间掩码。

    参数量：仅 2 个 BN 参数 + Conv7x7（输入2通道，极少）。
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 在通道维度分别做均值池化和最大池化，各得到 1 通道空间图
        avg = x.mean(dim=1, keepdim=True)          # (B,1,H,W) 均值
        mx  = x.amax(dim=1, keepdim=True)          # (B,1,H,W) 最大值
        # 拼接后卷积得到空间权重
        mask = self.conv(torch.cat([avg, mx], dim=1))  # (B,1,H,W)
        return x * mask
# ─────────────────────────────────────────────────────────────────────
# CrossFuse — 交叉感知融合块 + 出口 SAGate
# ─────────────────────────────────────────────────────────────────────
class CrossFuse(nn.Module):
    """
    交叉感知融合块，专为 BiFPN 融合节点设计。

    完整流程：
        输入(c)
          → split(c//2, c//2)
          → 路A: Conv3x3（局部空间细节）
             路B: DW(1x5)+DW(5x1)（全局条带上下文，与backbone DCA感受野互补）
          → 交叉门控: fa *= σ(proj(fb)),  fb *= σ(proj(fa))
          → cat → Conv1x1 → 通道融合
          → SAGate → 空间校准（新增，出口位置）
    """

    def __init__(self, c):
        super().__init__()
        mid = c // 2
        assert mid % 2 == 0, f"CrossFuse: c//2={mid} must be even"

        # 路A：局部空间
        self.branch_local = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )

        # 路B：大感受野条带（DW，轻量）
        self.branch_strip = nn.Sequential(
            nn.Conv2d(mid, mid, (1, 5), padding=(0, 2), groups=mid, bias=False),
            nn.Conv2d(mid, mid, (5, 1), padding=(2, 0), groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )

        # 交叉门控
        self.gate_for_local = nn.Sequential(
            nn.Conv2d(mid, mid, 1, bias=False), nn.Sigmoid()
        )
        self.gate_for_strip = nn.Sequential(
            nn.Conv2d(mid, mid, 1, bias=False), nn.Sigmoid()
        )

        # 通道融合
        self.fuse = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        # 出口空间注意力（位置：通道融合之后，输出之前）
        self.sa = SAGate()

    def forward(self, x):
        a, b = x.chunk(2, dim=1)

        fa = self.branch_local(a)
        fb = self.branch_strip(b)

        fa = fa * self.gate_for_local(fb)
        fb = fb * self.gate_for_strip(fa)

        out = self.fuse(torch.cat([fa, fb], dim=1))

        # 空间校准：对融合后的结果做全局空间掩码
        return self.sa(out)
# ─────────────────────────────────────────────────────────────────────
# BiFPNFuse
# ─────────────────────────────────────────────────────────────────────
class BiFPNFuse(nn.Module):
    def __init__(self, c1, c2, n_inputs=2):
        super().__init__()
        self.n   = n_inputs
        self.w   = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.eps = 1e-4

        self.align = nn.Identity() if c1 == c2 else nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        self.cross_fuse = CrossFuse(c2)

    def forward(self, x: list):
        assert len(x) == self.n, f"BiFPNFuse expects {self.n} inputs, got {len(x)}"

        w   = torch.relu(self.w)
        w   = w / (w.sum() + self.eps)
        out = sum(w[i] * x[i] for i in range(self.n))
        out = self.align(out)

        return self.cross_fuse(out)


