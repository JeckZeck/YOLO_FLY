import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
class SAGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)      
        mx  = x.amax(dim=1, keepdim=True)     
        mask = self.conv(torch.cat([avg, mx], dim=1))  # (B,1,H,W)
        return x * mask
    
class CrossFuse(nn.Module):
    def __init__(self, c):
        super().__init__()
        mid = c // 2
        assert mid % 2 == 0, f"CrossFuse: c//2={mid} must be even"

        self.branch_local = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )

        self.branch_strip = nn.Sequential(
            nn.Conv2d(mid, mid, (1, 5), padding=(0, 2), groups=mid, bias=False),
            nn.Conv2d(mid, mid, (5, 1), padding=(2, 0), groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )

        self.gate_for_local = nn.Sequential(
            nn.Conv2d(mid, mid, 1, bias=False), nn.Sigmoid()
        )
        self.gate_for_strip = nn.Sequential(
            nn.Conv2d(mid, mid, 1, bias=False), nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        self.sa = SAGate()

    def forward(self, x):
        a, b = x.chunk(2, dim=1)

        fa = self.branch_local(a)
        fb = self.branch_strip(b)

        fa = fa * self.gate_for_local(fb)
        fb = fb * self.gate_for_strip(fa)

        out = self.fuse(torch.cat([fa, fb], dim=1))

        return self.sa(out)

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


