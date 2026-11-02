"""GIA and CDSA attention blocks for YOLO11, designed for thin, long road cracks.

  - GIA, Gradient-guided Image Attention: fixed Sobel filters measure where a feature map has strong
    edges (cracks are thin, high-gradient lines). A spatial gate (where) and a channel gate (which
    channels) built from that gradient re-weight the features.
  - CDSA, Contextual Dilated Spatial Attention: parallel dilated depthwise convolutions (dilation 1, 2, 4)
    plus global context give each position a wide view along long cracks, and produce a spatial
    attention map over the features.

Designed for this project from the module names (no published reference implementation was found),
from standard parts: Sobel gradients, CBAM-style spatial/channel gates (Woo et al., ECCV 2018) and
dilated convolutions (Yu & Koltun, ICLR 2016).

Every block is residual with a zero-initialised layer scale (gamma), so it starts as an identity and
does not disturb a pretrained YOLO11 at the start of fine-tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


# ------------------------------------------------------------------ Gradient-guided Image Attention
class GIA(nn.Module):
    """x + gamma * proj(r * spatial_gate * channel_gate), gates computed from the Sobel gradient of r.

    r is x reduced to c_ = max(32, c/4) channels. Spatial gate: 7x7 conv over [mean |grad|, max |grad|,
    mean r, max r]. Channel gate: small MLP over the average |grad| of each channel.
    """

    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        c_ = max(32, c // reduction)
        self.c_ = c_
        self.reduce = Conv(c, c_, 1)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        # depthwise kernels in the order (x, y) per channel: fixed, not learned
        self.register_buffer("sobel", torch.stack([sobel_x, sobel_x.t()])[:, None].repeat(c_, 1, 1, 1))
        self.spatial = nn.Conv2d(4, 1, 7, padding=3)
        hidden = max(8, c_ // 4)
        self.channel = nn.Sequential(nn.Conv2d(c_, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, c_, 1))
        self.proj = nn.Conv2d(c_, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # layer scale: block starts as identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.reduce(x)
        g = F.conv2d(r, self.sobel.to(r.dtype), padding=1, groups=self.c_)
        gx, gy = g[:, 0::2].float(), g[:, 1::2].float()            # float32: squares can overflow fp16
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6).to(r.dtype)      # B, c_, H, W
        s = torch.sigmoid(self.spatial(torch.cat([mag.mean(1, keepdim=True), mag.amax(1, keepdim=True),
                                                  r.mean(1, keepdim=True), r.amax(1, keepdim=True)], dim=1)))
        ch = torch.sigmoid(self.channel(mag.mean((2, 3), keepdim=True)))
        return x + self.gamma * self.proj(r * s * ch)


# ------------------------------------------------------------------ Contextual Dilated Spatial Attention
class CDSA(nn.Module):
    """x + gamma * proj(r * a), a = sigmoid(1x1 conv([dw3x3 d=1, d=2, d=4 of r, global mean of r])).

    r is x reduced to c_ = max(32, c/4) channels. The dilated branches see 3, 5 and 9 positions across,
    so the attention at each point uses context along the crack.
    """

    def __init__(self, c: int, reduction: int = 4, dilations: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        c_ = max(32, c // reduction)
        self.reduce = Conv(c, c_, 1)
        self.branches = nn.ModuleList(Conv(c_, c_, 3, g=c_, d=d) for d in dilations)  # depthwise, dilated
        self.attn = nn.Conv2d(c_ * (len(dilations) + 1), 1, 1)
        self.proj = nn.Conv2d(c_, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # layer scale: block starts as identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.reduce(x)
        context = [b(r) for b in self.branches] + [r.mean((2, 3), keepdim=True).expand_as(r)]
        a = torch.sigmoid(self.attn(torch.cat(context, dim=1)))
        return x + self.gamma * self.proj(r * a)
