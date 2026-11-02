"""Modules from the base paper BSW-YOLO (Wang et al., Eng. Appl. of AI 163, 2026, 112618), for YOLO11.

Written from the original method papers:
  - Dynamic Snake Convolution: Qi et al., "Dynamic Snake Convolution based on Topological
    Geometric Constraints for Tubular Structure Segmentation", ICCV 2023.
  - Bi-level Routing Attention: Zhu et al., "BiFormer", CVPR 2023; DropKey: Li et al., CVPR 2023.
  - Wise-IoU v3: Tong et al., "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing
    Mechanism", 2023.

Every block is residual and starts as an identity (zero-initialised last BN / zero layer scale),
so inserting it does not disturb a pretrained YOLO11 at the start of fine-tuning.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


# ------------------------------------------------------------------ Dynamic Snake Convolution
class DSConv(nn.Module):
    """Dynamic Snake Convolution along one axis.

    morph="x": the kernel runs along x and each point wiggles in y (for longitudinal-looking lines
    in feature space); morph="y": runs along y and wiggles in x. Offsets are accumulated outward
    from the kernel centre, so the kernel stays a connected, snake-like curve.
    """

    def __init__(self, c1: int, c2: int, k: int = 9, morph: str = "x", extend_scope: float = 1.0):
        super().__init__()
        assert morph in ("x", "y") and k % 2 == 1
        self.k, self.morph, self.scope = k, morph, extend_scope
        # Default init as in DSCNet. Zero-initialising this conv would feed BN a constant and blow up
        # its gradients; the enclosing SnakeConvBlock already starts as an identity.
        self.offset = nn.Conv2d(c1, 2 * k, 3, padding=1)
        self.offset_bn = nn.BatchNorm2d(2 * k)
        ks, st = ((1, k), (1, k)) if morph == "x" else ((k, 1), (k, 1))
        self.conv = Conv(c1, c2, k=ks, s=st, p=0)

    @staticmethod
    def _accumulate(off: torch.Tensor) -> torch.Tensor:
        """off (B, K, H, W) -> cumulative offsets from the centre outward, centre = 0."""
        c = off.shape[1] // 2
        right = torch.cumsum(off[:, c + 1:], dim=1)
        left = torch.cumsum(off[:, :c].flip(1), dim=1).flip(1)
        return torch.cat([left, torch.zeros_like(off[:, c:c + 1]), right], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        k, c = self.k, self.k // 2
        off = torch.tanh(self.offset_bn(self.offset(x))) * self.scope
        y_off, x_off = off[:, :k], off[:, k:]
        steps = torch.arange(k, device=x.device, dtype=off.dtype) - c
        yy = torch.arange(h, device=x.device, dtype=off.dtype).view(1, 1, h, 1)
        xx = torch.arange(w, device=x.device, dtype=off.dtype).view(1, 1, 1, w)
        if self.morph == "x":  # kernel along x, wiggle in y
            py = yy + self._accumulate(y_off)                      # B, K, H, W
            px = (xx + steps.view(1, k, 1, 1)).expand_as(py)
            # arrange so the K samples are consecutive along width: B, H, W*K
            py, px = py.permute(0, 2, 3, 1).reshape(b, h, w * k), px.permute(0, 2, 3, 1).reshape(b, h, w * k)
        else:  # kernel along y, wiggle in x
            px = xx + self._accumulate(x_off)
            py = (yy + steps.view(1, k, 1, 1)).expand_as(px)
            # consecutive along height: B, H*K, W
            py, px = py.permute(0, 2, 1, 3).reshape(b, h * k, w), px.permute(0, 2, 1, 3).reshape(b, h * k, w)
        gx = 2.0 * px / max(w - 1, 1) - 1.0
        gy = 2.0 * py / max(h - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).to(x.dtype)
        sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return self.conv(sampled)


class SnakeConvBlock(nn.Module):
    """Paper Fig. 4: [DSConv-x, DSConv-y, Conv] fused by a 1x1 conv, here with a residual.

    The branches run on a reduced width (c/4, min 32) to fit a 6 GB GPU.
    """

    def __init__(self, c: int, k: int = 9, reduction: int = 4):
        super().__init__()
        c_ = max(32, c // reduction)
        self.reduce = Conv(c, c_, 1)
        self.conv = Conv(c_, c_, 3)
        self.dsc_x = DSConv(c_, c_, k, "x")
        self.dsc_y = DSConv(c_, c_, k, "y")
        self.fuse = Conv(3 * c_, c, 1, act=False)
        nn.init.zeros_(self.fuse.bn.weight)  # block starts as identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.reduce(x)
        return x + self.fuse(torch.cat([self.conv(r), self.dsc_x(r), self.dsc_y(r)], dim=1))


# ------------------------------------------------------------------ Bi-level Routing Attention + DropKey
class BRADropKey(nn.Module):
    """Bi-level Routing Attention with DropKey and local context enhancement (LCE).

    1) Split the map into n_win x n_win regions; route each region to its top-k most related
       regions (pooled query . pooled key). 2) Token attention over the gathered keys/values.
    DropKey: during training, random attention logits are masked before softmax (prob=drop).
    """

    def __init__(self, c: int, n_win: int = 4, topk: int = 4, head_dim: int = 64, drop: float = 0.1):
        super().__init__()
        self.n_win, self.topk, self.drop = n_win, topk, drop
        self.heads = max(1, c // head_dim)
        self.scale = (c // self.heads) ** -0.5
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.lce = nn.Conv2d(c, c, 5, padding=2, groups=c)
        self.proj = nn.Conv2d(c, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # layer scale: module starts as identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.n_win
        hp, wp = math.ceil(h / s) * s, math.ceil(w / s) * s
        xp = F.pad(x, (0, wp - w, 0, hp - h))
        q, k, v = self.qkv(xp).chunk(3, dim=1)
        rh, rw = hp // s, wp // s
        n_reg, n_tok = s * s, rh * rw

        def regions(t):  # B, C, Hp, Wp -> B, S*S, rh*rw, C
            return t.view(b, c, s, rh, s, rw).permute(0, 2, 4, 3, 5, 1).reshape(b, n_reg, n_tok, c)

        qr, kr, vr = regions(q), regions(k), regions(v)
        topk = min(self.topk, n_reg)
        idx = (qr.mean(2) @ kr.mean(2).transpose(-1, -2)).topk(topk, dim=-1).indices  # B, S2, topk
        gather_idx = idx[..., None, None].expand(-1, -1, -1, n_tok, c)
        kg = torch.gather(kr.unsqueeze(1).expand(-1, n_reg, -1, -1, -1), 2, gather_idx).reshape(b, n_reg, topk * n_tok, c)
        vg = torch.gather(vr.unsqueeze(1).expand(-1, n_reg, -1, -1, -1), 2, gather_idx).reshape(b, n_reg, topk * n_tok, c)

        nh, d = self.heads, c // self.heads
        qh = qr.view(b, n_reg, n_tok, nh, d).transpose(2, 3)
        kh = kg.view(b, n_reg, topk * n_tok, nh, d).transpose(2, 3)
        vh = vg.view(b, n_reg, topk * n_tok, nh, d).transpose(2, 3)
        attn = (qh * self.scale) @ kh.transpose(-1, -2)                  # B, S2, nh, tok, topk*tok
        if self.training and self.drop > 0:                              # DropKey
            attn = attn.masked_fill(torch.rand_like(attn) < self.drop, -1e4)
        out = attn.softmax(dim=-1) @ vh                                   # B, S2, nh, tok, d
        out = out.transpose(2, 3).reshape(b, s, s, rh, rw, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, hp, wp)
        out = self.proj(out + self.lce(v))[:, :, :h, :w]
        return x + self.gamma * out


# ------------------------------------------------------------------ Wise-IoU v3
class WiseIoUv3(nn.Module):
    """Wise-IoU v3 box loss: L = r * R_WIoU * L_IoU.

    R_WIoU = exp(centre_dist^2 / enclosing_diag^2) with the enclosing box detached;
    r = beta / (delta * alpha^(beta - delta)), beta = L_IoU / running_mean(L_IoU) (non-monotonic
    focusing: both very good and very bad (likely mislabelled) boxes get less gradient).
    """

    def __init__(self, alpha: float = 1.9, delta: float = 3.0, momentum: float = 0.01, eps: float = 1e-7):
        super().__init__()
        self.alpha, self.delta, self.momentum, self.eps = alpha, delta, momentum, eps
        self.register_buffer("iou_mean", torch.tensor(1.0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: (N, 4) xyxy. Returns per-box loss (N, 1)."""
        (px1, py1, px2, py2), (tx1, ty1, tx2, ty2) = pred.float().unbind(-1), target.float().unbind(-1)
        iw = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0)
        ih = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
        inter = iw * ih
        union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - inter + self.eps
        l_iou = 1.0 - inter / union
        cw = torch.max(px2, tx2) - torch.min(px1, tx1)
        ch = torch.max(py2, ty2) - torch.min(py1, ty1)
        dist = ((px1 + px2 - tx1 - tx2) ** 2 + (py1 + py2 - ty1 - ty2) ** 2) / 4
        r_wiou = torch.exp(dist / (cw ** 2 + ch ** 2 + self.eps).detach())
        if torch.is_grad_enabled() and l_iou.numel():  # update only on training steps, not val loss
            self.iou_mean.mul_(1 - self.momentum).add_(self.momentum * l_iou.detach().mean())
        beta = l_iou.detach() / self.iou_mean.clamp(min=self.eps)
        r = beta / (self.delta * self.alpha ** (beta - self.delta))
        return (r * r_wiou * l_iou).unsqueeze(-1)
