"""BSW-YOLO11: insert the base paper's modules into a pretrained YOLO11 and train it with Ultralytics.

  - BRA-DropKey after layer 8 (the last backbone C3k2, i.e. before SPPF)
  - SnakeConv after layers 16, 19, 22 (the neck outputs feeding Detect P3/P4/P5)
  - Wise-IoU v3 instead of CIoU in the box loss

Layers are wrapped in place (keeping the .f/.i/.type attributes Ultralytics' forward loop uses),
so pretrained COCO weights load first and the new blocks start as identities.
Checkpoints pickle these classes: import this module (src/ on sys.path) before loading them.
"""
import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss

from bsw_modules import BRADropKey, SnakeConvBlock, WiseIoUv3

BRA_LAYER = 8
SNAKE_LAYERS = (16, 19, 22)


class WIoUBboxLoss(BboxLoss):
    """Ultralytics BboxLoss with Wise-IoU v3 in place of CIoU (DFL part unchanged)."""

    def __init__(self, reg_max: int = 16):
        super().__init__(reg_max)
        self.wiou = WiseIoUv3()

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                target_scores_sum, fg_mask, imgsz, stride):
        _, loss_dfl = super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                      target_scores_sum, fg_mask, imgsz, stride)
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        loss_iou = (self.wiou(pred_bboxes[fg_mask], target_bboxes[fg_mask]) * weight).sum() / target_scores_sum
        return loss_iou, loss_dfl


class WIoUDetectionLoss(v8DetectionLoss):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.bbox_loss = WIoUBboxLoss(self.reg_max).to(next(model.parameters()).device)


class BSWDetectionModel(DetectionModel):
    """DetectionModel that remembers its BSW options and uses Wise-IoU when asked."""

    bsw = {"snake": False, "bra": False, "wiou": False}

    def init_criterion(self):
        return WIoUDetectionLoss(self) if self.bsw.get("wiou") else super().init_criterion()


class Wrapped(nn.Module):
    """base layer followed by an extra block; carries Ultralytics' routing attributes."""

    def __init__(self, base: nn.Module, extra: nn.Module):
        super().__init__()
        self.base, self.extra = base, extra
        self.f, self.i = base.f, base.i
        self.type = f"{base.type}+{extra.__class__.__name__}"
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self.extra(self.base(x))


def _out_channels(model: DetectionModel, layers) -> dict[int, int]:
    """Output channels of the given layer indices, found with a tiny forward pass."""
    found, hooks = {}, []
    for i in layers:
        hooks.append(model.model[i].register_forward_hook(lambda m, a, o, i=i: found.__setitem__(i, o.shape[1])))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 128, 128, device=next(model.parameters()).device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return found


def apply_bsw(model: DetectionModel, snake: bool, bra: bool) -> DetectionModel:
    """Wrap YOLO11 layers with BRA-DropKey / SnakeConv. Checks the layout is the YOLO11 detect one."""
    assert model.model[BRA_LAYER].type.endswith("C3k2") and model.model[BRA_LAYER + 1].type.endswith("SPPF"), \
        "unexpected layout: layer 8 must be C3k2 followed by SPPF (YOLO11 detect)"
    assert list(model.model[-1].f) == list(SNAKE_LAYERS), "Detect must read layers 16, 19, 22"
    ch = _out_channels(model, (BRA_LAYER, *SNAKE_LAYERS))
    device = next(model.parameters()).device
    if bra:
        model.model[BRA_LAYER] = Wrapped(model.model[BRA_LAYER], BRADropKey(ch[BRA_LAYER]).to(device))
    if snake:
        for i in SNAKE_LAYERS:
            model.model[i] = Wrapped(model.model[i], SnakeConvBlock(ch[i]).to(device))
    return model


def build_bsw_model(cfg, nc: int, ch: int, weights=None, snake=False, bra=False, wiou=False,
                    verbose=True, names=None):
    """Build BSWDetectionModel. Plain pretrained weights load before surgery; BSW weights after.

    names are set before loading, as Ultralytics does, so the class head can remap by name.
    """
    model = BSWDetectionModel(cfg, nc=nc, ch=ch, verbose=verbose)
    if names:
        model.names = names
    model.bsw = {"snake": snake, "bra": bra, "wiou": wiou}
    model.yaml["bsw"] = dict(model.bsw)
    weights_are_bsw = weights is not None and isinstance(weights, nn.Module) and hasattr(weights, "bsw") \
        and any(isinstance(m, Wrapped) for m in weights.model)
    if weights is not None and not weights_are_bsw:
        model.load(weights)                      # COCO / plain YOLO11 weights into the original layout
    apply_bsw(model, snake, bra)
    if weights_are_bsw:
        model.load(weights)                      # resuming a BSW checkpoint: same layout on both sides
    return model


def make_trainer(snake: bool = False, bra: bool = False, wiou: bool = False):
    """DetectionTrainer subclass that builds BSW-YOLO11 with the given modules."""

    class BSWTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if isinstance(cfg, dict):
                cfg = {k: v for k, v in cfg.items() if k != "bsw"}
            names = self.data["names"] if getattr(self.args, "cls_remap", True) and self.data.get("names") else None
            return build_bsw_model(cfg, self.data["nc"], self.data["channels"], weights, snake=snake, bra=bra,
                                   wiou=wiou, verbose=verbose and RANK == -1, names=names)

    BSWTrainer.__name__ = f"BSWTrainer(snake={snake},bra={bra},wiou={wiou})"
    return BSWTrainer
