"""YOLO11 + GIA + CDSA: insert the two attention blocks into a pretrained YOLO11 and train it with Ultralytics.

  - GIA (Gradient-guided Image Attention) after layers 4 and 6: the backbone C3k2 outputs at stride 8
    and 16, where the features still follow image edges
  - CDSA (Contextual Dilated Spatial Attention) after layers 16, 19, 22: the neck outputs feeding Detect
  - standard YOLO11 loss; none of the BSW parts (SnakeConv, BRA-DropKey, Wise-IoU)

Layers are wrapped in place (keeping the .f/.i/.type attributes Ultralytics' forward loop uses),
so pretrained COCO weights load first and the new blocks start as identities.
Checkpoints pickle these classes: import this module (src/ on sys.path) before loading them.
"""
import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

from gc_modules import CDSA, GIA

GIA_LAYERS = (4, 6)
CDSA_LAYERS = (16, 19, 22)


class GCDetectionModel(DetectionModel):
    """DetectionModel that remembers which GIA / CDSA blocks it has."""

    gc = {"gia": False, "cdsa": False}


class GCWrapped(nn.Module):
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


def apply_gc(model: DetectionModel, gia: bool, cdsa: bool) -> DetectionModel:
    """Wrap YOLO11 layers with GIA / CDSA. Checks the layout is the YOLO11 detect one."""
    assert all(model.model[i].type.endswith("C3k2") for i in GIA_LAYERS), \
        "unexpected layout: layers 4 and 6 must be C3k2 (YOLO11 detect)"
    assert list(model.model[-1].f) == list(CDSA_LAYERS), "Detect must read layers 16, 19, 22"
    ch = _out_channels(model, (*GIA_LAYERS, *CDSA_LAYERS))
    device = next(model.parameters()).device
    if gia:
        for i in GIA_LAYERS:
            model.model[i] = GCWrapped(model.model[i], GIA(ch[i]).to(device))
    if cdsa:
        for i in CDSA_LAYERS:
            model.model[i] = GCWrapped(model.model[i], CDSA(ch[i]).to(device))
    return model


def build_gc_model(cfg, nc: int, ch: int, weights=None, gia=False, cdsa=False, verbose=True, names=None):
    """Build GCDetectionModel. Plain pretrained weights load before the blocks are added; GIA/CDSA weights after.

    names are set before loading, as Ultralytics does, so the class head can remap by name.
    """
    model = GCDetectionModel(cfg, nc=nc, ch=ch, verbose=verbose)
    if names:
        model.names = names
    model.gc = {"gia": gia, "cdsa": cdsa}
    model.yaml["gc"] = dict(model.gc)
    weights_are_gc = weights is not None and isinstance(weights, nn.Module) \
        and any(isinstance(m, GCWrapped) for m in weights.model)
    if weights is not None and not weights_are_gc:
        model.load(weights)                      # COCO / plain YOLO11 weights into the original layout
    apply_gc(model, gia, cdsa)
    if weights_are_gc:
        model.load(weights)                      # resuming a GIA/CDSA checkpoint: same layout on both sides
    return model


def make_gc_trainer(gia: bool = False, cdsa: bool = False):
    """DetectionTrainer subclass that builds YOLO11 with the given GIA / CDSA blocks."""

    class GCTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if isinstance(cfg, dict):
                cfg = {k: v for k, v in cfg.items() if k != "gc"}
            names = self.data["names"] if getattr(self.args, "cls_remap", True) and self.data.get("names") else None
            return build_gc_model(cfg, self.data["nc"], self.data["channels"], weights, gia=gia, cdsa=cdsa,
                                  verbose=verbose and RANK == -1, names=names)

    GCTrainer.__name__ = f"GCTrainer(gia={gia},cdsa={cdsa})"
    return GCTrainer
