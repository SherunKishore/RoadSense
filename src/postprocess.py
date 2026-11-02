"""Post-processing for road-damage detections.

Boxes are numpy arrays in original-image pixels: boxes (N, 4) xyxy, scores (N,), classes (N,).

- tta_views(): test-time augmentation views (original, horizontal flip, 1.25x larger network
  input) with the function that maps boxes back. Ultralytics' built-in augment=True downscales
  (1/0.83/0.67), which loses thin cracks, so we upscale instead.
- weighted_boxes_fusion(): fuse boxes from several views/models (Solovyev et al., 2021).
- merge_crack_fragments(): join collinear pieces of one crack into a single box, the way
  annotators draw long cracks (D00 runs along the road, D10 across it).
- best_f1_thresholds(): per-class confidence thresholds for deployment.
"""
import numpy as np

LONGITUDINAL, TRANSVERSE = 0, 1


# ---------------------------------------------------------------- test-time augmentation
def tta_views(img: np.ndarray, scales=(1.25,), flip: bool = True):
    """Yield (view_image, unmap, size_mult) triples.

    size_mult scales the network input size for that view (1.25 -> 640 px becomes 800 px), which
    is how the upscale view keeps extra detail. unmap(boxes) maps view boxes back to img pixels.
    """
    w = img.shape[1]
    yield img, (lambda b: b), 1.0
    if flip:
        def unflip(b):
            b = b.copy()
            b[:, [0, 2]] = w - b[:, [2, 0]]
            return b
        yield img[:, ::-1].copy(), unflip, 1.0
    for s in scales:
        yield img, (lambda b: b), s


# ---------------------------------------------------------------- box utilities
def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


# ---------------------------------------------------------------- weighted boxes fusion
def weighted_boxes_fusion(views, n_views: int, iou_thr: float = 0.55, skip_thr: float = 0.0,
                          return_sources: bool = False):
    """Fuse a list of (boxes, scores, classes) from n_views sources.

    Boxes of the same class that overlap by more than iou_thr are averaged, weighted by score.
    The fused score is the mean score scaled by how many sources agreed (WBF 'avg' mode).
    With return_sources=True a 4th array gives, per fused box, how many different views contributed.
    """
    boxes = np.concatenate([v[0] for v in views]) if views else np.zeros((0, 4))
    scores = np.concatenate([v[1] for v in views]) if views else np.zeros(0)
    classes = np.concatenate([v[2] for v in views]) if views else np.zeros(0)
    src = np.concatenate([np.full(len(v[1]), k) for k, v in enumerate(views)]) if views else np.zeros(0)
    keep = scores >= skip_thr
    boxes, scores, classes, src = boxes[keep], scores[keep], classes[keep], src[keep]
    out_b, out_s, out_c, out_n = [], [], [], []
    for c in np.unique(classes):
        m = classes == c
        b, s, v = boxes[m], scores[m], src[m]
        order = np.argsort(-s)
        fused, members = [], []  # fused box per cluster, list of (box, score, view) per cluster
        for i in order:
            if fused:
                ious = iou_one_to_many(b[i], np.asarray(fused))
                j = int(ious.argmax())
                if ious[j] > iou_thr:
                    members[j].append((b[i], s[i], v[i]))
                    mb = np.array([x[0] for x in members[j]]); ms = np.array([x[1] for x in members[j]])
                    fused[j] = (mb * ms[:, None]).sum(0) / ms.sum()
                    continue
            fused.append(b[i].astype(np.float64))
            members.append([(b[i], s[i], v[i])])
        for fb, mem in zip(fused, members):
            ms = np.array([x[1] for x in mem])
            out_b.append(fb)
            out_s.append(ms.mean() * min(len(mem), n_views) / n_views)
            out_c.append(c)
            out_n.append(len({x[2] for x in mem}))
    if not out_b:
        empty = (np.zeros((0, 4)), np.zeros(0), np.zeros(0))
        return (*empty, np.zeros(0, int)) if return_sources else empty
    fused = (np.asarray(out_b), np.asarray(out_s), np.asarray(out_c))
    return (*fused, np.asarray(out_n)) if return_sources else fused


# ---------------------------------------------------------------- crack fragment merging
def _collinear(a: np.ndarray, b: np.ndarray, axis: int, gap: float, overlap: float = 0.5) -> bool:
    """True if a and b line up along `axis` (1 = vertical run, 0 = horizontal run) with a small gap."""
    o = 1 - axis  # the cross axis must overlap
    cross = min(a[o + 2], b[o + 2]) - max(a[o], b[o])
    narrower = min(a[o + 2] - a[o], b[o + 2] - b[o])
    if cross < overlap * narrower:
        return False
    along_gap = max(a[axis], b[axis]) - min(a[axis + 2], b[axis + 2])
    return along_gap <= gap


def merge_crack_fragments(boxes, scores, classes, img_hw, gap_frac: float = 0.04, min_conf: float = 0.1):
    """Merge collinear same-class crack boxes (D00 vertically, D10 horizontally).

    Only boxes with score >= min_conf are merged; the merged box is the union and keeps the
    highest score. Other classes and low-confidence boxes are returned unchanged.
    """
    h, w = img_hw
    keep = np.ones(len(boxes), dtype=bool)
    new_b, new_s, new_c = [], [], []
    for cls, axis, size in ((LONGITUDINAL, 1, h), (TRANSVERSE, 0, w)):
        idx = [i for i in np.where((classes == cls) & (scores >= min_conf))[0]]
        if len(idx) < 2:
            continue
        groups = [[i] for i in idx]
        merged = True
        while merged:  # repeat until no two groups line up
            merged = False
            for gi in range(len(groups)):
                for gj in range(gi + 1, len(groups)):
                    ub_i = _union(boxes[groups[gi]]); ub_j = _union(boxes[groups[gj]])
                    if iou_one_to_many(ub_i, ub_j[None])[0] > 0 or _collinear(ub_i, ub_j, axis, gap_frac * size):
                        groups[gi] += groups.pop(gj)
                        merged = True
                        break
                if merged:
                    break
        for g in groups:
            if len(g) > 1:
                keep[g] = False
                new_b.append(_union(boxes[g])); new_s.append(scores[g].max()); new_c.append(cls)
    if not new_b:
        return boxes, scores, classes
    return (np.concatenate([boxes[keep], np.asarray(new_b)]),
            np.concatenate([scores[keep], np.asarray(new_s)]),
            np.concatenate([classes[keep], np.asarray(new_c)]))


def _union(b: np.ndarray) -> np.ndarray:
    return np.array([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()])


# ---------------------------------------------------------------- deployment thresholds
def best_f1_thresholds(f1_curve: np.ndarray, conf_axis: np.ndarray, class_ids) -> dict:
    """Per-class confidence that maximises F1 (from ultralytics ap_per_class curves)."""
    return {int(c): round(float(conf_axis[f1_curve[i].argmax()]), 3) for i, c in enumerate(class_ids)}
