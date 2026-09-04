"""Pure-numpy helpers to turn a VLM 2D point into a scene object id.

These functions are intentionally free of Isaac Gym / torch dependencies so
they can be unit-tested standalone. The only domain fact they encode is the
segmentation-id convention used throughout FetchBench:

    seg_pixel_value == object_old_id + SEG_ID_OFFSET

(background == 0, robot == 1, table == 2, scene == 3, objects start at 4).
See `fetch_base._create_envs` (`seg_idx` starts at 1) and
`fetch_solution_base.py` where `goal_seg_id = task_obj_index + 4`.

--------------------------------------------------------------------------
Review fix #3 - the raw point must stay observable.
--------------------------------------------------------------------------
An earlier "snap everything" resolver snapped ANY point to the nearest candidate
centroid as a last resort. That made `resolve_rate == 1.0` unconditionally and
silently converted "the VLM pointed at a completely different object" into
"resolved to some candidate". Grounding error became invisible.

`resolve_point` now returns a full record including:
  * `raw_seg_val` / `raw_old_id` - what the point actually landed on, whether
    or not it is a candidate (negative ids mean background/robot/table/scene),
  * `point_on_candidate` - True only for a genuine direct hit,
  * `snapped`             - True when the answer came from centroid snapping.

`allow_centroid_snap=False` disables the snap entirely so a miss stays a miss.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# seg pixel value = old_id + 4  (mirrors fetch_solution_base.get_cgn_input)
SEG_ID_OFFSET = 4

# Human-readable names for the non-object seg values, so a raw point that
# landed on the table/robot/background is legible in the logs instead of
# showing up as a meaningless negative old_id.
_NON_OBJECT_SEG = {0: "background", 1: "robot", 2: "table", 3: "scene"}


def seg_to_old_id(seg_value: int, seg_offset: int = SEG_ID_OFFSET) -> int:
    return int(seg_value) - int(seg_offset)


def seg_value_name(seg_value: int) -> str:
    """'table' / 'robot' / ... for structural ids, 'obj_<n>' for objects."""
    v = int(seg_value)
    if v in _NON_OBJECT_SEG:
        return _NON_OBJECT_SEG[v]
    return "obj_{}".format(v - SEG_ID_OFFSET)


def _as_2d(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg)
    if seg.ndim == 3:  # HxWx1
        seg = seg[..., 0]
    return seg


def candidate_pixel_count(
    seg: np.ndarray,
    cand_old_ids: Sequence[int],
    seg_offset: int = SEG_ID_OFFSET,
) -> int:
    """Number of pixels in `seg` that belong to any candidate object."""
    seg = _as_2d(seg)
    if seg.size == 0 or len(cand_old_ids) == 0:
        return 0
    cand_seg_vals = np.asarray([int(c) + int(seg_offset) for c in cand_old_ids])
    return int(np.isin(seg, cand_seg_vals).sum())


def visible_pixel_count(
    seg: np.ndarray,
    old_id: int,
    seg_offset: int = SEG_ID_OFFSET,
) -> int:
    """Pixels belonging to one object in this view (review fix #5)."""
    seg = _as_2d(seg)
    if seg.size == 0:
        return 0
    return int((seg == int(old_id) + int(seg_offset)).sum())


def visible_object_old_ids(
    seg: np.ndarray,
    seg_offset: int = SEG_ID_OFFSET,
    min_pixels: int = 1,
) -> List[int]:
    """Every *object* old_id with at least `min_pixels` pixels in this view.

    Used by review fix #6: referring-expression ambiguity has to be judged
    against what is actually in the image, not against the candidate subset.
    """
    seg = _as_2d(seg)
    if seg.size == 0:
        return []
    vals, counts = np.unique(seg, return_counts=True)
    out = []
    for v, c in zip(vals.tolist(), counts.tolist()):
        if int(v) >= int(seg_offset) and c >= min_pixels:
            out.append(int(v) - int(seg_offset))
    return sorted(out)


def pick_camera(
    seg_cams: Sequence[np.ndarray],
    cand_old_ids: Sequence[int],
    seg_offset: int = SEG_ID_OFFSET,
    target_old_id: Optional[int] = None,
    max_cameras: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Choose the camera the VLM will look through.

    Review fix #9: `max_cameras` clips the search to the *sensor* cameras.
    `fetch_base._create_envs` appends one extra fixed third-person "vis" camera
    at [-2, 0, 2.5] after the `cfg.env.cam.num_cam` sensor cameras; with
    `num_cam: 1` (the GO configs) that stray view could otherwise win the
    coverage vote and silently become the perception input.

    Review fix #5: when `target_old_id` is given, cameras where the target is
    invisible are only used as a last resort, and target visibility is
    reported so it can be conditioned on downstream.

    Returns (camera_index, info).
    """
    cams = list(seg_cams)
    if max_cameras is not None:
        cams = cams[: int(max_cameras)]
    if not cams:
        return 0, {"n_cameras_considered": 0, "target_visible_px": 0}

    cand_counts, tgt_counts = [], []
    for seg in cams:
        seg2d = _as_2d(seg)
        cand_counts.append(candidate_pixel_count(seg2d, cand_old_ids, seg_offset))
        tgt_counts.append(
            visible_pixel_count(seg2d, target_old_id, seg_offset)
            if target_old_id is not None else -1
        )

    if target_old_id is not None and max(tgt_counts) > 0:
        # Prefer views where the instructed object is actually visible; among
        # those, keep the original "most candidate pixels" heuristic.
        eligible = [i for i, t in enumerate(tgt_counts) if t > 0]
        best_idx = max(eligible, key=lambda i: (cand_counts[i], tgt_counts[i]))
    else:
        best_idx = int(np.argmax(cand_counts)) if cand_counts else 0

    info = {
        "n_cameras_considered": len(cams),
        "cand_px_per_camera": [int(c) for c in cand_counts],
        "target_px_per_camera": [int(t) for t in tgt_counts],
        "target_visible_px": int(max(tgt_counts[best_idx], 0)),
        "target_visible": bool(tgt_counts[best_idx] > 0),
        "target_visible_in_any_camera": bool(max(tgt_counts) > 0)
        if target_old_id is not None else None,
    }
    return int(best_idx), info


def _candidate_centroids(
    seg: np.ndarray,
    cand_old_ids: Sequence[int],
    seg_offset: int,
) -> Dict[int, Tuple[float, float]]:
    """{old_id: (x_px, y_px)} centroid of each candidate's visible pixels."""
    centroids: Dict[int, Tuple[float, float]] = {}
    for old_id in cand_old_ids:
        ys, xs = np.where(seg == int(old_id) + int(seg_offset))
        if xs.size:
            centroids[int(old_id)] = (float(xs.mean()), float(ys.mean()))
    return centroids


def resolve_point(
    point_frac: Optional[Tuple[float, float]],
    seg: np.ndarray,
    cand_old_ids: Sequence[int],
    seg_offset: int = SEG_ID_OFFSET,
    window: int = 15,
    allow_centroid_snap: bool = True,
) -> Dict[str, Any]:
    """Map a VLM point (fractional x, y in [0, 1]) to a candidate object.

    Resolution ladder:
      1. `direct`          : the exact pixel under the point is a candidate.
      2. `window_vote`     : most frequent candidate in a (2*window+1) box -
                             tolerates a point that landed on a thin handle or
                             a 1-2 px silhouette edge.
      3. `nearest_centroid`: nearest candidate centroid. This is a *snap*, not
                             a detection: it succeeds even when the point sat
                             squarely on an unrelated object. Gated behind
                             `allow_centroid_snap` and always flagged.

    Returns a record; `old_id` is None when unresolved.
    """
    seg = _as_2d(seg)
    rec: Dict[str, Any] = {
        "old_id": None,
        "method": "no_point",
        "raw_seg_val": None,
        "raw_old_id": None,
        "raw_seg_name": None,
        "point_on_candidate": False,
        "snapped": False,
        "px": None,
        "py": None,
    }
    if point_frac is None:
        return rec

    h, w = seg.shape[:2]
    x_frac, y_frac = point_frac
    px = int(np.clip(round(float(x_frac) * (w - 1)), 0, w - 1))
    py = int(np.clip(round(float(y_frac) * (h - 1)), 0, h - 1))
    cand_set = {int(c) for c in cand_old_ids}

    raw_val = int(seg[py, px])
    raw_old = seg_to_old_id(raw_val, seg_offset)
    rec.update(
        px=px, py=py,
        raw_seg_val=raw_val,
        raw_old_id=int(raw_old),
        raw_seg_name=seg_value_name(raw_val),
    )

    # 1. direct hit
    if raw_old in cand_set:
        rec.update(old_id=int(raw_old), method="direct", point_on_candidate=True)
        return rec

    # 2. window vote
    y0, y1 = max(0, py - window), min(h, py + window + 1)
    x0, x1 = max(0, px - window), min(w, px + window + 1)
    patch = np.asarray(seg[y0:y1, x0:x1]).reshape(-1).astype(np.int64) - int(seg_offset)
    votes: Dict[int, int] = {}
    for oid in patch.tolist():
        if oid in cand_set:
            votes[oid] = votes.get(oid, 0) + 1
    if votes:
        best = max(votes.items(), key=lambda kv: kv[1])[0]
        rec.update(old_id=int(best), method="window_vote",
                   window_votes={int(k): int(v) for k, v in votes.items()})
        return rec

    # 3. nearest candidate centroid (snap - records the fact loudly)
    if not allow_centroid_snap:
        rec["method"] = "miss_no_snap"
        return rec

    centroids = _candidate_centroids(seg, cand_old_ids, seg_offset)
    if centroids:
        best = min(
            centroids.items(),
            key=lambda kv: (kv[1][0] - px) ** 2 + (kv[1][1] - py) ** 2,
        )[0]
        rec.update(old_id=int(best), method="nearest_centroid", snapped=True)
        return rec

    rec["method"] = "no_cand_pixels"
    return rec
