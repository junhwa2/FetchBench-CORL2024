"""VLM target-selection pass (Phase 1 of the VLM+VORM pipeline).

This task NEVER touches the singulation planner. It reuses the parent's scene,
camera and label-lookup machinery to do exactly one job per task_idx:

    render RGB+seg  ->  a VLM points at (or picks the marker on) the instructed
    object  ->  seg-map that answer to a scene object id  ->  record it.

Phase 2 then runs `FetchMeshCuroboGORunOverride` (a no-edit subclass) with
`pred_folder=<VORM>` and `target_override=<targets.npz>`, which plans on the
VLM's target while the KB comes from VORM's `collision_pred`.

Registration is done here (module import mutates the shared
`isaacgym_task_map`), so `isaacgymenvs/tasks/__init__.py` is left untouched.

===========================================================================
Review fixes applied in this file
===========================================================================
#2  `correct` is now STRICT: a task that fell back to the ground-truth target
    (API error, unresolvable point) counts as WRONG. The old lenient number is
    kept separately as `correct_lenient` for comparison, never as the headline.
#3  The raw seg id under the VLM's point is recorded, so "pointed at the right
    object" (`point_on_gt`) is separable from "snapped onto some candidate".
#5  Target visibility in the chosen camera is measured and logged, so accuracy
    can be conditioned on the instructed object actually being on screen.
#6  Referring-expression ambiguity (the left/right qualifier) is judged against
    every object VISIBLE IN THE IMAGE, not just the candidate subset. Otherwise
    the instruction can be genuinely ambiguous and the VLM gets blamed for it.
#7  `write_task_config` defaults to False. The benchmark dataset under
    $ASSET_PATH is no longer mutated; Phase 2 reads a sidecar targets file.
#8  `settle_steps` defaults to `solution.init_steps` (60), so the VLM sees the
    same settled scene the planner will act on.
#9  Camera choice is restricted to the `cfg.env.cam.num_cam` sensor cameras,
    excluding the trailing fixed third-person "vis" camera.
Plus: `seg_offset` honoured in the debug overlay, cached label lookup,
self-consistency voting, and an opt-in numbered-marker prompting mode.
"""

import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .fetch_mesh_curobo_go_run import FetchMeshCuroboGORun
from .utils.vlm_target import (
    SEG_ID_OFFSET,
    VLMCallError,
    build_client,
    pick_camera,
    resolve_point,
    seg_value_name,
    visible_object_old_ids,
    visible_pixel_count,
)


class FetchMeshCuroboGORunVLM(FetchMeshCuroboGORun):
    """Phase-1 VLM target selector. `solve()` records the target and returns no
    frames (the eval loop skips video for an empty list)."""

    # ------------------------------------------------------------------ #
    # config helpers
    # ------------------------------------------------------------------ #
    def _vlm_cfg(self) -> Dict[str, Any]:
        return dict(self.cfg["solution"].get("vlm", {}) or {})

    def _vlm_client(self):
        # cache one client across all task_idx of the scene
        if getattr(self, "_vlm_client_obj", None) is None:
            self._vlm_client_obj = build_client(self._vlm_cfg())
        return self._vlm_client_obj

    def _vlm_log_dir(self) -> str:
        exp = self.cfg.get("experiment_name", "vlm_target")
        d = os.path.join("vlm_target_logs", str(exp))
        os.makedirs(d, exist_ok=True)
        return d

    # ------------------------------------------------------------------ #
    # candidates / ground truth
    # ------------------------------------------------------------------ #
    def _candidate_old_ids(self, env_idx: int, task_idx: int) -> List[int]:
        cand = self.task_cand_obj_index[env_idx][task_idx]
        try:
            cand = cand.tolist()
        except AttributeError:
            cand = list(cand)
        return [int(c) for c in cand if int(c) >= 0]

    def _reference_task_obj_index(self, env_idx: int = 0):
        """Original (pre-VLM) task_obj_index from the .vlm_bak snapshot, if any.

        Only relevant in the legacy `write_task_config=True` mode, where
        `save_planner_target_to_config` overwrites the live task_obj_index and
        the scoring reference would otherwise drift across repeated runs. With
        the default `write_task_config=False` the dataset is never touched, so
        the in-memory array stays the true benchmark target and this returns
        None immediately.
        """
        cache = getattr(self, "_ref_toi_cache", None)
        if cache is None:
            cache = self._ref_toi_cache = {}
        if env_idx in cache:
            return cache[env_idx]
        arr = None
        if bool(self._vlm_cfg().get("write_task_config", False)):
            bak = self._scene_task_config_path(env_idx) + ".vlm_bak"
            if os.path.isfile(bak):
                try:
                    arr = np.asarray(
                        np.load(bak, allow_pickle=True)["task_obj_index"]).reshape(-1)
                except Exception as exc:
                    print("[GORunVLM] .vlm_bak read failed ({}: {}); "
                          "using live task_obj_index as gt"
                          .format(type(exc).__name__, exc))
        cache[env_idx] = arr
        return arr

    def _gt_target_old(self, env_idx: int, task_idx: int) -> int:
        ref = self._reference_task_obj_index(env_idx)
        if ref is not None and 0 <= task_idx < len(ref):
            return int(ref[task_idx])
        v = self.task_obj_index[env_idx][task_idx]
        try:
            return int(v.item())
        except AttributeError:
            return int(v)

    # ------------------------------------------------------------------ #
    # semantic object names (category) + spatial disambiguation
    # ------------------------------------------------------------------ #
    def _scene_asset_config_path(self, env_idx: int = 0) -> str:
        return os.path.join(
            os.environ["ASSET_PATH"], "Task",
            self.cfg["task"]["scene_config_path"][env_idx],
            "asset_config.json",
        )

    def _object_categories(self, env_idx: int = 0) -> Dict[int, str]:
        """{old_id: category} parsed from asset_config.json object_config paths
        (e.g. .../benchmark_objects/Telephone/<id>/mesh.urdf -> 'Telephone').
        Cached per env since the scene is fixed for a run."""
        cache = getattr(self, "_obj_cat_cache", None)
        if cache is None:
            cache = self._obj_cat_cache = {}
        if env_idx in cache:
            return cache[env_idx]
        cats: Dict[int, str] = {}
        try:
            with open(self._scene_asset_config_path(env_idx)) as f:
                ac = json.load(f)
            oc = ac.get("object_config", []) or []
            # `object_config` holds ONLY rigid objects, indexed 0..n_rigid-1.
            # old_id (== seg_id - 4 == task_obj_index) indexes the full actor
            # list, which fetch_base._create_envs builds with combo actors
            # (2 per combo) BEFORE the rigids. When combos are present, cats
            # would be offset by 2*n_combo and every category label (instruction
            # AND chosen_label) would be wrong. Every benchmark_eval scene here
            # has combo_config == 0 (verified), so the 1:1 mapping is correct.
            # We only WARN for a future combo-bearing scene set rather than
            # guessing the (untested) actor ordering. Diagnose with
            # scripts/check_object_id_alignment.py.
            n_combo = len(ac.get("combo_config", []) or [])
            if n_combo > 0:
                print("[GORunVLM][WARN] scene has {} combo(s); object_config "
                      "category mapping may be offset by 2*n_combo vs old_id. "
                      "Verify with scripts/check_object_id_alignment.py before "
                      "trusting category-based instructions/labels."
                      .format(n_combo))
            for i, e in enumerate(oc):
                root = str(e.get("asset_root", "")).replace("\\", "/")
                cat = ""
                for anchor in ("/benchmark_objects/", "/objects/"):
                    if anchor in root:
                        cat = root.split(anchor)[-1].split("/")[0]
                        break
                cats[i] = re.sub(r"^infinigen_", "", cat)
        except Exception as exc:
            print("[GORunVLM] category load failed ({}: {}) - falling back to "
                  "object_labels".format(type(exc).__name__, exc))
        cache[env_idx] = cats
        return cats

    def _lookup_obj_category(self, old_id: int, env_idx: int = 0) -> str:
        """Raw category string (e.g. 'Telephone'); '' if unavailable."""
        return self._object_categories(env_idx).get(int(old_id), "")

    def _lookup_obj_label(self, old_id, env_idx: int = 0) -> str:
        """Cached override of the parent lookup, which re-opened
        rearrange_config.npz on every single call."""
        cache = getattr(self, "_obj_label_cache", None)
        if cache is None:
            cache = self._obj_label_cache = {}
        key = (int(env_idx), int(old_id))
        if key not in cache:
            cache[key] = super(FetchMeshCuroboGORunVLM, self)._lookup_obj_label(
                old_id, env_idx)
        return cache[key]

    @staticmethod
    def _humanize(cat: str) -> str:
        """'DeskLamp' -> 'desk lamp', 'BeerBottle' -> 'beer bottle'."""
        s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", cat)
        s = s.replace("_", " ").strip()
        return s.lower() if s else cat

    # Same-object categories that differ only in spelling across asset sources.
    # Used ONLY to decide whether two objects count as duplicates (so the
    # left/right qualifier fires); the displayed name still comes from the
    # target's own category. Keep conservative - only merge true synonyms.
    _CATEGORY_SYNONYMS = {
        "plantcontainer": "pottedplant",
        "plant": "pottedplant",
        "pottedplant": "pottedplant",
    }

    @classmethod
    def _normalize_category(cls, cat: str) -> str:
        """Case/spacing-insensitive key for duplicate detection (e.g. 'Vase' and
        'vase' -> 'vase'; 'plantcontainer'/'PottedPlant' -> 'pottedplant')."""
        s = re.sub(r"[^a-z0-9]", "", str(cat).lower())
        return cls._CATEGORY_SYNONYMS.get(s, s)

    @staticmethod
    def _spatial_qualifier(rank: int, n: int) -> Optional[str]:
        """Left/right descriptor for the `rank`-th of `n` same-category objects
        ordered left->right in the image."""
        if n <= 1:
            return None
        if n == 2:
            return ["left", "right"][rank]
        if n == 3:
            return ["left", "middle", "right"][rank]
        if rank == 0:
            return "leftmost"
        if rank == n - 1:
            return "rightmost"
        ordinals = ["first", "second", "third", "fourth", "fifth",
                    "sixth", "seventh", "eighth", "ninth", "tenth"]
        ord_word = ordinals[rank] if rank < len(ordinals) else "{}th".format(rank + 1)
        return "{}-from-left".format(ord_word)

    def _build_category_label(
        self, gt_old: int, cand_old_ids: List[int], seg: np.ndarray,
        seg_offset: int, env_idx: int, normalize: bool = True,
        ambiguity_scope: str = "visible", min_visible_px: int = 30,
    ) -> Tuple[str, Dict[str, Any]]:
        """Instruction label = humanized category + a left/right qualifier when
        another object of the same category is present.

        Review fix #6 - `ambiguity_scope`:
          'visible'   (default) duplicates are judged over every object with at
                      least `min_visible_px` pixels in the chosen view. This is
                      what the VLM actually sees, so the resulting referring
                      expression is unambiguous *in the image*.
          'candidate' legacy behaviour: only candidate objects are considered.
                      A visually identical non-candidate look-alike then makes
                      the instruction ambiguous and the VLM is scored as wrong
                      for a defect in the prompt.
        """
        tgt_cat = self._lookup_obj_category(gt_old, env_idx)
        if not tgt_cat:
            # no category info -> keep the coarse benchmark label
            return self._lookup_obj_label(gt_old, env_idx), {
                "category": None, "qualifier": None, "ambiguity_scope": ambiguity_scope}

        human = self._humanize(tgt_cat)
        cats = self._object_categories(env_idx)
        key = (lambda c: self._normalize_category(c)) if normalize else (lambda c: c)
        tgt_key = key(tgt_cat)

        if ambiguity_scope == "candidate":
            scope_ids = [int(o) for o in cand_old_ids]
        else:
            scope_ids = visible_object_old_ids(seg, seg_offset,
                                               min_pixels=int(min_visible_px))
            if int(gt_old) not in scope_ids:
                scope_ids = scope_ids + [int(gt_old)]

        same = [o for o in scope_ids
                if cats.get(int(o), "") and key(cats.get(int(o), "")) == tgt_key]

        info: Dict[str, Any] = {
            "category": tgt_cat,
            "normalized": bool(normalize),
            "ambiguity_scope": ambiguity_scope,
            "n_same_category": len(same),
            "n_same_category_among_candidates": len(
                [o for o in cand_old_ids
                 if cats.get(int(o), "") and key(cats.get(int(o), "")) == tgt_key]),
            "same_category_old_ids": [int(o) for o in same],
            "qualifier": None,
        }
        if len(same) <= 1:
            return human, info

        # order the same-category objects left->right by seg centroid x
        xs: Dict[int, float] = {}
        for o in same:
            _, xcol = np.where(seg == int(o) + seg_offset)
            if xcol.size:
                xs[int(o)] = float(xcol.mean())
        if int(gt_old) not in xs:
            # The instructed object is not on screen: no spatial qualifier can
            # be grounded. Flagged so these tasks can be excluded from scoring.
            info["note"] = "target not visible in seg; qualifier skipped"
            info["ambiguous_unqualified"] = True
            return human, info

        ordered = sorted(xs.keys(), key=lambda o: xs[o])
        rank = ordered.index(int(gt_old))
        qualifier = self._spatial_qualifier(rank, len(ordered))
        info.update(qualifier=qualifier, rank=int(rank),
                    order_left_to_right=[int(o) for o in ordered])
        label = "{} {}".format(qualifier, human) if qualifier else human
        return label, info

    # ------------------------------------------------------------------ #
    # numbered-marker (set-of-mark) prompting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _draw_candidate_marks(rgb: np.ndarray, seg: np.ndarray,
                              cand_old_ids: List[int], seg_offset: int
                              ) -> Tuple[np.ndarray, Dict[int, int]]:
        """Overlay 1..N numbered discs on the visible candidates.

        Returns (marked_rgb, {mark_number: old_id}). Marks are numbered
        left-to-right so the numbering is itself interpretable.
        """
        from PIL import Image, ImageDraw

        centroids = []
        for o in cand_old_ids:
            ys, xs = np.where(seg == int(o) + int(seg_offset))
            if xs.size:
                centroids.append((float(xs.mean()), float(ys.mean()), int(o)))
        centroids.sort(key=lambda t: t[0])

        img = Image.fromarray(
            np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(img)
        mark_to_old: Dict[int, int] = {}
        r = max(9, int(min(img.size) * 0.022))
        for i, (cx, cy, old) in enumerate(centroids, start=1):
            mark_to_old[i] = old
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            txt = str(i)
            try:
                tw = draw.textlength(txt)
            except AttributeError:  # older Pillow
                tw = 6 * len(txt)
            draw.text((cx - tw / 2.0, cy - r * 0.6), txt, fill=(0, 0, 0))
        return np.asarray(img), mark_to_old

    # ------------------------------------------------------------------ #
    # main pass
    # ------------------------------------------------------------------ #
    def solve(self):
        assert self.num_envs == 1, "GORunVLM assumes num_envs == 1 (per-scene task_config)"
        env_idx = 0
        task_idx = self.get_task_idx()
        vlm_cfg = self._vlm_cfg()
        seg_offset = int(vlm_cfg.get("seg_offset", SEG_ID_OFFSET))

        # Review fix #8 - settle for as long as the planner does (init_steps),
        # so the VLM reasons over the same at-rest scene the planner acts on.
        settle = vlm_cfg.get("settle_steps", None)
        if settle is None:
            settle = self.cfg["solution"].get("init_steps", 60)
        for _ in range(int(settle)):
            self.env_physics_step()
            self.post_phy_step()

        gt_target_old = self._gt_target_old(env_idx, task_idx)
        cand_old_ids = self._candidate_old_ids(env_idx, task_idx)

        # ---- render + camera choice (review fixes #5, #9) ------------------
        rgb_all, seg_all = self.get_camera_image(rgb=True, seg=True)
        rgb_cams = rgb_all[env_idx]
        seg_cams = [np.asarray(s) for s in seg_all[env_idx]]

        num_sensor_cams = int(self.cfg["env"]["cam"].get("num_cam", len(seg_cams)))
        max_cams = None if bool(vlm_cfg.get("use_vis_camera", False)) else num_sensor_cams
        cam_idx, cam_info = pick_camera(
            seg_cams, cand_old_ids, seg_offset,
            target_old_id=gt_target_old, max_cameras=max_cams,
        )
        rgb = np.ascontiguousarray(
            np.asarray(rgb_cams[cam_idx])[..., :3]).astype(np.uint8)
        seg = seg_cams[cam_idx]
        if seg.ndim == 3:
            seg = seg[..., 0]

        gt_visible_px = visible_pixel_count(seg, gt_target_old, seg_offset)
        visible_cands = [c for c in cand_old_ids
                         if visible_pixel_count(seg, c, seg_offset) > 0]

        # ---- instruction ---------------------------------------------------
        label_source = str(vlm_cfg.get("label_source", "category"))
        if label_source == "category":
            target_label, disamb = self._build_category_label(
                gt_target_old, cand_old_ids, seg, seg_offset, env_idx,
                normalize=bool(vlm_cfg.get("normalize_category", True)),
                ambiguity_scope=str(vlm_cfg.get("ambiguity_scope", "visible")),
                min_visible_px=int(vlm_cfg.get("min_visible_px", 30)),
            )
        else:
            target_label, disamb = self._lookup_obj_label(gt_target_old, env_idx), {}

        template = vlm_cfg.get("instruction_template", "Pick up the {label}.")
        instruction = template.format(label=target_label)

        # ---- query the VLM -------------------------------------------------
        prompt_mode = str(vlm_cfg.get("prompt_mode", "point"))
        num_samples = max(1, int(vlm_cfg.get("num_samples", 1)))
        client = self._vlm_client()

        context: Dict[str, Any] = {
            "seg": seg,
            "cand_old_ids": cand_old_ids,
            "target_old_id": gt_target_old,
            "seed": task_idx,
        }

        vlm_error: Optional[str] = None
        point_frac: Optional[Tuple[float, float]] = None
        samples: List[Dict[str, Any]] = []
        chosen_old: Optional[int] = None
        method = "no_point"
        resolve_rec: Dict[str, Any] = {}
        mark_to_old: Dict[int, int] = {}
        marked_rgb = None

        if prompt_mode == "mark":
            # Set-of-mark: number the candidates in the image and let the model
            # return an index. No seg snapping is involved, so the answer is
            # unambiguous by construction.
            marked_rgb, mark_to_old = self._draw_candidate_marks(
                rgb, seg, cand_old_ids, seg_offset)
            context["mark_to_old_id"] = mark_to_old
            try:
                mark = client.choose_index(marked_rgb, instruction,
                                           len(mark_to_old), context=context)
            except (VLMCallError, Exception) as exc:
                mark = None
                vlm_error = "{}: {}".format(type(exc).__name__, exc)
                print("[GORunVLM] task {}: VLM call failed -> {}".format(task_idx, vlm_error))
            if mark is not None and int(mark) in mark_to_old:
                chosen_old = int(mark_to_old[int(mark)])
                method = "mark"
            else:
                method = "mark_invalid" if vlm_error is None else "vlm_error"
            resolve_rec = {"mark": mark, "mark_to_old_id":
                           {int(k): int(v) for k, v in mark_to_old.items()}}
        else:
            try:
                pts = client.point_samples(rgb, instruction, context=context,
                                           n=num_samples)
            except (VLMCallError, Exception) as exc:
                pts = [None] * num_samples
                vlm_error = "{}: {}".format(type(exc).__name__, exc)
                print("[GORunVLM] task {}: VLM call failed -> {}".format(task_idx, vlm_error))

            allow_snap = bool(vlm_cfg.get("allow_centroid_snap", True))
            window = int(vlm_cfg.get("resolve_window", 15))
            for p in pts:
                rec = resolve_point(p, seg, cand_old_ids, seg_offset,
                                    window=window, allow_centroid_snap=allow_snap)
                rec["point_frac"] = list(p) if p is not None else None
                samples.append(rec)

            # Self-consistency: majority vote over the resolved ids.
            votes: Dict[int, int] = {}
            for rec in samples:
                if rec["old_id"] is not None:
                    votes[int(rec["old_id"])] = votes.get(int(rec["old_id"]), 0) + 1
            if votes:
                chosen_old = max(votes.items(), key=lambda kv: kv[1])[0]
                winner = next(r for r in samples
                              if r["old_id"] is not None and int(r["old_id"]) == chosen_old)
            else:
                winner = samples[0] if samples else {}
            resolve_rec = dict(winner)
            method = resolve_rec.get("method", "no_point")
            point_frac = (tuple(resolve_rec["point_frac"])
                          if resolve_rec.get("point_frac") else None)
            if num_samples > 1:
                resolve_rec["vote_counts"] = {int(k): int(v) for k, v in votes.items()}

        # ---- fallback (review fix #2) --------------------------------------
        # A fallback keeps the sweep alive but is NEVER a correct answer.
        vlm_chosen_old = chosen_old
        fallback = str(vlm_cfg.get("fallback", "gt"))
        used_fallback = False
        if chosen_old is None or chosen_old not in cand_old_ids:
            used_fallback = True
            vlm_chosen_old = None
            if fallback == "gt":
                chosen_old = gt_target_old
            elif fallback == "first_cand" and cand_old_ids:
                chosen_old = cand_old_ids[0]
            else:
                chosen_old = None

        raw_old = resolve_rec.get("raw_old_id")
        point_on_gt = (raw_old is not None and int(raw_old) == int(gt_target_old))
        # STRICT correctness: the VLM itself had to land on the gt object.
        correct = (vlm_chosen_old is not None
                   and int(vlm_chosen_old) == int(gt_target_old))
        correct_lenient = (chosen_old is not None
                           and int(chosen_old) == int(gt_target_old))

        log: Dict[str, Any] = {
            "task_idx": int(task_idx),
            "scene": self.cfg["task"]["scene_config_path"][env_idx],
            "instruction": instruction,
            "prompt_mode": prompt_mode,
            "label_source": label_source,
            "instruction_label": target_label,
            "target_category": disamb.get("category"),
            "disambiguation": disamb,
            "gt_target_old": int(gt_target_old),
            "cand_old_ids": [int(c) for c in cand_old_ids],
            "n_candidates": len(cand_old_ids),
            # chance-level denominator (review fix #4)
            "n_candidates_visible": len(visible_cands),
            # target visibility (review fix #5)
            "gt_visible_px": int(gt_visible_px),
            "target_visible": bool(gt_visible_px > 0),
            "target_visible_in_any_camera": cam_info.get("target_visible_in_any_camera"),
            "cam_idx": int(cam_idx),
            "cam_info": cam_info,
            "vlm_point_frac": list(point_frac) if point_frac is not None else None,
            # raw pointing diagnostics (review fix #3)
            "raw_seg_val": resolve_rec.get("raw_seg_val"),
            "raw_old_id": raw_old,
            "raw_seg_name": (seg_value_name(resolve_rec["raw_seg_val"])
                             if resolve_rec.get("raw_seg_val") is not None else None),
            "point_on_gt": bool(point_on_gt),
            "point_on_candidate": bool(resolve_rec.get("point_on_candidate", False)),
            "snapped": bool(resolve_rec.get("snapped", False)),
            "resolve_method": method,
            "num_samples": num_samples,
            "vote_counts": resolve_rec.get("vote_counts"),
            "used_fallback": bool(used_fallback),
            "vlm_error": vlm_error,
            "vlm_chosen_old": None if vlm_chosen_old is None else int(vlm_chosen_old),
            "chosen_old": None if chosen_old is None else int(chosen_old),
            "correct": bool(correct),
            "correct_lenient": bool(correct_lenient),
        }

        if chosen_old is None:
            print("[GORunVLM] task {}: could not resolve a target "
                  "(method={}, fallback={}) - nothing recorded"
                  .format(task_idx, method, fallback))
            self._write_vlm_log(log, marked_rgb if marked_rgb is not None else rgb,
                                seg, point_frac, chosen_old, vlm_cfg, seg_offset)
            return [], log

        if label_source == "category":
            chosen_label = (self._lookup_obj_category(int(chosen_old), env_idx)
                            or self._lookup_obj_label(int(chosen_old), env_idx))
        else:
            chosen_label = self._lookup_obj_label(int(chosen_old), env_idx)
        log["chosen_label"] = chosen_label

        # ---- persistence (review fix #7) -----------------------------------
        # By default the benchmark dataset is NOT modified. `vlm_target_gen.py`
        # writes a targets.npz sidecar that Phase 2 reads via
        # `task.solution.target_override=...`.
        if bool(vlm_cfg.get("write_task_config", False)):
            if bool(vlm_cfg.get("backup_task_config", True)):
                self._backup_task_config_once(env_idx)
            self.save_planner_target_to_config(int(chosen_old), chosen_label, env_idx)

        print("[GORunVLM] task {}: instruction={!r} -> obj_old={} ('{}') "
              "[method={}, fallback={}, gt={}, correct={}, "
              "point_on_gt={}, gt_px={}]"
              .format(task_idx, instruction, chosen_old, chosen_label, method,
                      used_fallback, gt_target_old, correct, point_on_gt,
                      int(gt_visible_px)))

        self._write_vlm_log(log, marked_rgb if marked_rgb is not None else rgb,
                            seg, point_frac, chosen_old, vlm_cfg, seg_offset)
        return [], log

    # ------------------------------------------------------------------ #
    # persistence / debugging
    # ------------------------------------------------------------------ #
    def _backup_task_config_once(self, env_idx: int) -> None:
        path = self._scene_task_config_path(env_idx)
        bak = path + ".vlm_bak"
        if os.path.isfile(path) and not os.path.isfile(bak):
            try:
                shutil.copy2(path, bak)
                print("[GORunVLM] backed up {} -> {}".format(path, bak))
            except Exception as exc:
                print("[GORunVLM] backup failed ({}: {})"
                      .format(type(exc).__name__, exc))

    def _write_vlm_log(self, log, rgb, seg, point_frac, chosen_old, vlm_cfg,
                       seg_offset=SEG_ID_OFFSET) -> None:
        d = self._vlm_log_dir()
        scene_tag = str(log["scene"]).replace("/", "_")
        stem = "{}_t{}".format(scene_tag, log["task_idx"])
        with open(os.path.join(d, stem + ".json"), "w") as f:
            json.dump(log, f, indent=2, default=str)

        if not bool(dict(vlm_cfg or {}).get("save_debug_image", False)):
            return
        try:
            from PIL import Image, ImageDraw

            img = Image.fromarray(
                np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)).convert("RGB")
            draw = ImageDraw.Draw(img)
            h, w = np.asarray(rgb).shape[:2]
            if point_frac is not None:
                px, py = int(point_frac[0] * (w - 1)), int(point_frac[1] * (h - 1))
                r = 8
                draw.ellipse([px - r, py - r, px + r, py + r],
                             outline=(255, 0, 0), width=3)
            # green = chosen, blue = ground truth. Uses the CONFIGURED offset
            # (the old code hard-coded 4 and drew the wrong box when
            # seg_offset was overridden).
            for old_id, color in ((chosen_old, (0, 255, 0)),
                                  (log.get("gt_target_old"), (0, 128, 255))):
                if old_id is None:
                    continue
                ys, xs = np.where(seg == int(old_id) + int(seg_offset))
                if xs.size:
                    draw.rectangle([int(xs.min()), int(ys.min()),
                                    int(xs.max()), int(ys.max())],
                                   outline=color, width=3)
            img.save(os.path.join(d, stem + ".png"))
        except Exception as exc:
            print("[GORunVLM] debug image write failed ({}: {})"
                  .format(type(exc).__name__, exc))


# --- self-register so tasks/__init__.py stays untouched ------------------- #
try:
    from isaacgymenvs.tasks import isaacgym_task_map

    isaacgym_task_map["FetchMeshCuroboGORunVLM"] = FetchMeshCuroboGORunVLM
except Exception as _exc:  # pragma: no cover
    print("[GORunVLM] task registration deferred: {}: {}"
          .format(type(_exc).__name__, _exc))
