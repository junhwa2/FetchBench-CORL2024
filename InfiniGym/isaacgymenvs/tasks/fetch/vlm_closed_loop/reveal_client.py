"""Reveal-reasoning VLM client for the VLM+VORM closed loop (ThinkGrasp-style).

New file — edits nothing. The stock `GeminiERClient._POINT_PROMPT` only asks the
model to point at the referred object; it does not reason about occlusion. This
subclass adds a prompt + `decide()` that make the model choose ONE next action:

  grasp_target    -> the target is visible/graspable now; point at the target.
  remove_occluder -> the target is hidden/blocked; point at the object to move
                     away FIRST to reveal it (NOT the target, which may be
                     invisible).

The model returns this decision directly as structured JSON, so the closed loop
no longer needs the server-side category-match proxy to tell target from
occluder. This is the reveal reasoning VORM cannot do: with a fully-occluded
target there is no point cloud, so only the VLM can name what to remove.
"""

from typing import Any, Dict, Optional

from ..utils.vlm_target.gemini_er_client import (
    GeminiERClient,
    build_client,
    _DEFAULT_SCENE_DESCRIPTION,
)

_REVEAL_PROMPT = (
    "You are the vision system of a robot arm that must PICK UP one object from "
    "a cluttered scene. {scene_description}\n"
    "Target to pick up: {instruction}\n"
    "\n"
    "How to choose what to point at:\n"
    "1) The target, and anything you may point at, is a SINGLE discrete movable "
    "object resting in the scene (a bottle, box, mug, tool, toy, small piece of "
    "furniture, etc.). NEVER point at the table top, the floor, a shelf, a wall, "
    "the background, or the robot itself.\n"
    "2) Match the target by shape, colour and typical real-world appearance. "
    "Interpret the name as the physical object (e.g. 'desktop' = a desktop "
    "computer/monitor object, NOT the surface of a desk). If several objects "
    "could match, choose the single most likely one.\n"
    "3) Then choose exactly ONE next action:\n"
    "   - \"grasp_target\": you can clearly and confidently see the target and it "
    "is not blocked -> point at the target object itself.\n"
    "   - \"remove_occluder\": the target is hidden, mostly covered, OR you cannot "
    "confidently identify it in the image -> point at the single movable object "
    "most likely sitting on top of / in front of it that should be removed first. "
    "Do NOT guess a random object as the target, and do NOT point at the target, "
    "the table, or the floor.\n"
    "\n"
    "First list the movable objects you see, decide which is the target, then "
    "answer with ONLY this JSON object:\n"
    "{{\"visible_objects\": \"<comma-separated movable objects you see>\", "
    "\"target_found\": true | false, "
    "\"action\": \"grasp_target\" | \"remove_occluder\", \"point\": [y, x], "
    "\"reason\": \"<one short sentence>\"}}\n"
    "The point MUST land on the chosen object's body. y and x are integers in "
    "0-1000, normalized to the image height and width respectively. Give exactly "
    "one point."
)


class RevealERClient(GeminiERClient):
    """GeminiERClient + a reveal-reasoning `decide()` returning action + point.

    Reuses the parent's REST plumbing, retry/backoff, JSON extraction and
    [y,x]-0-1000 -> fractional point parsing verbatim.
    """

    def decide(self, image_rgb, instruction, context=None) -> Dict[str, Any]:
        """-> {action: 'grasp_target'|'remove_occluder'|None,
                point: (x_frac, y_frac)|None, reason: str|None, raw: str}."""
        self.last_error = None
        prompt = _REVEAL_PROMPT.format(
            instruction=instruction, scene_description=self.scene_description)
        text = self._request(image_rgb, prompt, temperature=self.temperature)

        point = self._parse_point(text)          # dict{"point":[y,x]} handled
        data = self._load_json(text)
        if isinstance(data, list) and data:
            data = data[0]
        action, reason, seen, found = None, None, None, None
        if isinstance(data, dict):
            a = str(data.get("action", "")).strip().lower()
            if "grasp" in a or "target" in a:
                action = "grasp_target"
            elif "remove" in a or "occlud" in a or "block" in a:
                action = "remove_occluder"
            reason = data.get("reason")
            seen = data.get("visible_objects")
            found = data.get("target_found")
            # If the model itself says it did not find the target, treat this as
            # an occluder-removal step regardless of the action string — it must
            # not be scored/executed as a confident target grasp.
            if found is False and action == "grasp_target":
                action = "remove_occluder"
        return {"action": action, "point": point, "reason": reason,
                "visible_objects": seen, "target_found": found, "raw": text}


def build_reveal_client(vlm_cfg: Optional[Dict[str, Any]]):
    """Reveal client when backend=gemini and vlm.reveal is on; otherwise defer
    to the stock factory (oracle / reveal-off have no `decide`, so the server
    falls back to plain pointing + the category proxy)."""
    cfg = dict(vlm_cfg or {})
    backend = str(cfg.get("backend", "gemini")).lower()
    if backend == "gemini" and bool(cfg.get("reveal", True)):
        return RevealERClient(
            model=cfg.get("model", "gemini-robotics-er-2-preview"),
            api_key=cfg.get("api_key") or None,
            temperature=float(cfg.get("temperature", 0.0)),
            thinking_budget=cfg.get("thinking_budget", 0),
            timeout=float(cfg.get("timeout", 60.0)),
            max_retries=int(cfg.get("max_retries", 3)),
            retry_base_delay=float(cfg.get("retry_base_delay", 2.0)),
            scene_description=cfg.get("scene_description") or _DEFAULT_SCENE_DESCRIPTION,
        )
    return build_client(cfg)
