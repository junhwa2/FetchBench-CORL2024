"""VLM clients that map (RGB image + language instruction) -> a target.

Two backends share one interface so the rest of the pipeline never branches on
which is active:

  * GeminiERClient  : calls the Gemini Robotics-ER API (spatial pointing).
  * OracleVLMClient : no network; uses ground-truth segmentation to emit the
                      centroid of a chosen candidate. Lets the full Phase-1
                      pipeline (camera pick -> point -> seg resolution ->
                      target write) run and be validated with no API key.
                      `mode='random'` gives the chance-level baseline that
                      any reported grounding accuracy must be compared to.

Two query styles:
  * `point()`        : free-form pointing. Returns (x_frac, y_frac) in [0, 1],
                       or None when the model declined / failed.
  * `choose_index()` : the image already carries numbered markers on the
                       candidates; the model returns one marker number. This
                       removes both the seg-snapping ambiguity and the
                       "pointed at a visually identical non-candidate" failure
                       mode, at the cost of showing the model where to look.

`context` carries optional scene info (seg mask, candidate ids, gt target)
that only the oracle consumes.

--------------------------------------------------------------------------
Review fixes applied here
--------------------------------------------------------------------------
* Retry with exponential backoff on 429/5xx/timeouts. Previously a single
  transient API error fell through to `fallback: gt`, which the old scoring
  then counted as a correct answer.
* `finishReason` / `promptFeedback` are inspected, so a safety block or a
  MAX_TOKENS truncation reports a real error instead of a silent `None`.
* `num_samples > 1` enables self-consistency sampling (the caller votes).
* The scene description in the prompt is configurable instead of being
  hard-coded to "tabletop / shelf".
"""

import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class VLMCallError(RuntimeError):
    """Raised when the backend could not produce an answer at all."""


class BaseVLMClient:
    def point(
        self,
        image_rgb: np.ndarray,
        instruction: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[float, float]]:
        raise NotImplementedError

    def point_samples(
        self,
        image_rgb: np.ndarray,
        instruction: str,
        context: Optional[Dict[str, Any]] = None,
        n: int = 1,
    ) -> List[Optional[Tuple[float, float]]]:
        """`n` independent draws, for self-consistency voting."""
        return [self.point(image_rgb, instruction, context) for _ in range(max(1, int(n)))]

    def choose_index(
        self,
        image_rgb: np.ndarray,
        instruction: str,
        n_marks: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Gemini Robotics-ER
# --------------------------------------------------------------------------- #

_POINT_PROMPT = (
    "You are a robot vision system. {scene_description}\n"
    "Instruction: {instruction}\n"
    "Point at the single object the instruction refers to. Respond with ONLY a "
    'JSON list of the form [{{"point": [y, x], "label": "<n>"}}] where y and '
    "x are integers in 0-1000 normalized to the image height and width "
    "respectively. Return exactly one point."
)

_MARK_PROMPT = (
    "You are a robot vision system. {scene_description}\n"
    "Each graspable object is tagged with a numbered marker "
    "(1-{n_marks}).\n"
    "Instruction: {instruction}\n"
    "Decide which numbered marker is placed on the object the instruction "
    "refers to. Respond with ONLY a JSON object of the form "
    '{{"mark": <integer>}} using one of the numbers shown. '
    "Do not invent a number that is not in the image."
)

_DEFAULT_SCENE_DESCRIPTION = "The image shows a cluttered indoor scene viewed by a robot."

# HTTP statuses worth retrying: rate limit + transient server-side failures.
_RETRY_STATUS = (408, 429, 500, 502, 503, 504)


class GeminiERClient(BaseVLMClient):
    """Gemini Robotics-ER client via the Gemini REST API.

    Uses `requests` (not the google-genai SDK) so it works on Python 3.8 - the
    SDK requires Python >= 3.9, which this repo's pinned env does not have. Set
    the key via `api_key` or the GEMINI_API_KEY / GOOGLE_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "gemini-robotics-er-2-preview",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        thinking_budget: Optional[int] = 0,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        scene_description: str = _DEFAULT_SCENE_DESCRIPTION,
    ):
        self.model = model
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_base_delay = float(retry_base_delay)
        self.scene_description = scene_description or _DEFAULT_SCENE_DESCRIPTION
        # Populated per call so the task can log why an attempt failed.
        self.last_error: Optional[str] = None
        self.last_raw_text: Optional[str] = None

    # ------------------------------------------------------------------ #
    # plumbing
    # ------------------------------------------------------------------ #
    def _resolve_key(self) -> str:
        import os

        key = (self._api_key
               or os.environ.get("GEMINI_API_KEY")
               or os.environ.get("GOOGLE_API_KEY"))
        if not key:
            raise VLMCallError(
                "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY), "
                "pass vlm.api_key=..., or use vlm.backend=oracle."
            )
        return key

    @staticmethod
    def _encode_png(image_rgb: np.ndarray) -> bytes:
        img = np.ascontiguousarray(image_rgb[..., :3]).astype(np.uint8)
        try:
            from PIL import Image
            import io

            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            import imageio.v3 as iio

            return bytes(iio.imwrite("<bytes>", img, extension=".png"))

    def _request(self, image_rgb: np.ndarray, prompt: str,
                 temperature: Optional[float] = None) -> str:
        """POST one generateContent call, with retry/backoff. Returns raw text.

        Raises VLMCallError when every attempt failed - the caller must record
        that as an error, NOT silently fall back to the ground-truth target.
        """
        import base64
        import requests

        b64 = base64.b64encode(self._encode_png(image_rgb)).decode("ascii")
        gen_cfg: Dict[str, Any] = {
            "temperature": self.temperature if temperature is None else float(temperature)
        }
        if self.thinking_budget is not None:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": int(self.thinking_budget)}

        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": b64}},
                    {"text": prompt},
                ],
            }],
            "generationConfig": gen_cfg,
        }
        url = "{}/models/{}:generateContent".format(self.base_url, self.model)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._resolve_key(),
        }

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers,
                                     timeout=self.timeout)
            except Exception as exc:  # network/timeout
                last_exc = "{}: {}".format(type(exc).__name__, exc)
                resp = None

            if resp is not None and resp.status_code == 200:
                text, err = self._extract_text(resp.json())
                if err:
                    # A safety block or truncation is a real failure, not an
                    # empty answer to be quietly treated as "model declined".
                    last_exc = err
                    if "SAFETY" in err or "PROHIBITED" in err:
                        raise VLMCallError(err)  # deterministic; retrying is pointless
                else:
                    self.last_raw_text = text
                    return text
            elif resp is not None:
                last_exc = "HTTP {} for model={}: {}".format(
                    resp.status_code, self.model, resp.text[:300])
                if resp.status_code not in _RETRY_STATUS:
                    raise VLMCallError(last_exc)  # 400/401/404 - retrying is pointless

            if attempt < self.max_retries:
                delay = self.retry_base_delay * (2 ** attempt) * (0.5 + random.random())
                print("[GeminiER] attempt {}/{} failed ({}); retrying in {:.1f}s"
                      .format(attempt + 1, self.max_retries + 1, last_exc, delay))
                time.sleep(delay)

        raise VLMCallError("all {} attempts failed - last: {}"
                           .format(self.max_retries + 1, last_exc))

    @staticmethod
    def _extract_text(resp_json: dict) -> Tuple[str, Optional[str]]:
        """(text, error). Concatenates the text parts of the first candidate.

        Surfaces `promptFeedback.blockReason` and a non-STOP `finishReason` as
        an explicit error string so blocked/truncated responses are not
        mistaken for the model declining to answer.
        """
        feedback = resp_json.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            return "", "prompt blocked: {}".format(feedback["blockReason"])

        cands = resp_json.get("candidates") or []
        if not cands:
            return "", "no candidates in response"

        cand = cands[0]
        finish = cand.get("finishReason")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if finish and finish not in ("STOP", "MAX_TOKENS"):
            return text, "finishReason={}".format(finish)
        if finish == "MAX_TOKENS" and not text.strip():
            return text, "finishReason=MAX_TOKENS with empty text (raise thinking_budget?)"
        return text, None

    # ------------------------------------------------------------------ #
    # parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_json(text: str):
        import json
        import re

        if not text:
            return None
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        for pattern in (r"\[.*\]", r"\{.*\}"):
            m = re.search(pattern, cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    continue
        return None

    @classmethod
    def _parse_point(cls, text: str) -> Optional[Tuple[float, float]]:
        """Gemini pointing output -> fractional (x, y).

        Points come back as [y, x] with each coordinate in 0-1000 normalized
        to image height / width respectively.
        """
        data = cls._load_json(text)
        if data is None:
            return None
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list) or not data:
            return None
        first = data[0]
        pt = first.get("point") if isinstance(first, dict) else first
        if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
            return None
        try:
            y_norm, x_norm = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            return None
        return (float(np.clip(x_norm / 1000.0, 0.0, 1.0)),
                float(np.clip(y_norm / 1000.0, 0.0, 1.0)))

    @classmethod
    def _parse_mark(cls, text: str, n_marks: int) -> Optional[int]:
        import re

        data = cls._load_json(text)
        val = None
        if isinstance(data, dict):
            val = data.get("mark", data.get("index", data.get("id")))
        elif isinstance(data, list) and data:
            head = data[0]
            val = head.get("mark") if isinstance(head, dict) else head
        if val is None:
            m = re.search(r"\d+", text or "")
            val = m.group(0) if m else None
        try:
            idx = int(val)
        except (TypeError, ValueError):
            return None
        return idx if 1 <= idx <= int(n_marks) else None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def point(self, image_rgb, instruction, context=None):
        self.last_error = None
        prompt = _POINT_PROMPT.format(instruction=instruction,
                                      scene_description=self.scene_description)
        return self._parse_point(self._request(image_rgb, prompt))

    def point_samples(self, image_rgb, instruction, context=None, n=1):
        n = max(1, int(n))
        prompt = _POINT_PROMPT.format(instruction=instruction,
                                      scene_description=self.scene_description)
        out: List[Optional[Tuple[float, float]]] = []
        for i in range(n):
            # A strictly-greedy model returns the same answer n times, so
            # self-consistency needs a non-zero temperature after the first draw.
            temp = self.temperature if (n == 1 or i == 0) else max(self.temperature, 0.4)
            out.append(self._parse_point(self._request(image_rgb, prompt, temperature=temp)))
        return out

    def choose_index(self, image_rgb, instruction, n_marks, context=None):
        self.last_error = None
        prompt = _MARK_PROMPT.format(instruction=instruction, n_marks=int(n_marks),
                                     scene_description=self.scene_description)
        return self._parse_mark(self._request(image_rgb, prompt), n_marks)


# --------------------------------------------------------------------------- #
# Oracle (no-API) backend: plumbing check + chance-level baseline
# --------------------------------------------------------------------------- #

class OracleVLMClient(BaseVLMClient):
    """Emits the seg-centroid of a chosen candidate, using ground truth.

    mode:
      'gt'     -> always point at context['target_old_id'] (sanity: the whole
                  pipeline should reproduce the benchmark target).
      'random' -> point at a uniformly random *visible* candidate. This is the
                  chance-level baseline (review fix #4): any reported grounding
                  accuracy is only meaningful relative to this number, which is
                  roughly 1 / (number of visible candidates).
    """

    def __init__(self, mode: str = "gt", seg_offset: int = 4):
        assert mode in ("gt", "random")
        self.mode = mode
        self.seg_offset = seg_offset

    def _chosen_old_id(self, context) -> Optional[int]:
        seg = np.asarray(context["seg"])
        if seg.ndim == 3:
            seg = seg[..., 0]
        cand: Sequence[int] = list(context.get("cand_old_ids", []))
        if self.mode == "gt":
            return int(context["target_old_id"])
        seed = int(context.get("seed", 0))
        rng = np.random.default_rng(seed)
        pool = [c for c in cand if (seg == c + self.seg_offset).any()]
        if not pool:
            return None
        return int(rng.choice(pool))

    def point(self, image_rgb, instruction, context=None):
        if not context:
            return None
        seg = np.asarray(context["seg"])
        if seg.ndim == 3:
            seg = seg[..., 0]
        h, w = seg.shape[:2]

        chosen = self._chosen_old_id(context)
        if chosen is None:
            return None

        ys, xs = np.where(seg == chosen + self.seg_offset)
        if xs.size == 0:
            return None
        # Return an in-mask pixel (nearest mask pixel to the centroid) rather
        # than the raw centroid - for non-convex / occluded objects the centroid
        # can land on a neighbour, which would make the oracle "miss" its own
        # target. Nearest-in-mask guarantees the point is on `chosen`.
        cx, cy = float(xs.mean()), float(ys.mean())
        k = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        return (float(xs[k]) / (w - 1), float(ys[k]) / (h - 1))

    def choose_index(self, image_rgb, instruction, n_marks, context=None):
        """Marker index of the oracle's pick, using the caller's mark map."""
        if not context:
            return None
        chosen = self._chosen_old_id(context)
        mark_to_old = context.get("mark_to_old_id") or {}
        for mark, old in mark_to_old.items():
            if int(old) == int(chosen):
                return int(mark)
        return None


# --------------------------------------------------------------------------- #

def build_client(vlm_cfg: Optional[Dict[str, Any]]) -> BaseVLMClient:
    """Factory from a `solution.vlm` config dict."""
    vlm_cfg = dict(vlm_cfg or {})
    backend = str(vlm_cfg.get("backend", "gemini")).lower()
    if backend == "gemini":
        return GeminiERClient(
            model=vlm_cfg.get("model", "gemini-robotics-er-2-preview"),
            api_key=vlm_cfg.get("api_key") or None,
            temperature=float(vlm_cfg.get("temperature", 0.0)),
            thinking_budget=vlm_cfg.get("thinking_budget", 0),
            timeout=float(vlm_cfg.get("timeout", 60.0)),
            max_retries=int(vlm_cfg.get("max_retries", 3)),
            retry_base_delay=float(vlm_cfg.get("retry_base_delay", 2.0)),
            scene_description=vlm_cfg.get("scene_description")
            or _DEFAULT_SCENE_DESCRIPTION,
        )
    if backend == "oracle":
        return OracleVLMClient(
            mode=str(vlm_cfg.get("oracle_mode", "gt")),
            seg_offset=int(vlm_cfg.get("seg_offset", 4)),
        )
    raise ValueError("Unknown vlm.backend={!r} (expected 'gemini' or 'oracle')"
                     .format(backend))
