"""VLM target-selection utilities (Phase 1 of the VLM+VORM pipeline).

Given an RGB image + a language instruction, a VLM (Gemini Robotics-ER) points
at the referred object; `target_resolver` maps that point to a scene object id
via the segmentation mask. Kept free of Isaac Gym imports so it is unit-testable.
"""

from .gemini_er_client import (
    BaseVLMClient,
    GeminiERClient,
    OracleVLMClient,
    VLMCallError,
    build_client,
)
from .target_resolver import (
    SEG_ID_OFFSET,
    candidate_pixel_count,
    pick_camera,
    resolve_point,
    seg_to_old_id,
    seg_value_name,
    visible_object_old_ids,
    visible_pixel_count,
)

__all__ = [
    # clients
    "BaseVLMClient",
    "GeminiERClient",
    "OracleVLMClient",
    "VLMCallError",
    "build_client",
    # resolver
    "SEG_ID_OFFSET",
    "candidate_pixel_count",
    "pick_camera",
    "resolve_point",
    "seg_to_old_id",
    "seg_value_name",
    "visible_object_old_ids",
    "visible_pixel_count",
]
