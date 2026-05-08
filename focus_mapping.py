#!/usr/bin/env python3
"""专注度映射与估计工具。"""

from __future__ import annotations

import json
import os
from typing import Dict, List

DEFAULT_EMOTION_SCORE_MAP = {
    "neutral": 90.0,
    "surprise": 95.0,
    "happy": 75.0,
    "sad": 60.0,
    "fear": 30.0,
    "angry": 20.0,
    "disgust": 20.0,
    "unknown": 60.0,
}


def normalize_emotion_label(emotion_label: str) -> str:
    label = (emotion_label or "").lower()
    for k in DEFAULT_EMOTION_SCORE_MAP:
        if k in label:
            return k
    return "unknown"


class FocusEstimator:
    def __init__(self, score_map: Dict[str, float] | None = None):
        self.score_map = dict(DEFAULT_EMOTION_SCORE_MAP)
        if score_map:
            for k, v in score_map.items():
                self.score_map[str(k).lower()] = float(v)

    @classmethod
    def from_json_path(cls, path: str | None):
        if not path or not os.path.exists(path):
            return cls(), False
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(cfg.get("emotion_score_map") or {}), True

    def emotion_to_score(self, emotion_label: str) -> float:
        key = normalize_emotion_label(emotion_label)
        return float(self.score_map.get(key, self.score_map["unknown"]))

    def weighted_score_from_preds(self, preds: List[dict]) -> float:
        if not preds:
            return 0.0
        weighted_sum = 0.0
        conf_sum = 0.0
        for item in preds:
            label = item.get("label", "")
            conf = float(item.get("score", 0.0))
            weighted_sum += self.emotion_to_score(label) * conf
            conf_sum += conf
        if conf_sum <= 0:
            return 0.0
        return weighted_sum / conf_sum
