"""Production guardrail pipeline utilities for Part 5.

This module provides:
- Regex-based input filtering (hard blocklist)
- Calibrated model probability inference
- A three-layer moderation decision pipeline
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


BLOCKLIST: dict[str, list[re.Pattern[str]]] = {
    "direct_threat": [
        re.compile(r"\b(?:i\s*(?:will|'ll)|we\s*(?:will|'ll))\s+(?:kill|murder|shoot|stab|hurt)\s+you\b", re.IGNORECASE),
        re.compile(r"\b(?:kill|murder|shoot|stab|hurt)\s+you\b", re.IGNORECASE),
        re.compile(r"\byou\s+(?:deserve|need)\s+to\s+(?:die|be\s+killed|be\s+shot|be\s+stabbed|be\s+hurt)\b", re.IGNORECASE),
        re.compile(r"\b(?:going\s+to|gonna)\s+(?:kill|murder|shoot|stab|hurt)\s+you\b", re.IGNORECASE),
        re.compile(r"\b(?:kill|murder|shoot|stab|hurt)\s+your\s+family\b", re.IGNORECASE),
    ],
    "self_harm_directed": [
        re.compile(r"\b(?:kill|hurt|cut|hang)\s+yourself\b", re.IGNORECASE),
        re.compile(r"\byou\s+should\s+(?:die|kill\s+yourself|hurt\s+yourself)\b", re.IGNORECASE),
        re.compile(r"\bgo\s+(?:die|hurt\s+yourself|jump\s+off\s+a\s+bridge)\b", re.IGNORECASE),
        re.compile(r"\b(?:end|take)\s+your\s+life\b", re.IGNORECASE),
    ],
    "doxxing_stalking": [
        re.compile(r"\bi\s+(?:know|found)\s+where\s+you\s+live\b", re.IGNORECASE),
        re.compile(r"\bi\s+(?:have|got)\s+your\s+(?:address|phone\s*number)\b", re.IGNORECASE),
        re.compile(r"\b(?:we\s+are|we're)\s+coming\s+to\s+your\s+(?:house|home)\b", re.IGNORECASE),
        re.compile(r"\b(?:i\s+am|i'm)\s+watching\s+you\b", re.IGNORECASE),
    ],
    "dehumanization": [
        re.compile(r"\b(?:not|isn't|aren't|are\s+not)\s+(?:human|people|person)\b", re.IGNORECASE),
        re.compile(r"\b(?:these|those)\s+(?:human|people|person)\s+are\s+(?:animals|vermin|trash)\b", re.IGNORECASE),
        re.compile(r"\b(?:human|people|person)\s+like\s+you\s+are\s+(?:filth|garbage|subhuman)\b", re.IGNORECASE),
        re.compile(r"\b(?:exterminate|remove)\s+(?:that|these|those)?\s*(?:human|people|person)\b", re.IGNORECASE),
    ],
    "coordinated_harassment": [
        re.compile(r"(?=.*\beveryone\b)(?=.*\b(report|dogpile|harass)\b)(?=.*\bthem\b)", re.IGNORECASE),
        re.compile(r"\blet'?s\s+all\s+(?:report|mass\s*report|harass|spam)\s+\w+", re.IGNORECASE),
        re.compile(r"\bjoin\s+me\s+to\s+(?:harass|target|attack)\s+\w+", re.IGNORECASE),
    ],
}


def input_filter(text: str) -> dict[str, Any] | None:
    """Run regex blocklist matching on raw user input.

    Returns a standardized block decision when a pattern matches, else None.
    """
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {
                    "decision": "block",
                    "layer": "input_filter",
                    "category": category,
                    "confidence": 1.0,
                }
    return None


def batch_predict_probs(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int = 64,
    max_length: int = 128,
    device: str | None = None,
) -> np.ndarray:
    """Return model toxic probabilities for class 1 using softmax logits."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    probs_out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) for t in texts[i : i + batch_size]]
            encoded = tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            probs_out.append(probs.detach().cpu().numpy())

    if not probs_out:
        return np.array([], dtype=float)
    return np.concatenate(probs_out).astype(float)


@dataclass
class ModerationPipeline:
    """Three-layer moderation policy.

    Layer 1: hard regex blocklist
    Layer 2: calibrated model with auto block/allow thresholds
    Layer 3: human review for uncertain cases
    """

    model: Any
    tokenizer: Any
    calibrator: Any
    block_threshold: float = 0.6
    allow_threshold: float = 0.4

    def _raw_prob(self, text: str) -> float:
        probs = batch_predict_probs(self.model, self.tokenizer, [text], batch_size=1)
        return float(probs[0])

    def _calibrated_prob(self, raw_prob: float) -> float:
        calibrated = self.calibrator.predict(np.array([raw_prob], dtype=float))
        return float(np.clip(calibrated[0], 0.0, 1.0))

    def predict(self, text: str) -> dict[str, Any]:
        """Run layered moderation and return a structured decision dict."""
        layer1 = input_filter(text)
        if layer1 is not None:
            return layer1

        raw_prob = self._raw_prob(text)
        calibrated_prob = self._calibrated_prob(raw_prob)

        if calibrated_prob >= self.block_threshold:
            return {
                "decision": "block",
                "layer": "model_block",
                "category": "model_toxicity",
                "confidence": calibrated_prob,
            }

        if calibrated_prob <= self.allow_threshold:
            return {
                "decision": "allow",
                "layer": "model_allow",
                "category": "safe",
                "confidence": 1.0 - calibrated_prob,
            }

        return {
            "decision": "review",
            "layer": "human_review",
            "category": "uncertain",
            "confidence": 1.0 - abs(calibrated_prob - 0.5) * 2.0,
        }
