"""Lightweight learning memory with gated write policy and fuzzy retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class MemoryEntry:
    pattern_key: str
    drift_type: str
    support: int = 0
    positive: int = 0
    action_stats: dict[tuple[str, tuple[str, ...]], tuple[int, int]] | None = None
    step_action_stats: dict[int, dict[tuple[str, tuple[str, ...]], tuple[int, int]]] | None = None

    @property
    def confidence(self) -> float:
        if self.support == 0:
            return 0.0
        return self.positive / self.support


class LearningMemory:
    def __init__(self, min_support: int = 2, min_confidence: float = 0.4) -> None:
        self.min_support = min_support
        self.min_confidence = min_confidence
        self._entries: dict[tuple[str, str], MemoryEntry] = {}

    @staticmethod
    def _normalize_for_signature(task_text: str) -> list[str]:
        # Drop explicit drift notices to reduce cross-scenario collisions.
        clean = re.sub(r"\[Drift Notice\][^\n]*", " ", task_text, flags=re.IGNORECASE)
        clean = re.sub(r"\[Policy Update\][^\n]*", " ", clean, flags=re.IGNORECASE)
        # Normalize user-specific surface forms to keep a task-intent signature.
        norm_tokens: list[str] = []
        for raw in clean.split():
            tok = raw.strip(".,:;!?()[]{}'\"").lower()
            if not tok:
                continue
            if "@" in tok:
                tok = "<email>"
            elif re.fullmatch(r"[#a-z]*\d+[a-z0-9_-]*", tok):
                tok = "<id>"
            elif tok.isdigit():
                tok = "<num>"
            norm_tokens.append(tok)
        return norm_tokens

    @classmethod
    def _pattern_key(cls, task_text: str) -> str:
        norm_tokens = cls._normalize_for_signature(task_text)
        if not norm_tokens:
            return ""
        # Keep both head and tail context: identity + explicit request.
        if len(norm_tokens) <= 24:
            sig = norm_tokens
        else:
            sig = norm_tokens[:16] + norm_tokens[-8:]
        return " ".join(sig)

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return inter / union

    def observe(
        self,
        task_text: str,
        drift_type: str,
        expected_action: dict[str, Any],
        success: bool,
        step: int = 0,
    ) -> None:
        """Update entry stats from one episode outcome."""
        action_name = str(expected_action.get("name", ""))
        args = expected_action.get("arguments", {})
        arg_keys = tuple(sorted(args.keys())) if isinstance(args, dict) else tuple()
        key = (self._pattern_key(task_text), drift_type)
        entry = self._entries.get(key)
        if entry is None:
            entry = MemoryEntry(
                pattern_key=key[0],
                drift_type=drift_type,
                action_stats={},
                step_action_stats={},
            )
            self._entries[key] = entry
        if entry.action_stats is None:
            entry.action_stats = {}
        if entry.step_action_stats is None:
            entry.step_action_stats = {}
        action_key = (action_name, arg_keys)
        action_support, action_positive = entry.action_stats.get(action_key, (0, 0))
        action_support += 1
        if success:
            action_positive += 1
        entry.action_stats[action_key] = (action_support, action_positive)
        step_bucket = entry.step_action_stats.setdefault(step, {})
        step_support, step_positive = step_bucket.get(action_key, (0, 0))
        step_support += 1
        if success:
            step_positive += 1
        step_bucket[action_key] = (step_support, step_positive)
        entry.support += 1
        if success:
            entry.positive += 1

    def _best_action_for_entry(
        self,
        entry: MemoryEntry,
        *,
        step: int | None = None,
    ) -> tuple[str, list[str]] | None:
        stats = entry.action_stats
        if step is not None and entry.step_action_stats:
            stats = entry.step_action_stats.get(step)
        if not stats:
            return None
        best_key: tuple[str, tuple[str, ...]] | None = None
        best_score = -1.0
        for action_key, (support, positive) in stats.items():
            conf = (positive / support) if support > 0 else 0.0
            # Confidence first, then support.
            score = conf * 1000 + support
            if score > best_score:
                best_score = score
                best_key = action_key
        if best_key is None:
            return None
        return best_key[0], list(best_key[1])

    def retrieve_step_hint(self, task_text: str, drift_type: str, step: int = 0) -> dict[str, Any] | None:
        query_key = self._pattern_key(task_text)
        key = (query_key, drift_type)
        entry = self._entries.get(key)
        best_similarity = 1.0

        if entry is None:
            # Fuzzy fallback over same drift type.
            query_tokens = set(self._normalize_for_signature(task_text))
            best_entry: MemoryEntry | None = None
            best_similarity = 0.0
            for (pat, drift), cand in self._entries.items():
                if drift != drift_type:
                    continue
                sim = self._jaccard(query_tokens, set(pat.split()))
                if sim > best_similarity:
                    best_similarity = sim
                    best_entry = cand
            if best_entry is None or best_similarity < 0.25:
                return None
            entry = best_entry

        if entry is None:
            return None
        if entry.support < self.min_support:
            return None
        if entry.confidence < self.min_confidence:
            return None
        best = self._best_action_for_entry(entry, step=step)
        if best is None and step != 0:
            best = self._best_action_for_entry(entry)
        if best is None:
            return None
        tool_name, arg_keys = best
        return {
            "tool_name": tool_name,
            "arg_keys": arg_keys,
            "support": entry.support,
            "confidence": round(entry.confidence, 4),
            "key_similarity": round(best_similarity, 4),
        }

    def retrieve_hint(self, task_text: str, drift_type: str) -> dict[str, Any] | None:
        return self.retrieve_step_hint(task_text, drift_type, step=0)

    def stats(self) -> dict[str, Any]:
        rows = []
        for entry in self._entries.values():
            best = self._best_action_for_entry(entry)
            tool_name, arg_keys = ("", [])
            if best is not None:
                tool_name, arg_keys = best
            rows.append(
                {
                    "pattern_key": entry.pattern_key,
                    "drift_type": entry.drift_type,
                    "tool_name": tool_name,
                    "arg_keys": arg_keys,
                    "support": entry.support,
                    "positive": entry.positive,
                    "confidence": round(entry.confidence, 4),
                    "num_actions": len(entry.action_stats or {}),
                    "num_step_buckets": len(entry.step_action_stats or {}),
                }
            )
        return {
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
            "entries": rows,
        }
