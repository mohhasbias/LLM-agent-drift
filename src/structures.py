# src/structures.py
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    matched: bool
    raw_output: str

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "matched": self.matched,
            "raw_output": self.raw_output,
        }


@dataclass
class TaskResult:
    task_id: str
    task_type: str  # single / multi / clarify / chat
    predicted_calls: list[ToolCall] = field(default_factory=list)
    ground_truth_calls: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "predicted_calls": [tc.to_dict() for tc in self.predicted_calls],
            "ground_truth_calls": self.ground_truth_calls,
            "completed": self.completed,
        }


@dataclass
class ScenarioResult:
    scenario_id: str
    condition: str  # 4-bit string e.g. "0110"
    model: str
    tasks: list[TaskResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "condition": self.condition,
            "model": self.model,
            "tasks": [t.to_dict() for t in self.tasks],
        }
