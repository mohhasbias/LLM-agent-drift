"""Tests for run_task_success_evaluation helpers."""


def test_execute_call_normalize_true_inverts_tool_name():
    """With normalize=True (default), drifted tool names are inverted."""
    from src.run_task_success_evaluation import _normalize_tool_name
    drift_meta = {"tool_rename_map": {"cancel_order": "terminate_order"}}
    # inverse: terminate_order -> cancel_order
    assert _normalize_tool_name("terminate_order", drift_meta) == "cancel_order"


def test_normalize_args_strips_v2_suffix():
    """With normalization, _v2 suffixed keys are stripped."""
    from src.run_task_success_evaluation import _normalize_tool_args
    drift_meta = {"schema_rename_map": {"order_id": "order_id_v2"}}
    result = _normalize_tool_args({"order_id_v2": "W123"}, drift_meta)
    assert result == {"order_id": "W123"}
