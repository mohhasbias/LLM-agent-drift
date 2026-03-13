from src.learning_memory import LearningMemory


def test_learning_memory_gated_retrieval():
    mem = LearningMemory(min_support=2, min_confidence=0.4)
    task = "You are Yusuf Rossi and want to exchange order item."
    expected = {"name": "exchange_delivered_order_items", "arguments": {"order_id": "#W1"}}

    # One observation is not enough support.
    mem.observe(task_text=task, drift_type="toolset", expected_action=expected, success=False)
    assert mem.retrieve_hint(task_text=task, drift_type="toolset") is None

    # Second observation reaches support and confidence threshold.
    mem.observe(task_text=task, drift_type="toolset", expected_action=expected, success=True)
    hint = mem.retrieve_hint(task_text=task, drift_type="toolset")
    assert hint is not None
    assert hint["tool_name"] == "exchange_delivered_order_items"


def test_learning_memory_distinguishes_drift_types():
    mem = LearningMemory(min_support=1, min_confidence=0.0)
    task = "same task text for both drift types"
    expected = {"name": "foo", "arguments": {}}
    mem.observe(task_text=task, drift_type="schema", expected_action=expected, success=True)
    assert mem.retrieve_hint(task_text=task, drift_type="schema") is not None
    assert mem.retrieve_hint(task_text=task, drift_type="policy") is None


def test_learning_memory_prefers_higher_confidence_action_for_collided_key():
    mem = LearningMemory(min_support=1, min_confidence=0.0)
    task = "User asks for return flow and does not remember email."
    action_a = {"name": "return_delivered_order_items", "arguments": {"order_id": "x"}}
    action_b = {"name": "modify_pending_order_items", "arguments": {"order_id": "x"}}
    mem.observe(task_text=task, drift_type="schema", expected_action=action_a, success=True)
    mem.observe(task_text=task, drift_type="schema", expected_action=action_a, success=False)
    mem.observe(task_text=task, drift_type="schema", expected_action=action_b, success=True)

    hint = mem.retrieve_hint(task_text=task, drift_type="schema")
    assert hint is not None
    assert hint["tool_name"] == "modify_pending_order_items"


def test_learning_memory_fuzzy_retrieval():
    mem = LearningMemory(min_support=1, min_confidence=0.0)
    train_task = (
        "You are Yusuf and want to return everything but tablet in delivered order."
    )
    eval_task = (
        "You are Yusuf Gonzalez and want to return all items except tablet from delivered order."
    )
    action = {
        "name": "return_delivered_order_items",
        "arguments": {"order_id": "#1", "item_ids": ["i1"], "payment_method_id": "p1"},
    }
    mem.observe(task_text=train_task, drift_type="toolset", expected_action=action, success=True)
    hint = mem.retrieve_hint(task_text=eval_task, drift_type="toolset")
    assert hint is not None
    assert hint["tool_name"] == "return_delivered_order_items"
    assert hint["key_similarity"] > 0.25


def test_learning_memory_supports_step_specific_retrieval():
    mem = LearningMemory(min_support=1, min_confidence=0.0)
    task = "User wants to update a pending order after reviewing product variants."
    read_action = {"name": "get_order_details", "arguments": {"order_id": "#1"}}
    write_action = {
        "name": "modify_pending_order_items",
        "arguments": {"order_id_v2": "#1", "item_ids_v2": ["a"], "new_item_ids_v2": ["b"]},
    }

    mem.observe(task_text=task, drift_type="schema", expected_action=read_action, success=True, step=0)
    mem.observe(task_text=task, drift_type="schema", expected_action=write_action, success=True, step=4)

    step0 = mem.retrieve_step_hint(task_text=task, drift_type="schema", step=0)
    step4 = mem.retrieve_step_hint(task_text=task, drift_type="schema", step=4)

    assert step0 is not None
    assert step0["tool_name"] == "get_order_details"
    assert step4 is not None
    assert step4["tool_name"] == "modify_pending_order_items"
