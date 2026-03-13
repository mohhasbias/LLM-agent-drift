from src.drift_injector import apply_drift


def _base_scenario():
    return {
        "id": "s1",
        "english_env_info": "env",
        "english_tools": [
            {
                "type": "function",
                "function": {
                    "name": "book_reservation",
                    "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}},
                },
            }
        ],
        "english_tasks": ["Book a reservation."],
        "english_answer_list": [
            [
                {
                    "idx": 0,
                    "dependency_list": [],
                    "action": {"name": "book_reservation", "arguments": {"user_id": "u1"}},
                    "observation": "ok",
                }
            ]
        ],
        "english_task_types": ["test"],
    }


def test_schema_drift_renames_argument_keys():
    scenario, meta = apply_drift(_base_scenario(), "schema")
    args = scenario["english_answer_list"][0][0]["action"]["arguments"]
    assert "user_id_v2" in args
    assert "user_id" not in args
    assert "schema_rename_map" in meta


def test_toolset_drift_renames_tool_name():
    scenario, meta = apply_drift(_base_scenario(), "toolset")
    assert scenario["english_answer_list"][0][0]["action"]["name"] == "book_reservation__new"
    assert scenario["english_tools"][0]["function"]["name"] == "book_reservation__new"
    assert meta["tool_rename_map"]["book_reservation"] == "book_reservation__new"


def test_policy_drift_inserts_confirmation_step():
    scenario, meta = apply_drift(_base_scenario(), "policy")
    first = scenario["english_answer_list"][0][0]["action"]["name"]
    assert first == "confirm_policy_compliance"
    assert meta["required_first_tool"] == "confirm_policy_compliance"
