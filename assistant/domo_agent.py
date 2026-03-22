from assistant.audit import log_event
from assistant.runtime import execute_tool_call, plan_domo_action
from assistant.schemas import PlannedToolCall


def plan_domo(user_input: str):
    return plan_domo_action(user_input)


def run_domo_tool(tool_call: PlannedToolCall):
    result = execute_tool_call(tool_call)

    if hasattr(result, "__iter__") and not isinstance(result, str):
        for chunk in result:
            yield chunk
    else:
        yield str(result)

    log_event(
        "tool_execution_finished",
        request_id=tool_call.request_id,
        tool_name=tool_call.tool_name,
    )
