import json


def parse_router_response(response: str) -> dict:
    return json.loads(response)
