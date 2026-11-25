"""Scripted demo of the API, using FastAPI's TestClient (no server needed).

Sends a few valid and invalid requests and saves the responses to
results/api_examples.json as evidence that the deployment works.

    python -m part3_machine_learning.api_demo
"""

import json
import logging
import sys

from fastapi.testclient import TestClient

from part3_machine_learning import ml_common as mc
from part3_machine_learning.api import app

logger = logging.getLogger(__name__)

REQUESTS = [
    ("GET", "/health", None),
    ("POST", "/predict", {"date_time": "2018-10-02 08:00", "weather_main": "Clear", "temp_c": 12}),
    ("POST", "/predict", {"date_time": "2018-10-02 03:00", "weather_main": "Clear"}),
    ("POST", "/predict", {"date_time": "2018-12-25 08:00", "weather_main": "Snow", "temp_c": -8}),
    ("POST", "/recommend", {"date": "2018-10-02", "earliest": 6, "latest": 10, "window": 1}),
    ("POST", "/predict", {"date_time": "02/10/2018 8am", "weather_main": "Clear"}),     # invalid date
    ("POST", "/predict", {"date_time": "2018-10-02 08:00", "weather_main": "Hail"}),    # invalid weather
]


def main() -> int:
    client = TestClient(app)
    examples = []
    for method, path, body in REQUESTS:
        resp = client.get(path) if method == "GET" else client.post(path, json=body)
        data = resp.json()
        if path == "/recommend" and resp.status_code == 200:
            data = {k: v for k, v in data.items() if k != "hourly_forecast"}   # keep the file short
        examples.append({"request": f"{method} {path}", "body": body, "status": resp.status_code, "response": data})
        print(f"{method} {path}  {json.dumps(body) if body else ''}\n  -> {resp.status_code} "
              f"{json.dumps(data)[:200]}\n")
    out = mc.RESULTS_DIR / "api_examples.json"
    out.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    logger.info("Saved API examples to %s", out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
