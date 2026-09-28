"""Load test for the retrieval-only endpoint (no LLM in the loop).

    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 -t 2m --csv loadtest/results/search
"""

import json
import random
from pathlib import Path

from locust import FastHttpUser, between, task

QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"
QUESTIONS = [
    json.loads(line)["question"]
    for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]


class SearchUser(FastHttpUser):
    wait_time = between(0.1, 0.5)

    @task(4)
    def search(self) -> None:
        payload = {"query": random.choice(QUESTIONS), "k": 5}
        with self.client.post("/search", json=payload, catch_response=True) as response:
            if response.status_code != 200 or not response.json()["results"]:
                response.failure(f"unexpected response: {response.status_code}")

    @task(1)
    def search_by_category(self) -> None:
        payload = {"query": random.choice(QUESTIONS), "k": 3, "category": "solo"}
        self.client.post("/search", json=payload, name="/search?category")
