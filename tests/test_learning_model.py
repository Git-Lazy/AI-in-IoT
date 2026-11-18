"""Learned preferences, piece 2: the decision-tree models and their API."""

from __future__ import annotations

import random

from app.learning import LEAVE
from app.learning.model import MIN_SAMPLES, _clock, learner, train_one
from tests.test_platform import _setup_house, client  # noqa: F401  (fixture)


def _rows(n: int = 60, seed: int = 0) -> list[tuple[dict, str]]:
    """Light on at 60% after 6pm, otherwise left alone -- with no noise."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        f = {"hour": round(rng.uniform(8, 23), 2), "weekend": rng.randint(0, 1),
             "temp_f": rng.uniform(65, 80), "others_home": 0}
        out.append((f, "ON:60" if f["hour"] >= 18 else LEAVE))
    return out


def test_tree_finds_the_habit_and_beats_the_baseline():
    tree, accuracy, baseline = train_one(_rows())
    assert accuracy >= 0.9 and accuracy > baseline
    assert tree.predict([[20, 0, 70, 0]])[0] == "ON:60"
    assert tree.predict([[10, 0, 70, 0]])[0] == LEAVE


def test_clock_wording():
    assert _clock(18.05) == "6pm"
    assert _clock(7.5) == "7:30am"
    assert _clock(0.1) == "12am"


def test_demo_history_trains_explainable_models(client):
    ids = _setup_house(client)
    h = ids["admin"]
    assert client.post("/api/learning/demo-history", json={"weeks": 6}).status_code == 401

    r = client.post("/api/learning/demo-history", json={"weeks": 6}, headers=h)
    assert r.status_code == 201 and r.json()["added"] >= MIN_SAMPLES
    assert r.json()["models"] == 4                     # 2 lights, AC, TV -- never the lock or garage

    models = {m["device"]: m for m in client.get("/api/learning/models").json()["models"]}
    assert set(models) == {"Entry Light", "Living Room Light", "AC", "TV"}
    tv = models["TV"]
    assert tv["accuracy"] > tv["baseline"]
    assert any(r["when"] == ["on the weekend"] and r["setting"] == "ON" for r in tv["rules"])

    # Weekday, 8pm, hot: living room medium, entry bright, AC cooling, TV on -- and says why.
    preds = client.get(f"/api/learning/predict/{ids['sam']}",
                       params={"hour": 20, "weekend": False, "temp_f": 84}).json()["predictions"]
    by = {p["device"]: p for p in preds}
    assert by["Living Room Light"]["setting"] == "ON:60"
    assert by["Living Room Light"]["because"] == ["after 6pm", "on a weekday"]
    assert by["Entry Light"]["setting"] == "ON:100"
    assert by["AC"]["setting"].startswith("cool")
    assert by["TV"]["setting"] == "ON"
    assert all(0 < p["confidence"] <= 1 and p["support"] >= 3 for p in preds)

    # Weekday morning, mild: nothing to do.
    preds = client.get(f"/api/learning/predict/{ids['sam']}",
                       params={"hour": 9, "weekend": False, "temp_f": 71}).json()["predictions"]
    assert all(p["setting"] == LEAVE for p in preds)

    r = client.delete("/api/learning/demo-history", headers=h)
    assert r.json()["removed"] >= MIN_SAMPLES and r.json()["models"] == 0
    assert learner.models() == []


def test_too_few_arrivals_means_no_model(client):
    ids = _setup_house(client)
    from app.db import session_scope
    from app.db.models import PreferenceSample

    with session_scope() as s:
        for f, label in _rows(MIN_SAMPLES - 1):
            s.add(PreferenceSample(person_id=ids["sam"], features=f, before={},
                                   labels={"light.living_room": label}))
    assert learner.train() == []
