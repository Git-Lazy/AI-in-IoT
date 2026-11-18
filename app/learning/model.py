"""The learned-preferences model: one small decision tree per person and device.

Why a decision tree:
  * It learns from a few dozen examples -- a household produces about one per
    arrival, so a few weeks is all there will ever be.
  * It explains itself. Every prediction is a path of plain conditions ("after
    6 pm, on a weekday, warmer than 77°F"), which is what we show the person
    when the house acts on it, and what makes a wrong guess fixable.

Each tree is scored with 5-fold cross-validation against a baseline that always
predicts the person's most common choice for that device. A model that can't
beat "do the usual" hasn't learned anything about the situation and shouldn't
be trusted with it.

Training takes milliseconds, so models are just rebuilt from the database on
startup and whenever new examples come in -- no model files to keep in sync.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from app.db import session_scope
from app.db.models import Device, Person, PreferenceSample
from app.learning.samples import LEAVE, describe

log = logging.getLogger(__name__)

# What a tree may split on. Weekday is recorded but "weekend" carries the habit
# and reads better in an explanation.
FEATURES = ("hour", "weekend", "temp_f", "others_home")
MIN_SAMPLES = 8          # fewer complete arrivals than this: don't train at all
MAX_DEPTH = 3            # at most three conditions per rule -- readable, and hard to overfit
MIN_LEAF = 3             # every rule must be backed by at least this many arrivals


@dataclass
class Prediction:
    device_id: str
    device: str
    kind: str
    setting: str                 # a named setting, or "leave"
    confidence: float            # share of similar past arrivals that ended this way
    support: int                 # how many past arrivals fell under the same rule
    because: list[str]           # the rule's conditions, in words

    @property
    def wording(self) -> str:
        return describe(self.kind, self.setting)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id, "device": self.device, "kind": self.kind,
            "setting": self.setting, "wording": self.wording,
            "confidence": round(self.confidence, 2), "support": self.support,
            "because": self.because,
        }


@dataclass
class DeviceModel:
    person_id: int
    person: str
    device_id: str
    device: str
    kind: str
    tree: DecisionTreeClassifier
    samples: int
    accuracy: float              # cross-validated
    baseline: float              # "always the most common choice", cross-validated
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def beats_baseline(self) -> bool:
        return self.accuracy >= self.baseline

    def predict(self, features: dict[str, float]) -> Prediction:
        x = _row(features)
        node, bounds = _descend(self.tree, x)
        setting, share = _outcome(self.tree, node)
        return Prediction(
            device_id=self.device_id, device=self.device, kind=self.kind,
            setting=setting, confidence=share,
            support=int(self.tree.tree_.n_node_samples[node]),
            because=_explain(bounds),
        )

    def rules(self) -> list[dict[str, Any]]:
        """The tree as "when ... -> setting (n arrivals, x% of them)" rules."""
        t = self.tree.tree_
        settled = _settled(self.tree)
        out = []

        def walk(node: int, bounds: dict[str, list[float]]) -> None:
            if settled[node] >= 0:
                setting, share = _outcome(self.tree, node)
                out.append({
                    "when": _explain(bounds) or ["always"],
                    "setting": setting, "wording": describe(self.kind, setting),
                    "arrivals": int(t.n_node_samples[node]), "share": round(share, 2),
                })
                return
            name, thr = FEATURES[t.feature[node]], float(t.threshold[node])
            lo, hi = bounds.get(name, [-np.inf, np.inf])
            walk(t.children_left[node], {**bounds, name: [lo, min(hi, thr)]})
            walk(t.children_right[node], {**bounds, name: [max(lo, thr), hi]})

        walk(0, {})
        # Rules that change something first, most-supported first.
        return sorted(out, key=lambda r: (r["setting"] == LEAVE, -r["arrivals"]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id, "person": self.person,
            "device_id": self.device_id, "device": self.device, "kind": self.kind,
            "samples": self.samples, "accuracy": round(self.accuracy, 2),
            "baseline": round(self.baseline, 2), "beats_baseline": self.beats_baseline,
            "counts": self.counts, "rules": self.rules(),
        }


# ------------------------------------------------------------------ training


def train_one(rows: list[tuple[dict, str]]) -> tuple[DecisionTreeClassifier, float, float]:
    """Fit a tree on (features, setting) rows; return it with CV and baseline accuracy."""
    X = np.array([[float(f.get(k, 0)) for k in FEATURES] for f, _ in rows])
    y = np.array([label for _, label in rows])
    tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, random_state=0)
    cv = KFold(n_splits=min(5, len(rows)), shuffle=True, random_state=0)
    accuracy = float(np.mean(cross_val_score(tree, X, y, cv=cv)))
    baseline = float(np.mean(cross_val_score(DummyClassifier(strategy="most_frequent"), X, y, cv=cv)))
    tree.fit(X, y)
    return tree, accuracy, baseline


class Learner:
    """Holds the current models; rebuilt wholesale by train()."""

    def __init__(self) -> None:
        self._models: dict[tuple[int, str], DeviceModel] = {}
        self._lock = threading.Lock()
        self.trained_at: datetime | None = None

    def train(self) -> list[DeviceModel]:
        with session_scope() as session:
            people = {p.id: p.name for p in session.query(Person)}
            devs = {d.id: (d.name, d.kind) for d in session.query(Device)}
            data: dict[tuple[int, str], list[tuple[dict, str]]] = {}
            for s in session.query(PreferenceSample).filter(PreferenceSample.labels.isnot(None)):
                for device_id, label in (s.labels or {}).items():
                    data.setdefault((s.person_id, device_id), []).append((s.features or {}, label))

        models: dict[tuple[int, str], DeviceModel] = {}
        for (person_id, device_id), rows in data.items():
            if len(rows) < MIN_SAMPLES or person_id not in people or device_id not in devs:
                continue
            tree, accuracy, baseline = train_one(rows)
            counts: dict[str, int] = {}
            for _, label in rows:
                counts[label] = counts.get(label, 0) + 1
            name, kind = devs[device_id]
            models[(person_id, device_id)] = DeviceModel(
                person_id=person_id, person=people[person_id], device_id=device_id,
                device=name, kind=kind, tree=tree, samples=len(rows),
                accuracy=accuracy, baseline=baseline, counts=counts,
            )

        with self._lock:
            self._models = models
            self.trained_at = datetime.now(timezone.utc)
        log.info("learning: trained %d model(s)", len(models))
        return list(models.values())

    def models(self, person_id: int | None = None) -> list[DeviceModel]:
        with self._lock:
            found = list(self._models.values())
        return [m for m in found if person_id is None or m.person_id == person_id]

    def predict(self, person_id: int, features: dict[str, float]) -> list[Prediction]:
        return [m.predict(features) for m in self.models(person_id)]


learner = Learner()


# ------------------------------------------------------------- explanations


def _row(features: dict[str, float]) -> np.ndarray:
    return np.array([[float(features.get(k, 0)) for k in FEATURES]])


def _settled(tree: DecisionTreeClassifier) -> np.ndarray:
    """Per node: the class index if every leaf below it predicts that class, else -1.

    A tree often splits a range into pieces that all end the same way ("6-7:30pm
    -> on", "after 7:30pm -> on"); treating such a node as one rule ("after 6pm
    -> on") is the same prediction, said once.
    """
    t = tree.tree_
    out = np.full(t.node_count, -1)

    def visit(node: int) -> int:
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:
            out[node] = int(np.argmax(t.value[node][0]))
        else:
            a, b = visit(left), visit(right)
            out[node] = a if a == b else -1
        return out[node]

    visit(0)
    return out


def _outcome(tree: DecisionTreeClassifier, node: int) -> tuple[str, float]:
    """The setting a node predicts and the share of its arrivals that ended that way."""
    t = tree.tree_
    settled = _settled(tree)[node]
    if settled < 0:                                   # not a rule node: majority of what's here
        value = t.value[node][0]
        settled = int(np.argmax(value))
    # Leaf class shares, weighted by how many arrivals each leaf holds.
    agree = total = 0.0

    def gather(n: int) -> None:
        nonlocal agree, total
        if t.children_left[n] == -1:
            value = t.value[n][0]
            agree += value[settled] / value.sum() * t.n_node_samples[n]
            total += t.n_node_samples[n]
        else:
            gather(t.children_left[n])
            gather(t.children_right[n])

    gather(node)
    return str(tree.classes_[settled]), float(agree / total)


def _descend(tree: DecisionTreeClassifier, x: np.ndarray) -> tuple[int, dict[str, list[float]]]:
    """Follow x down the tree to its rule node; return it and the ranges on the way."""
    t = tree.tree_
    settled = _settled(tree)
    bounds: dict[str, list[float]] = {}
    node = 0
    while settled[node] < 0:
        name, thr = FEATURES[t.feature[node]], float(t.threshold[node])
        lo, hi = bounds.get(name, [-np.inf, np.inf])
        if x[0, t.feature[node]] <= thr:
            bounds[name], node = [lo, min(hi, thr)], t.children_left[node]
        else:
            bounds[name], node = [max(lo, thr), hi], t.children_right[node]
    return node, bounds


def _clock(hour: float) -> str:
    """18.05 -> "6pm". Thresholds fall between examples, so to the quarter hour is honest enough."""
    quarters = int(round(hour * 4))
    h, m = quarters // 4, quarters % 4 * 15
    suffix = "am" if h % 24 < 12 else "pm"
    h12 = h % 12 or 12
    return f"{h12}{suffix}" if m == 0 else f"{h12}:{m:02d}{suffix}"


def _explain(bounds: dict[str, list[float]]) -> list[str]:
    """Feature ranges -> short phrases, e.g. ["after 6pm", "on a weekday"]."""
    out = []
    for name in FEATURES:
        if name not in bounds:
            continue
        lo, hi = bounds[name]
        has_lo, has_hi = np.isfinite(lo), np.isfinite(hi)
        if name == "hour":
            if has_lo and has_hi:
                out.append(f"between {_clock(lo)} and {_clock(hi)}")
            elif has_lo:
                out.append(f"after {_clock(lo)}")
            else:
                out.append(f"before {_clock(hi)}")
        elif name == "weekend":
            out.append("on the weekend" if has_lo else "on a weekday")
        elif name == "temp_f":
            if has_lo and has_hi:
                out.append(f"between {lo:.0f}°F and {hi:.0f}°F inside")
            elif has_lo:
                out.append(f"warmer than {lo:.0f}°F inside")
            else:
                out.append(f"{hi:.0f}°F or cooler inside")
        elif name == "others_home":
            out.append("someone else home" if has_lo else "nobody else home")
    return out
