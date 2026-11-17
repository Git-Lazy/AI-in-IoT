"""Learned preferences: the house picks up how each person likes it.

Every arrival becomes a training example (samples.py): the situation when they
walked in, and what they changed in the minutes after. A model per person and
device learns from those examples which setting they want in which situation.
"""

from app.learning.samples import (
    LEARNABLE_KINDS,
    LEAVE,
    finalize_due,
    on_arrival,
    on_left,
    payload_for,
    setting_of,
)

__all__ = [
    "LEARNABLE_KINDS",
    "LEAVE",
    "finalize_due",
    "on_arrival",
    "on_left",
    "payload_for",
    "setting_of",
]
