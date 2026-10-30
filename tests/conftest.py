"""Test environment: stub models, no camera, no broker, a throwaway database.

Environment variables beat .env in pydantic-settings, so setting them here --
before anything imports app.config -- keeps tests off your real camera, your
real database, and the heavy YOLO/FaceNet models.
"""

import os
import tempfile
from pathlib import Path

_DB = Path(tempfile.mkdtemp()) / "test.db"

os.environ.update(
    DATABASE_URL=f"sqlite:///{_DB.as_posix()}",
    DETECTOR_BACKEND="stub",
    RECOGNIZER_BACKEND="stub",
    FACE_DETECTOR="haar",
    VISION_ENABLED="false",
    MQTT_HOST="127.0.0.1",
    MQTT_PORT="1",           # nothing listens here: connect fails fast
    CAMERAS="",
)
