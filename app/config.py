"""Central configuration, loaded from environment / .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Backend
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # Database
    database_url: str = "sqlite:///./data/homeai.db"

    # MQTT
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_client_id: str = "homeai-backend"
    mqtt_base_topic: str = "homeai"

    # Camera
    camera_source: str = "0"
    camera_fps: int = 10
    vision_enabled: bool = True
    # Several cameras: "name:Room:source,name:Room:source", e.g.
    #   CAMERAS=front_door:Entry:0,garage:Garage:1
    # A device is access-checked by the camera in its room. Empty = one camera,
    # "front_door" in the Entry, on CAMERA_SOURCE.
    cameras: str = ""

    # Admin
    # Admin login needs password + the admin's face on a camera. Turn off only
    # while you have no working camera.
    admin_face_verify: bool = True
    admin_session_hours: int = 8

    # Models
    detector_backend: str = "stub"
    recognizer_backend: str = "stub"
    yolo_model_path: str = "models/yolo11n.pt"
    detection_confidence: float = 0.5

    # Face recognition precision. See README "Face recognition tuning".
    face_detector: str = "mtcnn"          # mtcnn | haar
    face_match_threshold: float = 0.45    # max cosine distance to accept a match
    face_match_margin: float = 0.08       # best person must beat runner-up by this
    face_min_confidence: float = 0.95     # MTCNN face probability
    face_min_size: int = 60               # px; smaller faces are too low-detail to trust
    face_min_sharpness: float = 25.0      # Laplacian variance; rejects motion blur
    face_votes_required: int = 3          # agreeing frames before an identity is trusted
    face_vote_window: int = 5             # ...out of this many recent frames

    @property
    def camera_index_or_url(self) -> "int | str":
        """OpenCV wants an int for webcams and a str for RTSP/file sources."""
        return _source(self.camera_source)

    @property
    def camera_list(self) -> "list[tuple[str, str, int | str]]":
        """(name, room, source) for every configured camera."""
        if not self.cameras.strip():
            return [("front_door", "Entry", self.camera_index_or_url)]
        out = []
        for spec in self.cameras.split(","):
            # maxsplit keeps RTSP URLs ("rtsp://host:554/...") intact
            name, room, source = (s.strip() for s in spec.split(":", 2))
            out.append((name, room, _source(source)))
        return out


def _source(raw: str) -> "int | str":
    return int(raw) if raw.isdigit() else raw


settings = Settings()
