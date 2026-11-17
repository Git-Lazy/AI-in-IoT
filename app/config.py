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
    # Windows capture backend for webcams: "" (OpenCV default, MSMF), "dshow"
    # (DirectShow -- try this if a USB webcam freezes or won't open), "msmf".
    camera_api: str = ""

    # Door lines: walking across one is what counts as arriving (see
    # app/vision/doorway.py). "camera:x1,y1,x2,y2[:flip]", ";" between cameras.
    door_lines: str = ""
    door_band: float = 0.03               # dead zone either side of the line, frame fraction
    door_confirm_frames: int = 3          # frames on the new side before a crossing counts
    door_identify_seconds: float = 4.0    # wait this long for a face after someone walks in

    # Not seen for this long = left. Short without a door line (it's the only
    # clue); long with one, where walking out is seen and this is just a backup.
    presence_timeout_minutes: int = 5
    door_presence_timeout_minutes: int = 120

    # Learned preferences (app/learning). After someone arrives, whatever they
    # change within this many minutes is what they "wanted" for that situation.
    learning_window_minutes: int = 15

    # Admin
    # Admin login needs password + the admin's face on a camera. Turn off only
    # while you have no working camera.
    admin_face_verify: bool = True
    admin_session_hours: int = 8

    # Phone notifications via ntfy. Empty topic = dashboard only. Anyone who
    # knows the topic can read it, so make it long and unguessable.
    ntfy_topic: str = ""
    ntfy_server: str = "https://ntfy.sh"
    ntfy_token: str = ""                  # only for a password-protected server
    ntfy_min_priority: str = "normal"     # low | normal | high -- lower stays on the dashboard
    ntfy_cooldown_seconds: int = 60       # don't re-push the same alert within this

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
