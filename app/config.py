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

    # Models
    detector_backend: str = "stub"
    recognizer_backend: str = "stub"
    yolo_model_path: str = "models/yolo11n.pt"
    detection_confidence: float = 0.5
    face_match_threshold: float = 0.6

    @property
    def camera_index_or_url(self) -> "int | str":
        """OpenCV wants an int for webcams and a str for RTSP/file sources."""
        return int(self.camera_source) if self.camera_source.isdigit() else self.camera_source


settings = Settings()
