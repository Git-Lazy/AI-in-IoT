"""Application entrypoint -- wires every subsystem onto the event bus.

    camera -> detector -> recognizer ─┐
                                      ├─> EventBus ─> rules ─> MQTT -> devices
    MQTT devices/sensors ─────────────┘           ├─> database (event log)
                                                  └─> WebSocket -> dashboard

Run:  python -m app.main       (or: uvicorn app.main:app --reload)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app import __version__
from app.api import routes_admin, routes_devices, routes_events, routes_people, routes_stream
from app.api.schemas import HealthOut
from app.automation import expire_presence, register_handlers
from app.config import PROJECT_ROOT, settings
from app.db import init_db
from app.events import Event, EventType, bus
from app.iot import mqtt_bridge
from app.iot.simulator import simulator
from app.vision import cameras

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("homeai")

PRESENCE_SWEEP_SECONDS = 60


async def _presence_sweeper() -> None:
    """Periodically age out people who haven't been seen recently."""
    while True:
        await asyncio.sleep(PRESENCE_SWEEP_SECONDS)
        try:
            await expire_presence()
        except Exception:
            log.exception("presence sweep failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting Emerging AI IoT Integration v%s", __version__)

    bus.bind_loop(asyncio.get_running_loop())
    init_db()
    register_handlers()
    routes_stream.register_broadcast()

    mqtt_bridge.start()
    simulator.load()
    simulator.start()

    if settings.vision_enabled:
        cameras.start()

    sweeper = asyncio.create_task(_presence_sweeper())
    await bus.publish(Event(type=EventType.SYSTEM, source="backend", payload={"state": "online"}))

    yield

    log.info("shutting down")
    sweeper.cancel()
    await simulator.stop()
    cameras.stop()
    mqtt_bridge.stop()


app = FastAPI(
    title="Emerging AI IoT Integration",
    description="AI-driven smart home platform: computer vision + IoT + automation.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # local-network project; tighten before any exposure
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_admin.router)
app.include_router(routes_people.router)
app.include_router(routes_devices.router)
app.include_router(routes_events.router)
app.include_router(routes_stream.router)


@app.get("/api/health", response_model=HealthOut, tags=["system"])
def health():
    return HealthOut(
        status="ok",
        version=__version__,
        mqtt_connected=mqtt_bridge.connected,
        vision_running=cameras.running,
        detector_backend=settings.detector_backend,
        recognizer_backend=settings.recognizer_backend,
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    return (PROJECT_ROOT / "app" / "web" / "dashboard.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
        # Watch code only. The project root also holds data/homeai.db, which is
        # written on every event and floods the log with "changes detected".
        reload_dirs=["app"],
    )
