"""Live channels for the dashboard: event WebSocket and MJPEG camera preview."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import auth
from app.events import Event, bus
from app.vision import cameras
from app.vision.doorway import DoorLine, save_door_line

log = logging.getLogger(__name__)
router = APIRouter(tags=["stream"])


class ConnectionManager:
    """Fans bus events out to every open dashboard socket."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, event: Event) -> None:
        dead: list[WebSocket] = []
        for ws in self._clients:
            try:
                await ws.send_json(event.to_dict())
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


@router.websocket("/ws/events")
async def events_socket(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # We only push; this read keeps the connection alive and detects close.
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


@router.get("/api/cameras")
def list_cameras():
    """Every running camera, its room, and who it can see right now."""
    return [
        {
            "name": p.camera_name,
            "room": p.room,
            "present": [m.name for m in p.present_people()],
            "door": door.to_dict() if (door := getattr(p, "door", None)) else None,
        }
        for p in cameras.all()
    ]


class DoorLineIn(BaseModel):
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)
    flip: bool = False


def _camera(name: str):
    pipeline = cameras.get(name)
    if pipeline is None:
        raise HTTPException(404, f"no camera named '{name}' is running")
    return pipeline


@router.put("/api/cameras/{name}/door")
def set_door_line(name: str, body: DoorLineIn, _admin=Depends(auth.require_admin)):
    """Draw (or redraw) a camera's door line. Takes effect immediately."""
    pipeline = _camera(name)
    try:
        line = DoorLine.from_dict(body.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    save_door_line(name, line)
    pipeline.set_door(line)
    return line.to_dict()


@router.delete("/api/cameras/{name}/door", status_code=204)
def remove_door_line(name: str, _admin=Depends(auth.require_admin)):
    """No door line: sightings on this camera count as arrivals again."""
    pipeline = _camera(name)
    save_door_line(name, None)
    pipeline.set_door(None)


@router.get("/stream/preview")
async def preview(camera: str | None = None):
    """MJPEG stream of the annotated camera feed (default: the first camera)."""
    pipeline = cameras.get(camera)
    if pipeline is None:
        raise HTTPException(503, "no camera running" if camera is None else f"no camera '{camera}'")

    async def frames():
        import cv2

        while True:
            frame = pipeline.latest_preview()
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes()
                        + b"\r\n"
                    )
            await asyncio.sleep(0.1)

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


def register_broadcast() -> None:
    bus.subscribe(manager.broadcast)
