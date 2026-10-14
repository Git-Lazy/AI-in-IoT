# Emerging AI IoT Integration

Capstone project — Samuel Elesho, Neumont University.

An AI-driven smart home platform that unifies camera-based computer vision, IoT
device messaging, and automated decision-making into one system. Cameras produce
detections, MQTT devices produce readings, and a central event bus turns both into
decisions: notifications, device commands, and presence state.

## Architecture

```
  camera ──> detector ──> recognizer ──┐
   (OpenCV)   (YOLO11)     (FaceNet)   │
                                       ├──> EventBus ──┬──> rules engine ──> MQTT ──> devices
  MQTT devices / sensors ──────────────┘               ├──> SQLite event log
                                                       └──> WebSocket ──> dashboard
```

The event bus (`app/events.py`) is the single seam in the system. Every subsystem
either produces `Event`s or consumes them, and nothing talks directly to anything
else — which is what lets the AI models be swapped without rewiring the platform.

| Layer | Module | Role |
|---|---|---|
| Config | `app/config.py` | Environment-driven settings |
| Event bus | `app/events.py` | Async pub/sub; the system's nervous system |
| Vision | `app/vision/` | Camera capture, detection, face recognition |
| IoT | `app/iot/` | MQTT bridge, both directions |
| Decisions | `app/automation/rules.py` | Events → device commands + notifications |
| Storage | `app/db/` | People, face embeddings, devices, event log |
| API / UI | `app/api/`, `app/web/` | REST, WebSocket, MJPEG preview, dashboard |

## Swappable model backends

Detection and recognition sit behind abstract interfaces with three
implementations each, selected by environment variable:

| `DETECTOR_BACKEND` | Behavior |
|---|---|
| `stub` | Detects nothing — lets the platform run with no ML dependencies |
| `motion` | OpenCV background subtraction — a dependency-free baseline |
| `yolo` | Ultralytics YOLO11 with tracking (Week 3) |

| `RECOGNIZER_BACKEND` | Behavior |
|---|---|
| `stub` | Deterministic pixel-statistics vector — exercises enrollment and matching |
| `facenet` | FaceNet / InceptionResnetV1, 512-d embeddings (Week 3) |

The `motion` and `stub` backends are not throwaway code — they are the control
condition for the Week-10 evaluation. "YOLO vs. motion baseline, measured on the
same footage" is a real result; "YOLO works" is not.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Start the broker (and Home Assistant, when you get to Week 4):

```powershell
docker compose up -d mosquitto
```

Run the backend:

```powershell
python -m app.main
```

Then open <http://localhost:8000> for the dashboard and
<http://localhost:8000/docs> for the API.

In a second terminal, simulate IoT hardware:

```powershell
python scripts/fake_device.py
```

The dashboard should start showing `sensor_reading` events, and the on/off
buttons should drive the simulated devices through MQTT round-trip.

## Enrolling a face

```powershell
python scripts/enroll_face.py --name "Samuel" --images data/faces/samuel
```

Only the embedding vector is stored — never the photograph. That follows directly
from the FaceNet approach (Schroff et al., 2015) and is the project's main privacy
property.

## Testing without hardware

Point `CAMERA_SOURCE` at a video file instead of a webcam index:

```
CAMERA_SOURCE=data/test_footage/front_door.mp4
```

The camera loops the file, giving identical frames on every run — which is what
makes false-positive counts comparable between model configurations.

Run the unit tests:

```powershell
pytest
```

## Degraded-mode behavior

The platform boots with any subsystem missing: no broker, no camera, no models.
Each failure logs a warning and disables that path rather than aborting startup.
This is deliberate — it means Week 4 backend work isn't blocked on Week 3 model
work, or on hardware that hasn't shipped.

## MQTT topics

| Topic | Direction | Payload |
|---|---|---|
| `homeai/device/<id>/state` | device → backend | JSON state |
| `homeai/device/<id>/set` | backend → device | JSON command |
| `homeai/sensor/<id>` | device → backend | JSON reading |
| `homeai/event` | backend → subscribers | Serialized `Event` |

Publishing internal events to `homeai/event` is the integration point for Home
Assistant, and mirrors how Frigate exposes its detections.

## MVP progress

| # | Requirement | Status |
|---|---|---|
| 1 | Camera + multiple IoT devices connected | Scaffolded (simulated devices) |
| 2 | Capture and process live feeds | Done (`vision/camera.py`) |
| 3 | Detect people and basic events | Interface done; YOLO pending (Week 3) |
| 4 | Facial recognition for registered members | Flow done; FaceNet pending (Week 3) |
| 5 | Identify entering/leaving | Done (`track_presence` / `expire_presence`) |
| 6 | Store event logs | Done (`EventLog`) |
| 7 | Notifications | In-app sink done; external delivery pending |
| 8 | Dashboard | Done |
| 9 | AI-driven automation actions | Rules engine done, 3 rules |
| 10 | End-to-end communication | Demonstrable with `fake_device.py` |
