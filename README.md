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

## First run: admin, faces, devices

Open the dashboard. With no admin yet the house is in **setup mode** and the
Admin panel is open to anyone:

1. **Create the admin** (name + password). Use your existing name to promote
   yourself.
2. **Faces tab** — stand in front of the camera and press Capture ~10 times.
   From now on, admin login needs the password *and* your face on camera.
3. **Devices tab** — "Add a demo house" for a full set of simulated devices, or
   add your own.
4. **People & access** — add household members with a role, then enroll their
   faces.

Faces can also be enrolled from a folder of photos:

```powershell
python scripts/capture_faces.py --name "Samuel"      # backend must be stopped (camera)
python scripts/enroll_face.py --name "Samuel" --images data/faces/samuel
```

Only the embedding vector is stored — never the photograph. That follows directly
from the FaceNet approach (Schroff et al., 2015) and is the project's main privacy
property.

## Roles and camera-checked control

Every device button is checked against **whoever the camera in that device's
room can see** (`app/access.py`, `app/api/routes_devices.py`). Nobody
recognized, or the person isn't allowed → refused, logged as `access_denied`,
and a notification is raised. A logged-in admin skips the camera check.

| Role | Lights | Plugs / TV | AC | Door lock | Garage |
|---|---|---|---|---|---|
| admin | ✓ | ✓ | ✓ | ✓ | ✓ |
| adult | ✓ | ✓ | ✓ | ✓ | ✓ |
| child | ✓ | ✓ | ✗ | ✗ | ✗ |
| guest | ✓ | ✗ | ✗ | ✗ | ✗ |

An admin can override any cell per person, for a whole kind or for a single
device. Lookup order: device rule > kind rule > role default.

**Arrival preferences.** Each person can have devices set when they arrive
(e.g. living-room light at 40%, AC cooling to 70°F). Only devices that person
is allowed to use are applied. With none saved, arriving turns on the lights in
the room where they were seen. When the last person leaves, lights, plugs and
the AC are switched off.

**Several cameras.** `CAMERAS=front_door:Entry:0,garage:Garage:1` runs one
pipeline per camera; a device is checked by the camera whose room matches its
own, falling back to the first camera.

## Simulated devices

Devices run **inside the backend** until real hardware exists — no broker
needed (`app/iot/simulator.py`). Lights dim, the AC pulls the room temperature
toward its target over time, the garage takes a few seconds to open, motion
clears on its own, and sensors can be triggered from the dashboard.

When a real device arrives, switch it from Simulated to Real in Admin → Devices.
It keeps its id, so its permissions and preferences carry over, and commands
start going to `homeai/device/<id>/set` over MQTT instead.

## Face recognition tuning

| Stage | What it does | Setting |
|---|---|---|
| Detection | MTCNN instead of Haar; faces below a confidence are ignored | `FACE_MIN_CONFIDENCE` |
| Quality gate | Too-small or blurry faces are skipped, not guessed at | `FACE_MIN_SIZE`, `FACE_MIN_SHARPNESS` |
| Alignment | Face rotated so the eyes are level before embedding | — |
| Match | Nearest enrolled face must be within the threshold… | `FACE_MATCH_THRESHOLD` |
| Margin | …and clearly closer than the next-closest person | `FACE_MATCH_MARGIN` |
| Voting | 3 of the last 5 frames must agree before a track is trusted | `FACE_VOTES_REQUIRED`, `FACE_VOTE_WINDOW` |

Measured on the 10 enrolled photos, alignment cut the average same-person
cosine distance from 0.311 (Haar) to 0.173, and the worst pair from 0.540 to
0.302 — so the threshold could drop from 0.6 to 0.45. Embeddings store the
cropper they were made with (`facenet-vggface2/mtcnn`); changing the cropper
means re-enrolling.

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
| 1 | Camera + multiple IoT devices connected | Simulated devices in-backend; real hardware pending |
| 2 | Capture and process live feeds | Done (`vision/camera.py`) |
| 3 | Detect people and basic events | Done (YOLO11 + tracking) |
| 4 | Facial recognition for registered members | Done (MTCNN + FaceNet, voting) |
| 5 | Identify entering/leaving | Done (`track_presence` / `expire_presence`) |
| 6 | Store event logs | Done (`EventLog`) |
| 7 | Notifications | In-app sink done; external delivery pending |
| 8 | Dashboard | Done |
| 9 | AI-driven automation actions | Per-person arrival preferences, role-based access |
| 10 | End-to-end communication | Demonstrable with `fake_device.py` |
