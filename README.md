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

## Door lines: arriving and leaving through the door

Being in view of the entry camera isn't the same as coming home. Draw a line
across the doorway, and only **walking in across it** counts as an arrival.

**On the dashboard** (admin): pick the entry camera and press **Door line**.
Click the floor at one side of the doorway, then the other, just inside the
threshold. A green arrow shows which side counts as inside; press **Flip** if
it's backwards, then **Save line**. It takes effect immediately and is saved in
the database. Mount the camera inside, facing the door.

**Or in `.env`**, as a starting value (a line saved on the dashboard wins):

```
DOOR_LINES=front_door:0.15,0.70,0.85,0.70
```

The numbers are two points, `x1,y1,x2,y2`, as fractions of the frame
(0,0 is top-left, 1,1 is bottom-right). The side toward the bottom of the
picture is "inside"; add `:flip` to reverse it. The line appears in yellow on
the preview, and each person's feet appear as a dot.

- A person whose feet move from the far side to the near side, and stay there
  for 3 frames, has **entered**. Standing on the line (a dead band of
  `DOOR_BAND`) or a one-frame wobble doesn't count.
- The entry is announced with their name once the face is recognized. The
  system waits up to `DOOR_IDENTIFY_SECONDS` for a face; after that, the
  person is an unrecognized entry and the phone gets an alert.
- On that camera, being recognized while standing in view no longer marks
  you home. Other cameras still do: someone seen on the couch is clearly home.
- **Walking out** across the line marks the person away immediately, and the
  usual away rules run (lights and AC off once nobody is home). People leave
  with their back to the camera, so the exit is named by whoever the track
  already was. With no face at all, if exactly one person is home and no other
  camera can see them, it was them (shown as a guess in the feed).
- Stepping in and straight back out before being recognized counts as nothing.
- With a door line, not being seen only marks someone away after
  `DOOR_PRESENCE_TIMEOUT_MINUTES` (120), as a backup for missed exits. Without
  one, it's `PRESENCE_TIMEOUT_MINUTES` (5), since that's the only clue.

**When a crossing isn't detected:** each person's feet dot on the preview is
labeled **IN**, **OUT** or **?** (not clearly on either side yet). It has to
read OUT before you cross and IN after. If it says IN all the time, your feet
are probably out of frame: the bottom of your box is the bottom of the
picture. Aim the camera so the floor at the doorway is visible, or draw the
line higher. If the label resets when you walk past, the tracker lost you and
started a new track; walk a little slower or improve the lighting.

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

### House view

**House view** in the dashboard header shows a floor plan of a basic house:
Garage, Living Room, Kitchen, Bedroom, Entry and Bathroom. Each device appears
in its room as a live icon:

- Lights glow and tint the room with their brightness.
- The AC shows the room temperature, colored by mode.
- The lock shows a closed or open padlock.
- The garage door on the plan opens and closes.
- Motion sensors pulse red, and door sensors swing open.

Click a device to control it; control is camera-checked, as on the device
cards. Each camera appears in its room, with the names of the people it
recognizes. A device whose room isn't on the plan (say, "Porch") is listed
underneath. Room names match case-insensitively, with a few aliases ("Foyer"
goes in Entry, "Lounge" in Living Room). Open `/#house` to go straight there.

### Emulating real hardware over MQTT

`scripts/fake_device.py` is a separate program that acts like real hardware.
It talks to the backend **only through Mosquitto**, which tests the full chain
before you buy anything:

```powershell
docker compose up -d mosquitto        # Docker Desktop must be running
# Dashboard: Admin -> Devices -> switch e.g. light.living_room to Real
python scripts/fake_device.py         # every device set to Real
python scripts/fake_device.py light.living_room thermostat.main   # or pick devices
python scripts/fake_device.py --no-gui                            # console only
```

A window shows each device. The bulb glows with its brightness, and the AC
cools the room toward its target. Buttons stand in for the physical world:
a wall switch, walking past a motion sensor, opening a door. Those changes
reach the dashboard over MQTT, just as a real sensor's would. While a device
is still set to Simulated, the backend ignores MQTT reports for it, so the
two never compete over its state.

## Phone notifications

Alerts go to the dashboard and, if `NTFY_TOPIC` is set, to your phone through
[ntfy](https://ntfy.sh). It's free and needs no account.

1. Install the **ntfy** app (Android / iOS) and tap **+** to subscribe to a topic.
   Pick a long random name such as `homeai-yourname-<random letters>`. Anyone who knows the
   topic can read it.
2. Put the same name in `.env` as `NTFY_TOPIC=homeai-yourname-<random letters>` and restart the backend.
3. Log in as admin on the dashboard and press **Test phone** in the Notifications panel.

| Alert | Priority | On the phone |
|---|---|---|
| Unrecognized person | high | Alarm-level push, with a camera snapshot attached |
| Welcome home / someone left | normal | Normal push |

`NTFY_MIN_PRIORITY=high` limits pushes to security alerts only. A repeat of the same alert
within `NTFY_COOLDOWN_SECONDS` (60 by default) is kept on the dashboard but not
pushed again. Notifications sent to the phone are marked 📱 on the dashboard.

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
| 5 | Identify entering/leaving | Door-line crossing both ways (`app/vision/doorway.py`), timeout as backup |
| 6 | Store event logs | Done (`EventLog`) |
| 7 | Notifications | Dashboard + phone push (ntfy), with camera snapshot on unknown faces |
| 8 | Dashboard | Done |
| 9 | AI-driven automation actions | Per-person arrival preferences, role-based access |
| 10 | End-to-end communication | Demonstrable with `fake_device.py` |
