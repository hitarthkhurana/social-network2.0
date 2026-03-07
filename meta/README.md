# MetaGlasses — iOS App

Ray-Ban Meta glasses layer for the hackathon project.
Streams camera frames + records audio from the glasses, and hands them off to the backend team.

---

## What this app does

| Mode | What happens |
|------|-------------|
| **Identify** | Tap button → captures one JPEG frame from glasses → `POST /identify` → backend returns name + details → glasses speak it aloud |
| **Enroll** | Tap Record → records .mp4 clip (glasses camera + mic audio) → Tap Stop → `POST /enroll` → backend saves the person |

---

## Prerequisites

- **Xcode 14+** — [download from Mac App Store](https://apps.apple.com/us/app/xcode/id497799835)
- **iPhone** running iOS 15.2+ (real device — Bluetooth doesn't work in the simulator)
- **Ray-Ban Meta glasses** with firmware v20+ and **Meta AI app** v254+ on the same iPhone
- **XcodeGen** (optional but fastest): `brew install xcodegen`

---

## Setup (Option A — XcodeGen, recommended)

```bash
cd meta/
brew install xcodegen     # one-time install
xcodegen generate         # creates MetaGlasses.xcodeproj
open MetaGlasses.xcodeproj
```

Then in Xcode:
1. Select your iPhone as the run target
2. Sign in with your Apple ID: **Xcode → Settings → Accounts → +**
3. Set the team in **MetaGlasses target → Signing & Capabilities**
4. Hit **Run** (Cmd+R)

---

## Setup (Option B — Manual Xcode project)

1. Open Xcode → **File → New → Project → iOS → App**
2. Name it `MetaGlasses`, bundle ID `com.hackathon.MetaGlasses`, language Swift, interface SwiftUI
3. Delete the auto-generated `ContentView.swift` and `Assets.xcassets` that Xcode creates
4. Drag the entire `MetaGlasses/` folder from this repo into the Xcode project navigator (check "Copy items if needed")
5. Add the DAT SDK via SPM:
   - **File → Add Package Dependencies**
   - URL: `https://github.com/facebook/meta-wearables-dat-ios`
   - Version: `0.4.0` or later
   - Add both `MWDATCore` and `MWDATCamera` to the MetaGlasses target
6. Replace the auto-generated `Info.plist` with the one in this folder

---

## Enable developer mode on the glasses

1. Open **Meta AI app** on your iPhone
2. Go to **Settings → App Info**
3. Tap the **App Version number five times** — a Developer Mode toggle appears
4. Enable it
5. Make sure glasses firmware is v20+ (**Devices tab → your glasses → General → About → Version**)

---

## Point at the backend

Open `Services/BackendClient.swift` and change the base URL:

```swift
var baseURL: String = "http://YOUR_BACKEND_IP:8000"
```

---

## Backend API contract (for the backend team)

### `POST /identify`
- **Body**: raw JPEG bytes (`Content-Type: image/jpeg`)
- **Response**:
```json
{ "name": "John Doe", "details": "Met at the gym last June" }
// or, if unknown:
{ "name": null, "details": null }
```

### `POST /enroll`
- **Body**: `multipart/form-data` with field `video` containing a `.mp4` file
  - Video track: H.264, 360×640, 7 fps
  - Audio track: AAC, 8 kHz mono (HFP quality — wearer's voice)
- **Response**:
```json
{ "status": "saved", "name": "John Doe" }
```

---

## Project structure

```
meta/
├── project.yml                          # XcodeGen config
├── README.md
└── MetaGlasses/
    ├── Info.plist                       # All required DAT SDK keys
    ├── MetaGlassesApp.swift             # App entry point, SDK init, URL callback
    ├── ContentView.swift                # Root UI, camera preview, tab switcher
    ├── Views/
    │   ├── IdentifyView.swift           # One-tap identify flow
    │   └── EnrollView.swift             # Record + upload enroll flow
    ├── Services/
    │   ├── GlassesManager.swift         # DAT SDK wrapper (stream, photo capture)
    │   ├── VideoRecorder.swift          # AVAssetWriter: frames + audio → .mp4
    │   ├── BackendClient.swift          # HTTP: /identify and /enroll
    │   └── SpeechOutput.swift          # TTS through glasses speakers (A2DP)
    └── Extensions/
        └── UIImage+CVPixelBuffer.swift  # UIImage → CVPixelBuffer for video encoding
```

---

## Known limitations

- HFP mic beamforms on the **wearer's** voice. The other person's voice is quieter in the recording — the backend team should account for this in transcription.
- Bluetooth Classic bandwidth is limited; video quality adapts automatically. 360p @ 7fps is the sweet spot.
- Only one DAT session can run at a time on the glasses.
- App Store submission is not supported yet — use Meta's release channels or sideload for the demo.
