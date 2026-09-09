# Gaze Mouse

Camera-only mobile gaze mouse prototype.

## What it does
- Uses the front camera and MediaPipe Face Landmarker in the browser.
- Runs a 9-point gaze calibration.
- Maps eye/iris features to a full-viewport virtual cursor.
- Supports wink, double-blink, or dwell-to-click.
- Camera frames stay on-device; the MediaPipe runtime/model load from public CDNs.

## Important limitation
A normal web page cannot control the Android system cursor or other apps. This prototype validates gaze accuracy inside its browser viewport. A future Android Accessibility Service/native companion can translate gaze coordinates into OS-level taps/swipes.

## Run
Host over HTTPS (GitHub Pages is fine), then open in Chrome on your phone.
