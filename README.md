# Hand UI

A desktop interface you drive with one hand, using a webcam. It's built on MediaPipe hand landmarks and OpenCV, plus a small neural network that learns custom gestures.

| Gesture | What it does |
| --- | --- |
| **Fist or pinch** over a card | Grab it. Move to drag, open your hand to drop. Drop on the bin to delete. |
| **Swipe** an open hand left / right | Next / previous page |
| **Swipe** up / down | New card / tidy the page |
| **Hold an open palm** still (~0.7 s) | Toggle the radial menu. Point at an item, then grab to choose it. |
| **Custom gestures** you record | Whatever you bind them to in `handui/config.py` |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The first run downloads MediaPipe's hand landmark model (~7.5 MB) into `models/`.
On macOS, allow camera access for your terminal in **System Settings > Privacy & Security > Camera**.

## Run

```bash
.venv/bin/python -m handui
```

Keys: `q` quit · `c` camera background · `s` skeleton · `h` HUD · `m` menu · `n` new card · `t` tidy · `a`/`d` pages.

## Custom gestures

1. Record each gesture for about 8 seconds. While recording, move your hand around, tilt it, and change its distance from the camera.
   ```bash
   .venv/bin/python -m handui record thumbs_up
   .venv/bin/python -m handui record peace
   .venv/bin/python -m handui record none   # relaxed, random poses, so normal movement isn't misread
   ```
2. Train. Accuracy is measured on the last 20% of each recording, which the model doesn't see during training.
   ```bash
   .venv/bin/python -m handui train
   ```
3. Bind labels to actions in `BINDINGS` in `handui/config.py`, for example `"custom:thumbs_up": "new_card"`. Available actions are listed in `HandUI.ACTIONS`.

`python -m handui labels` lists what you've recorded. `python -m handui forget LABEL` deletes the samples for one label.

Features are invariant to position, scale and handedness, but not to rotation, so thumbs-up and thumbs-down stay different gestures.

## How it works

```
camera -> tracker.py      MediaPipe HandLandmarker -> 21 landmarks per hand
       -> features.py     finger extension, grab strength, pinch, classifier features
       -> gestures.py     GestureEngine: grab/drag, swipe, palm-dwell menu, custom labels -> Events
       -> ui.py           HandUI: cards, pages, radial menu, HUD (pure OpenCV drawing)
classifier.py             sample storage, augmentation, MLP training, inference
```

The cursor follows the palm centre, not a fingertip, so it stays put when you close your hand to grab. It's smoothed with a One Euro filter. Every threshold is in `GestureConfig` in `handui/config.py`.

## Tests

The tests drive the engine, classifier and UI with synthetic hands, so they don't need a camera:

```bash
.venv/bin/python -m unittest discover -s tests
```
