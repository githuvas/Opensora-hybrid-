# Offline AI Movie Maker

This project is a **self-contained offline pipeline** that turns a short script into a rendered video featuring *consistent AI-generated characters*, optional physics-aware 3D scenes, AI-generated speech, and a simple **Gradio** user-interface for rapid experimentation.

> **Status**: Proof-of-concept skeleton. Heavy AI functionality (Stable-Diffusion + IP-Adapter, ThreeStudio, TTS) is wrapped behind *pluggable stubs* so that the application runs out-of-the-box. Swap the stubbed calls with real model pipelines to unlock full power.

---

## 1. Features

1. Text-to-Video pipeline (1 s – 10 min)
2. Up to 10 image references to maintain character consistency through Tencent-AI Lab [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
3. Optional physics-aware 3 D scene generation via [ThreeStudio](https://github.com/threestudio-project/threestudio)
4. AI voice generation (Coqui-TTS as default)
5. Composited and encoded with **MoviePy** → downloadable MP4
6. Everything runs locally / offline — no external APIs required
7. Simple Gradio front-end with text input, image upload, duration slider, *Download* button

---

## 2. Architecture

```
main.py (Gradio UI) ─┬── char_gen.CharacterGenerator  (StableDiffusion + IP-Adapter)
                    ├── scene_gen.SceneGenerator      (ThreeStudio + physics)
                    ├── tts_gen.TTSGenerator          (Coqui / other TTS)
                    └── video_composer.VideoComposer  (MoviePy)
```

Each module exposes a minimal public interface and is **hot-swappable** so you can plug in your favourite models without touching the UI layer.

---

## 3. Quick Start

```bash
# 1. Create env & install deps (CPU-only by default)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Launch the UI
python main.py
```
Then open the printed URL (defaults to http://127.0.0.1:7860) and start creating!

---

## 4. Replacing Stubs with Real Models

All heavy lifting is isolated behind classes in `src/modules/`. Start by replacing the `generate_*` placeholder methods with real pipelines:

* `char_gen.py` – Stable Diffusion XL + IP-Adapter, *diffusers* library recommended.
* `scene_gen.py` – Follow ThreeStudio Colab or their API to render a mesh or image sequence.
* `tts_gen.py` – Swap stub with Coqui-TTS, Bark, or any HuggingFace TTS.

Make sure each method returns the expected datatype (numpy RGB frames, wav PCM bytes etc.). The GUI and composer will continue working.

---

## 5. Directory Layout

```
.
├── main.py
├── requirements.txt
├── src/
│   └── modules/
│       ├── __init__.py
│       ├── char_gen.py
│       ├── scene_gen.py
│       ├── tts_gen.py
│       └── video_composer.py
└── README.md
```

---

## 6. License

All code in this repository is released under the MIT license. Models you download might have their own licenses — please respect them.
