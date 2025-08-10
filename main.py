import gradio as gr
from pathlib import Path
from typing import List

from src.modules.char_gen import CharacterGenerator
from src.modules.scene_gen import SceneGenerator
from src.modules.tts_gen import TTSGenerator
from src.modules.video_composer import VideoComposer

# Instantiate once
char_gen = CharacterGenerator()
scene_gen = SceneGenerator()
tts_gen = TTSGenerator()
video_comp = VideoComposer()

def generate_movie(prompt: str, images: List[gr.Image], duration: int):
    # Convert gradio image objects to PIL images
    ref_images = [img for img in images if img is not None]
    frames = char_gen.generate_frames(prompt, ref_images, duration)
    frames = scene_gen.add_physics(frames)
    audio_path = tts_gen.generate_audio(prompt, duration)
    video_path = video_comp.compose(frames, audio_path)
    return video_path

with gr.Blocks() as demo:
    gr.Markdown("# Offline AI Movie Maker (Prototype)")
    with gr.Row():
        prompt_in = gr.Textbox(label="Script / Prompt", lines=3, placeholder="Describe your scene…")
    with gr.Row():
        img_inputs = gr.Image(type="pil", label="Reference Images (up to 10)", image_mode="RGB", tool="editor", shape=None, sources=["upload", "clipboard"], interactive=True, multiple=True)
    duration_in = gr.Slider(1, 600, value=5, step=1, label="Duration (seconds)")
    gen_btn = gr.Button("Generate Video")
    output_video = gr.Video(label="Generated Video", interactive=False)

    gen_btn.click(fn=generate_movie, inputs=[prompt_in, img_inputs, duration_in], outputs=output_video)

if __name__ == "__main__":
    demo.launch()