import argparse
import gc
import math
import os
import os.path as osp
import sys
import warnings
from typing import List, Optional, Tuple

import gradio as gr

warnings.filterwarnings('ignore')

# Ensure project root on path
sys.path.insert(0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))

import wan  # noqa: E402
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.utils.prompt_extend import (  # noqa: E402
    DashScopePromptExpander,
    QwenPromptExpander,
)
from wan.utils.utils import cache_video  # noqa: E402
import subprocess  # noqa: E402
import imageio_ffmpeg as iio_ffmpeg  # noqa: E402


# Globals for lazy loading
prompt_expander = None
wan_t2v = None
wan_i2v_480p = None
wan_i2v_720p = None
wan_vace = None


def _nearest_4n_plus_1(value: int) -> int:
    if value <= 1:
        return 1
    # Find n such that 4n+1 is closest to value
    n = max(0, round((value - 1) / 4))
    return 4 * n + 1


def _duration_to_frame_num(seconds: int, fps: int) -> int:
    frames = max(1, int(seconds * fps))
    return _nearest_4n_plus_1(frames)


def _parse_args():
    parser = argparse.ArgumentParser(description="Unified Gradio Movie Maker (Wan2.1)")
    parser.add_argument(
        "--ckpt_dir_t2v",
        type=str,
        default=None,
        help="Checkpoint directory for Wan2.1 T2V (14B or 1.3B).",
    )
    parser.add_argument(
        "--ckpt_dir_i2v_480p",
        type=str,
        default=None,
        help="Checkpoint directory for Wan2.1 I2V-14B-480P.",
    )
    parser.add_argument(
        "--ckpt_dir_i2v_720p",
        type=str,
        default=None,
        help="Checkpoint directory for Wan2.1 I2V-14B-720P.",
    )
    parser.add_argument(
        "--ckpt_dir_vace",
        type=str,
        default=None,
        help="Checkpoint directory for Wan2.1 VACE (1.3B or 14B).",
    )
    parser.add_argument(
        "--vace_model",
        type=str,
        default="vace-1.3B",
        choices=["vace-1.3B", "vace-14B"],
        help="Which VACE model config to use.",
    )
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="Prompt extension method.",
    )
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="Optional model name or path for prompt extension.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Enable public Gradio sharing.",
    )
    return parser.parse_args()


def _lazy_load_t2v():
    global wan_t2v
    if wan_t2v is None:
        assert args.ckpt_dir_t2v is not None, "--ckpt_dir_t2v is required for T2V"
        cfg = WAN_CONFIGS['t2v-14B']
        print("Loading Wan T2V...", end='', flush=True)
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir_t2v,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
        )
        print("done", flush=True)
    return wan_t2v


def _lazy_load_i2v(resolution: str):
    global wan_i2v_480p, wan_i2v_720p
    if resolution == '720P':
        if wan_i2v_720p is None:
            assert args.ckpt_dir_i2v_720p is not None, "--ckpt_dir_i2v_720p is required for 720P I2V"
            print("Loading Wan I2V 720P...", end='', flush=True)
            cfg = WAN_CONFIGS['i2v-14B']
            wan_i2v_720p = wan.WanI2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir_i2v_720p,
                device_id=0,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
            )
            print("done", flush=True)
        return wan_i2v_720p
    else:
        if wan_i2v_480p is None:
            assert args.ckpt_dir_i2v_480p is not None, "--ckpt_dir_i2v_480p is required for 480P I2V"
            print("Loading Wan I2V 480P...", end='', flush=True)
            cfg = WAN_CONFIGS['i2v-14B']
            wan_i2v_480p = wan.WanI2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir_i2v_480p,
                device_id=0,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
            )
            print("done", flush=True)
        return wan_i2v_480p


def _lazy_load_vace():
    global wan_vace
    if wan_vace is None:
        assert args.ckpt_dir_vace is not None, "--ckpt_dir_vace is required for VACE"
        print("Loading Wan VACE...", end='', flush=True)
        wan_vace = wan.WanVace(
            config=WAN_CONFIGS[args.vace_model],
            checkpoint_dir=args.ckpt_dir_vace,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
        )
        print("done", flush=True)
    return wan_vace


def _ensure_prompt_expander(is_vl: bool = False):
    global prompt_expander
    if prompt_expander is None:
        print("Init prompt expander...", end='', flush=True)
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl=is_vl
            )
        else:
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model, is_vl=is_vl, device=0
            )
        print("done", flush=True)
    return prompt_expander


# --------------- Generation callbacks ---------------

def cb_prompt_extend_t2v(prompt: str, tar_lang: str) -> str:
    pe = _ensure_prompt_expander(is_vl=False)
    out = pe(prompt, tar_lang=tar_lang.lower())
    return prompt if not getattr(out, 'status', False) else out.prompt


def cb_prompt_extend_i2v(prompt: str, image, tar_lang: str) -> str:
    if image is None:
        return prompt
    pe = _ensure_prompt_expander(is_vl=True)
    out = pe(prompt, image=image, tar_lang=tar_lang.lower())
    return prompt if not getattr(out, 'status', False) else out.prompt


def cb_t2v(prompt: str, resolution: str, seconds: int, fps: int, sd_steps: int,
           guide_scale: float, shift_scale: float, seed: int, n_prompt: str):
    pipe = _lazy_load_t2v()
    width, height = map(int, resolution.split("*"))
    frame_num = _duration_to_frame_num(seconds, fps)
    video = pipe.generate(
        prompt,
        size=(width, height),
        frame_num=frame_num,
        shift=shift_scale,
        sampling_steps=sd_steps,
        guide_scale=guide_scale,
        n_prompt=n_prompt,
        seed=seed,
        offload_model=True,
    )
    out_path = "movie_t2v.mp4"
    cache_video(
        tensor=video[None],
        save_file=out_path,
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    return out_path, out_path


def _mux_audio(video_path: str, audio_path: Optional[str]) -> str:
    if not audio_path:
        return video_path
    ffmpeg_bin = iio_ffmpeg.get_ffmpeg_exe()
    out_path = os.path.splitext(video_path)[0] + "_with_audio.mp4"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_path
    except Exception:
        return video_path


def cb_t2v_with_audio(prompt: str, resolution: str, seconds: int, fps: int, sd_steps: int,
           guide_scale: float, shift_scale: float, seed: int, n_prompt: str, audio_file: Optional[str]):
    video_path, _ = cb_t2v(prompt, resolution, seconds, fps, sd_steps, guide_scale, shift_scale, seed, n_prompt)
    muxed = _mux_audio(video_path, audio_file)
    return muxed, muxed


def cb_i2v(prompt: str, image, resolution_choice: str, seconds: int, fps: int,
           sd_steps: int, guide_scale: float, shift_scale: float, seed: int,
           n_prompt: str):
    if image is None:
        return None, None
    pipe = _lazy_load_i2v(resolution_choice)
    if resolution_choice == '720P':
        max_area = MAX_AREA_CONFIGS['720*1280']
    else:
        max_area = MAX_AREA_CONFIGS['480*832']
    frame_num = _duration_to_frame_num(seconds, fps)
    video = pipe.generate(
        prompt,
        image,
        max_area=max_area,
        shift=shift_scale,
        sampling_steps=sd_steps,
        guide_scale=guide_scale,
        n_prompt=n_prompt,
        seed=seed,
        frame_num=frame_num,
        offload_model=True,
    )
    out_path = "movie_i2v.mp4"
    cache_video(
        tensor=video[None],
        save_file=out_path,
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    return out_path, out_path


def cb_i2v_with_audio(prompt: str, image, resolution_choice: str, seconds: int, fps: int,
           sd_steps: int, guide_scale: float, shift_scale: float, seed: int,
           n_prompt: str, audio_file: Optional[str]):
    video_path, _ = cb_i2v(prompt, image, resolution_choice, seconds, fps, sd_steps, guide_scale, shift_scale, seed, n_prompt)
    muxed = _mux_audio(video_path, audio_file)
    return muxed, muxed


def cb_vace(prompt: str, ref_images: List[Optional[str]], seconds: int, fps: int,
            width: int, height: int, sd_steps: int, guide_scale: float,
            shift_scale: float, seed: int, n_prompt: str):
    ref_images = [x for x in ref_images if x is not None]
    if len(ref_images) == 0:
        return None, None
    pipe = _lazy_load_vace()
    # prepare_source expects batch lists
    frame_num = _duration_to_frame_num(seconds, fps)
    src_video, src_mask, src_ref_images = pipe.prepare_source(
        [None],
        [None],
        [ref_images],
        num_frames=frame_num,
        image_size=(height, width),
        device=pipe.device,
    )
    video = pipe.generate(
        prompt,
        src_video,
        src_mask,
        src_ref_images,
        size=(width, height),
        frame_num=frame_num,
        shift=shift_scale,
        sampling_steps=sd_steps,
        guide_scale=guide_scale,
        n_prompt=n_prompt,
        seed=seed,
        offload_model=True,
    )
    out_path = "movie_vace.mp4"
    cache_video(
        tensor=video[None],
        save_file=out_path,
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    return out_path, out_path


def cb_vace_with_audio(prompt: str, ref_images: List[Optional[str]], seconds: int, fps: int,
            width: int, height: int, sd_steps: int, guide_scale: float,
            shift_scale: float, seed: int, n_prompt: str, audio_file: Optional[str]):
    video_path, _ = cb_vace(prompt, ref_images, seconds, fps, width, height, sd_steps, guide_scale, shift_scale, seed, n_prompt)
    muxed = _mux_audio(video_path, audio_file)
    return muxed, muxed


# --------------- UI ---------------

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <div style="text-align:center; font-size: 28px; font-weight: 700;">Offline AI Movie Maker (Wan2.1)</div>
            <div style="text-align:center; font-size: 14px;">Text-to-Video, Image-to-Video, and Multi-Reference Consistent Character (VACE). Upload up to 10 images. Control duration and download the result.</div>
            """
        )

        with gr.Tabs():
            # T2V
            with gr.Tab("Text → Video"):
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(label="Prompt", lines=3)
                        tar_lang = gr.Radio(choices=["ZH", "EN"], label="Prompt Enhance Target", value="ZH")
                        btn_extend = gr.Button("Prompt Enhance")
                        with gr.Accordion("Advanced", open=True):
                            resolution = gr.Dropdown(
                                label='Resolution (W*H)',
                                choices=list(SIZE_CONFIGS.keys()),
                                value='1280*720',
                            )
                            seconds = gr.Slider(label="Duration (seconds)", minimum=1, maximum=600, step=1, value=5)
                            fps = gr.Slider(label="FPS", minimum=4, maximum=30, step=1, value=16)
                            sd_steps = gr.Slider(label="Diffusion steps", minimum=1, maximum=1000, value=50, step=1)
                            guide_scale = gr.Slider(label="Guide scale", minimum=0.0, maximum=20.0, value=5.0, step=0.5)
                            shift_scale = gr.Slider(label="Shift scale", minimum=0.0, maximum=20.0, value=5.0, step=0.5)
                            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
                            n_prompt = gr.Textbox(label="Negative Prompt", lines=2)
                        audio_t2v = gr.Audio(label="Optional Audio (wav/mp3)", sources=['upload'], type='filepath')
                        btn_generate = gr.Button("Generate Video")
                    with gr.Column():
                        out_video = gr.Video(label='Generated Video', interactive=False, height=520)
                        out_file = gr.File(label='Download MP4')
                btn_extend.click(cb_prompt_extend_t2v, [prompt, tar_lang], [prompt])
                btn_generate.click(cb_t2v_with_audio, [prompt, resolution, seconds, fps, sd_steps, guide_scale, shift_scale, seed, n_prompt, audio_t2v], [out_video, out_file])

            # I2V
            with gr.Tab("Image → Video"):
                with gr.Row():
                    with gr.Column():
                        image = gr.Image(type="pil", label="Input Image")
                        prompt_i2v = gr.Textbox(label="Prompt", lines=3)
                        tar_lang_i2v = gr.Radio(choices=["ZH", "EN"], label="Prompt Enhance Target", value="ZH")
                        btn_extend_i2v = gr.Button("Prompt Enhance")
                        with gr.Accordion("Advanced", open=True):
                            resolution_choice = gr.Dropdown(label='Model Resolution', choices=['720P', '480P'], value='720P')
                            seconds_i2v = gr.Slider(label="Duration (seconds)", minimum=1, maximum=600, step=1, value=5)
                            fps_i2v = gr.Slider(label="FPS", minimum=4, maximum=30, step=1, value=16)
                            sd_steps_i2v = gr.Slider(label="Diffusion steps", minimum=1, maximum=1000, value=40, step=1)
                            guide_scale_i2v = gr.Slider(label="Guide scale", minimum=0.0, maximum=20.0, value=5.0, step=0.5)
                            shift_scale_i2v = gr.Slider(label="Shift scale", minimum=0.0, maximum=20.0, value=3.0, step=0.5)
                            seed_i2v = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
                            n_prompt_i2v = gr.Textbox(label="Negative Prompt", lines=2)
                        audio_i2v = gr.Audio(label="Optional Audio (wav/mp3)", sources=['upload'], type='filepath')
                        btn_generate_i2v = gr.Button("Generate Video")
                    with gr.Column():
                        out_video_i2v = gr.Video(label='Generated Video', interactive=False, height=520)
                        out_file_i2v = gr.File(label='Download MP4')
                btn_extend_i2v.click(cb_prompt_extend_i2v, [prompt_i2v, image, tar_lang_i2v], [prompt_i2v])
                btn_generate_i2v.click(
                    cb_i2v_with_audio,
                    [prompt_i2v, image, resolution_choice, seconds_i2v, fps_i2v, sd_steps_i2v, guide_scale_i2v, shift_scale_i2v, seed_i2v, n_prompt_i2v, audio_i2v],
                    [out_video_i2v, out_file_i2v],
                )

            # VACE multi-reference (1-10 images)
            with gr.Tab("Multi-Ref Consistent (1-10)"):
                with gr.Row():
                    with gr.Column():
                        prompt_vace = gr.Textbox(label="Prompt", lines=3)
                        # 10 image inputs of filepaths (VACE accepts filepaths or tensors). Use filepath to avoid large memory.
                        with gr.Row():
                            ref_imgs = [
                                gr.Image(label=f"Ref {i+1}", type='filepath', image_mode='RGB', sources=['upload'], height=180)
                                for i in range(5)
                            ]
                        with gr.Row():
                            ref_imgs2 = [
                                gr.Image(label=f"Ref {i+6}", type='filepath', image_mode='RGB', sources=['upload'], height=180)
                                for i in range(5)
                            ]
                        with gr.Accordion("Advanced", open=True):
                            seconds_v = gr.Slider(label="Duration (seconds)", minimum=1, maximum=600, step=1, value=5)
                            fps_v = gr.Slider(label="FPS", minimum=4, maximum=30, step=1, value=16)
                            width = gr.Slider(label="Width", minimum=480, maximum=1280, step=16, value=1280)
                            height = gr.Slider(label="Height", minimum=480, maximum=1280, step=16, value=720)
                            sd_steps_v = gr.Slider(label="Diffusion steps", minimum=1, maximum=1000, value=25, step=1)
                            guide_scale_v = gr.Slider(label="Guide scale", minimum=0.0, maximum=20.0, value=5.0, step=0.5)
                            shift_scale_v = gr.Slider(label="Shift scale", minimum=0.0, maximum=100.0, value=16.0, step=1.0)
                            seed_v = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=2025)
                            n_prompt_v = gr.Textbox(label="Negative Prompt", lines=2)
                        audio_v = gr.Audio(label="Optional Audio (wav/mp3)", sources=['upload'], type='filepath')
                        btn_generate_v = gr.Button("Generate Video")
                    with gr.Column():
                        out_video_v = gr.Video(label='Generated Video', interactive=False, height=520)
                        out_file_v = gr.File(label='Download MP4')
                all_refs = ref_imgs + ref_imgs2
                btn_generate_v.click(
                    cb_vace_with_audio,
                    [prompt_vace, *all_refs, seconds_v, fps_v, width, height, sd_steps_v, guide_scale_v, shift_scale_v, seed_v, n_prompt_v, audio_v],
                    [out_video_v, out_file_v],
                )

        return demo


if __name__ == "__main__":
    args = _parse_args()
    demo = build_ui()
    demo.launch(share=args.share)