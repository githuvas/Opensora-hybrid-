from typing import List, Optional
import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip

class VideoComposer:
    def __init__(self, fps: int = 24):
        self.fps = fps

    def compose(
        self,
        frames: List[np.ndarray],
        audio_path: Optional[str] = None,
        out_path: str = "output.mp4",
    ) -> str:
        clip = ImageSequenceClip(frames, fps=self.fps)
        if audio_path is not None:
            try:
                audio = AudioFileClip(audio_path)
                clip = clip.set_audio(audio)
            except Exception:
                print("[WARN] Failed to attach audio – proceeding without it.")
        clip.write_videofile(out_path, codec="libx264", audio_codec="aac", fps=self.fps, verbose=False, logger=None)
        clip.close()
        return out_path