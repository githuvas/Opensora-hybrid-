import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional
import random

class CharacterGenerator:
    """Stub implementation that returns placeholder frames.

    Swap `generate_frames` with your Stable Diffusion + IP-Adapter pipeline to
    unlock real character generation.
    """

    def __init__(self, fps: int = 24, resolution: tuple[int, int] = (512, 512)):
        self.fps = fps
        self.resolution = resolution

    def _make_placeholder(self, text: str, frame_idx: int) -> Image.Image:
        """Create a simple RGB image with text – demo only."""
        w, h = self.resolution
        img = Image.new("RGB", (w, h), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        draw = ImageDraw.Draw(img)
        msg = f"{text[:30]}...\nFrame {frame_idx}"
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except Exception:
            font = None
        draw.multiline_text((10, 10), msg, fill=(255, 255, 255), font=font)
        return img

    def generate_frames(
        self,
        prompt: str,
        ref_images: Optional[List[Image.Image]],
        duration_s: int,
        fps: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Return a list of RGB numpy arrays representing video frames."""
        fps = fps or self.fps
        frame_count = max(1, int(duration_s * fps))
        frames = [np.array(self._make_placeholder(prompt, i)) for i in range(frame_count)]
        return frames