import os
import torch
from cog import BasePredictor, Input, Path

from model import Model


class Predictor(BasePredictor):
    def setup(self):
        self.model = Model(device="cuda", dtype=torch.float16)

    def predict(
        self,
        model_name: str = Input(
            description="choose your model, the model should be avaliable on HF",
            default="dreamlike-art/dreamlike-photoreal-2.0",
        ),
        prompt: str = Input(
            description="Input Prompt", default="A horse galloping on a street"
        ),
        negative_prompt: str = Input(description="Negative Prompt", default=""),
        timestep_t0: int = Input(
            description="Perform DDPM steps from t0 to t1. The larger the gap between t0 and t1, the more variance between the frames. Ensure t0 < t1",
            default=44,
        ),
        timestep_t1: int = Input(
            description="Perform DDPM steps from t0 to t1. The larger the gap between t0 and t1, the more variance between the frames. Ensure t0 < t1",
            default=47,
        ),
        motion_field_strength_x: int = Input(default=12, ge=-20, le=20),
        motion_field_strength_y: int = Input(default=12, ge=-20, le=20),
        video_length: int = Input(description="Video length in seconds", default=20),
        fps: int = Input(description="video frames per second", default=4),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        print(f"Using seed: {seed}")

        out = "/tmp/out.mp4"
        self.model.process_text2video(
            prompt=prompt,
            n_prompt=negative_prompt,
            model_name=model_name,
            motion_field_strength_x=motion_field_strength_x,
            motion_field_strength_y=motion_field_strength_y,
            t0=timestep_t0,
            t1=timestep_t1,
            video_length=video_length * fps,
            seed=seed,
            path=out,
            fps=fps,
        )

        return Path(out)
