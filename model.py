from enum import Enum
import gc
import numpy as np

import torch
import decord
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDIMScheduler
from text_to_video.text_to_video_pipeline import TextToVideoPipeline

import utils
import gradio_utils

# decord.bridge.set_bridge('torch')


class ModelType(Enum):
    Pix2Pix_Video = 1,
    Text2Video = 2,
    ControlNetCanny = 3,
    ControlNetCannyDB = 4,
    ControlNetPose = 5,


class Model:
    def __init__(self, device, dtype, **kwargs):
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device=device)
        self.pipe_dict = {
            ModelType.Pix2Pix_Video: StableDiffusionInstructPix2PixPipeline,
            ModelType.Text2Video: TextToVideoPipeline,
            ModelType.ControlNetCanny: StableDiffusionControlNetPipeline,
            ModelType.ControlNetCannyDB: StableDiffusionControlNetPipeline,
            ModelType.ControlNetPose: StableDiffusionControlNetPipeline,
        }
        self.controlnet_attn_proc = utils.CrossFrameAttnProcessor(unet_chunk_size=2)
        self.pix2pix_attn_proc = utils.CrossFrameAttnProcessor(unet_chunk_size=3)
        self.text2video_attn_proc = utils.CrossFrameAttnProcessor(unet_chunk_size=2)

        self.pipe = None
        self.model_type = None

        self.states = {}

    def set_model(self, model_type: ModelType, model_id: str, **kwargs):
        if self.pipe is not None:
            del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        safety_checker = kwargs.pop('safety_checker', None)
        self.pipe = self.pipe_dict[model_type].from_pretrained(model_id, safety_checker=safety_checker, **kwargs).to(self.device).to(self.dtype)
        self.model_type = model_type

    def inference_chunk(self, frame_ids, **kwargs):
        if self.pipe is None:
            return

        prompt = np.array(kwargs.pop('prompt'))
        negative_prompt = np.array(kwargs.pop('negative_prompt', ''))
        latents = None
        if 'latents' in kwargs:
            latents = kwargs.pop('latents')[frame_ids]
        if 'image' in kwargs:
            kwargs['image'] = kwargs['image'][frame_ids]
        if 'video_length' in kwargs:
            kwargs['video_length'] = len(frame_ids)
        if self.model_type == ModelType.Text2Video:
            kwargs["frame_ids"] = frame_ids
        return self.pipe(prompt=prompt[frame_ids].tolist(),
                         negative_prompt=negative_prompt[frame_ids].tolist(),
                         latents=latents,
                         generator=self.generator,
                         **kwargs)

    def inference(self, split_to_chunks=False, chunk_size=8, **kwargs):
        if self.pipe is None:
            return
        seed = kwargs.pop('seed', 0)
        if seed < 0:
            seed = self.generator.seed()
        kwargs.pop('generator', '')

        if 'image' in kwargs:
            f = kwargs['image'].shape[0]
        else:
            f = kwargs['video_length']

        assert 'prompt' in kwargs
        prompt = [kwargs.pop('prompt')] * f
        negative_prompt = [kwargs.pop('negative_prompt', '')] * f

        # Processing chunk-by-chunk
        if split_to_chunks:
            chunk_ids = np.arange(0, f, chunk_size - 1)
            result = []
            for i in range(len(chunk_ids)):
                ch_start = chunk_ids[i]
                ch_end = f if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                frame_ids = [0] + list(range(ch_start, ch_end))
                self.generator.manual_seed(seed)
                print(f'Processing chunk {i + 1} / {len(chunk_ids)}')
                result.append(self.inference_chunk(frame_ids=frame_ids,
                                                   prompt=prompt,
                                                   negative_prompt=negative_prompt,
                                                   **kwargs).images[1:])
            result = np.concatenate(result)
            return result
        else:
            return self.pipe(prompt=prompt, negative_prompt=negative_prompt, generator=self.generator, **kwargs).images

    def process_controlnet_canny(self,
                                 video_path,
                                 prompt,
                                 chunk_size=8,
                                 watermark='Picsart AI Research',
                                 num_inference_steps=20,
                                 controlnet_conditioning_scale=1.0,
                                 guidance_scale=9.0,
                                 seed=42,
                                 eta=0.0,
                                 low_threshold=100,
                                 high_threshold=200,
                                 resolution=512,
                                 use_cf_attn=True,
                                 save_path=None):
        video_path = gradio_utils.edge_path_to_video_path(video_path)
        if self.model_type != ModelType.ControlNetCanny:
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
            self.set_model(ModelType.ControlNetCanny,model_id="runwayml/stable-diffusion-v1-5", controlnet=controlnet)
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config)
            if use_cf_attn:
                self.pipe.unet.set_attn_processor(
                    processor=self.controlnet_attn_proc)
                self.pipe.controlnet.set_attn_processor(
                    processor=self.controlnet_attn_proc)

        added_prompt = 'best quality, extremely detailed'
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

        video, fps = utils.prepare_video(
            video_path, resolution, self.device, self.dtype, False)
        control = utils.pre_process_canny(
            video, low_threshold, high_threshold).to(self.device).to(self.dtype)
        f, _, h, w = video.shape
        self.generator.manual_seed(seed)
        latents = torch.randn((1, 4, h//8, w//8), dtype=self.dtype,
                              device=self.device, generator=self.generator)
        latents = latents.repeat(f, 1, 1, 1)
        result = self.inference(image=control,
                                prompt=prompt + ', ' + added_prompt,
                                height=h,
                                width=w,
                                negative_prompt=negative_prompts,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                eta=eta,
                                latents=latents,
                                seed=seed,
                                output_type='numpy',
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                )
        return utils.create_video(result, fps, path=save_path, watermark=gradio_utils.logo_name_to_path(watermark))

    def process_controlnet_pose(self,
                                video_path,
                                prompt,
                                chunk_size=8,
                                watermark='Picsart AI Research',
                                num_inference_steps=20,
                                controlnet_conditioning_scale=1.0,
                                guidance_scale=9.0,
                                seed=42,
                                eta=0.0,
                                resolution=512,
                                use_cf_attn=True,
                                save_path=None):
        video_path = gradio_utils.motion_to_video_path(video_path)
        if self.model_type != ModelType.ControlNetPose:
            controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-openpose")
            self.set_model(ModelType.ControlNetPose, model_id="runwayml/stable-diffusion-v1-5", controlnet=controlnet)
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config)
            if use_cf_attn:
                self.pipe.unet.set_attn_processor(
                    processor=self.controlnet_attn_proc)
                self.pipe.controlnet.set_attn_processor(
                    processor=self.controlnet_attn_proc)

        video_path = gradio_utils.motion_to_video_path(
            video_path) if 'Motion' in video_path else video_path

        added_prompt = 'best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth'
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic'

        video, fps = utils.prepare_video(
            video_path, resolution, self.device, self.dtype, False, output_fps=4)
        control = utils.pre_process_pose(
            video, apply_pose_detect=False).to(self.device).to(self.dtype)
        f, _, h, w = video.shape
        self.generator.manual_seed(seed)
        latents = torch.randn((1, 4, h//8, w//8), dtype=self.dtype,
                              device=self.device, generator=self.generator)
        latents = latents.repeat(f, 1, 1, 1)
        result = self.inference(image=control,
                                prompt=prompt + ', ' + added_prompt,
                                height=h,
                                width=w,
                                negative_prompt=negative_prompts,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                eta=eta,
                                latents=latents,
                                seed=seed,
                                output_type='numpy',
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                )
        return utils.create_gif(result, fps, path=save_path, watermark=gradio_utils.logo_name_to_path(watermark))

    def process_controlnet_canny_db(self,
                                    db_path,
                                    video_path,
                                    prompt,
                                    chunk_size=8,
                                    watermark='Picsart AI Research',
                                    num_inference_steps=20,
                                    controlnet_conditioning_scale=1.0,
                                    guidance_scale=9.0,
                                    seed=42,
                                    eta=0.0,
                                    low_threshold=100,
                                    high_threshold=200,
                                    resolution=512,
                                    use_cf_attn=True,
                                    save_path=None):
        db_path = gradio_utils.get_model_from_db_selection(db_path)
        video_path = gradio_utils.get_video_from_canny_selection(video_path)
        # Load db and controlnet weights
        if 'db_path' not in self.states or db_path != self.states['db_path']:
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
            self.set_model(ModelType.ControlNetCannyDB, model_id=db_path, controlnet=controlnet)
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config)
            self.states['db_path'] = db_path

        if use_cf_attn:
            self.pipe.unet.set_attn_processor(
                processor=self.controlnet_attn_proc)
            self.pipe.controlnet.set_attn_processor(
                processor=self.controlnet_attn_proc)

        added_prompt = 'best quality, extremely detailed'
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

        video, fps = utils.prepare_video(
            video_path, resolution, self.device, self.dtype, False)
        control = utils.pre_process_canny(
            video, low_threshold, high_threshold).to(self.device).to(self.dtype)
        f, _, h, w = video.shape
        self.generator.manual_seed(seed)
        latents = torch.randn((1, 4, h//8, w//8), dtype=self.dtype,
                              device=self.device, generator=self.generator)
        latents = latents.repeat(f, 1, 1, 1)
        result = self.inference(image=control,
                                prompt=prompt + ', ' + added_prompt,
                                height=h,
                                width=w,
                                negative_prompt=negative_prompts,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                eta=eta,
                                latents=latents,
                                seed=seed,
                                output_type='numpy',
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                )
        return utils.create_gif(result, fps, path=save_path, watermark=gradio_utils.logo_name_to_path(watermark))

    def process_pix2pix(self,
                        video,
                        prompt,
                        resolution=512,
                        seed=0,
                        image_guidance_scale=1.0,
                        start_t=0,
                        end_t=-1,
                        out_fps=-1,
                        chunk_size=8,
                        watermark='Picsart AI Research',
                        use_cf_attn=True,
                        save_path=None,):
        if self.model_type != ModelType.Pix2Pix_Video:
            self.set_model(ModelType.Pix2Pix_Video,
                           model_id="timbrooks/instruct-pix2pix")
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config)
            if use_cf_attn:
                self.pipe.unet.set_attn_processor(
                    processor=self.pix2pix_attn_proc)
        video, fps = utils.prepare_video(
            video, resolution, self.device, self.dtype, True, start_t, end_t, out_fps)
        self.generator.manual_seed(seed)
        result = self.inference(image=video,
                                prompt=prompt,
                                seed=seed,
                                output_type='numpy',
                                num_inference_steps=50,
                                image_guidance_scale=image_guidance_scale,
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                )
        return utils.create_video(result, fps, path=save_path, watermark=gradio_utils.logo_name_to_path(watermark))

    def process_text2video(self,
                           prompt,
                           model_name,
                           motion_field_strength_x=12,
                           motion_field_strength_y=12,
                           t0=44,
                           t1=47,
                           n_prompt="",
                           chunk_size=8,
                           video_length=8,
                           watermark='Picsart AI Research',
                           inject_noise_to_warp=False,
                           resolution=512,
                           seed=-1,
                           fps=2,
                           use_cf_attn=True,
                           use_motion_field=True,
                           smooth_bg=False,
                           smooth_bg_strength=0.4,
                           path=None):

        if self.model_type != ModelType.Text2Video:
            unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
            self.set_model(ModelType.Text2Video, model_id=model_name, unet=unet)
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config)
            if use_cf_attn:
                self.pipe.unet.set_attn_processor(
                    processor=self.text2video_attn_proc)
            self.generator.manual_seed(seed)

        added_prompt = "high quality, HD, 8K, trending on artstation, high focus, dramatic lighting"
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic'

        prompt = prompt.rstrip()
        if len(prompt) > 0 and (prompt[-1] == "," or prompt[-1] == "."):
            prompt = prompt.rstrip()[:-1]
        prompt = prompt.rstrip()
        prompt = prompt + ", "+added_prompt
        if len(n_prompt) > 0:
            negative_prompt = n_prompt
        else:
            negative_prompt = None

        result = self.inference(prompt=prompt,
                                video_length=video_length,
                                height=resolution,
                                width=resolution,
                                num_inference_steps=50,
                                guidance_scale=7.5,
                                guidance_stop_step=1.0,
                                t0=t0,
                                t1=t1,
                                motion_field_strength_x=motion_field_strength_x,
                                motion_field_strength_y=motion_field_strength_y,
                                use_motion_field=use_motion_field,
                                smooth_bg=smooth_bg,
                                smooth_bg_strength=smooth_bg_strength,
                                seed=seed,
                                output_type='numpy',
                                negative_prompt=negative_prompt,
                                inject_noise_to_warp=inject_noise_to_warp,
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                )
        return utils.create_video(result, fps, path=path, watermark=gradio_utils.logo_name_to_path(watermark))
