import torch
from model import Model

model = Model(device = "cuda", dtype = torch.float16)


video_path = '__assets__/canny_videos_mp4/deer_pic.jpeg'
prompt = "Deer walking in the street"
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 2}

prompt = 'oil painting of a deer, a high-quality, detailed, and professional photo'
video_path = '__assets__/frames'
out_path = f'./text2video_edge_guidance_{prompt}.mp4'
model.process_controlnet_draw_frames(video_path, prompt=prompt, save_path=out_path)

