import torch
from model import Model
import cv2
import os

model = Model(device = "cuda", dtype = torch.float16)


def images_to_video(directory, output_file):
    # Get all image file names in the directory
    image_files = [os.path.join(directory, f) for f in os.listdir(directory)]

    # Get the first image to get dimensions for the video
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, 1, (width, height))

    # Iterate over all images in the directory and write each one to the video
    for image_file in image_files:
        print(image_file)
        print("processed")
        image = cv2.imread(image_file)
        video_writer.write(image)

    # Release the VideoWriter object and return
    video_writer.release()


video_path = '__assets__/canny_videos_mp4/deer_pic.jpeg'
prompt = "Deer walking in the street"
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 2}

prompt = 'oil painting of a deer, a high-quality, detailed, and professional photo'
images_to_video('__assets__/frames', '__assets__/canny_videos_mp4/myvideo.mp4')
exit()
video_path = '__assets__/canny_videos_mp4/myvideo.mp4'
out_path = f'./final_{prompt}.mp4'
model.process_controlnet_canny(video_path, prompt=prompt, save_path=out_path)

