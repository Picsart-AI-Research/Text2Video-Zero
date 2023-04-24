import torch
from model import Model
import cv2
import os
from moviepy.editor import ImageSequenceClip

model = Model(device = "cuda", dtype = torch.float16)


def images_to_video(input_dir, output_path, fps=30):
    # Get all image file names in the input directory
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('jpeg')]

    # Sort the file names alphabetically
    image_files.sort()

    # Determine the width and height of the images
    img = cv2.imread(image_files[0])
    height, width, channels = img.shape

    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Iterate over all images and add them to the video
    for image_file in image_files:
        img = cv2.imread(image_file)
        video_writer.write(img)

    # Release the VideoWriter object
    video_writer.release()

def images_to_video_new(input_dir, output_path, fps=30):
    # Get all image file names in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # Sort the file names alphabetically
    image_files.sort()

    # Create an ImageSequenceClip from the images
    clip = ImageSequenceClip([os.path.join(input_dir, f) for f in image_files], fps=fps)

    # Write the clip to a video file
    clip.write_videofile(output_path, fps=fps, codec='libx264')

def video2video(input_path, output_path, prompt):
    model = Model(device = "cuda", dtype = torch.float16)
    model.process_controlnet_canny(input_path, prompt=prompt, save_path=output_path)

#video_path = 'vid2vid/__assets__/canny_videos_mp4/deer_pic.jpeg'
#prompt = "Deer walking in the street"
#params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 2}

#prompt = 'A deer leaping over a hole, in the style of an oil painting”'
#images_to_video_new('__assets__/frames', '__assets__/canny_videos_mp4/kayak_draw.mp4', 1)
#video_path = 'big_deer_16X_20fps.mp4'
#out_path = f'./deer_over_hole.mp4'
#model.process_controlnet_canny(video_path, prompt=prompt, save_path=out_path)

