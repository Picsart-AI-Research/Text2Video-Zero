#!/usr/bin/env python

# import os
# import pathlib
# import shlex
# import subprocess

import gradio as gr
import torch

from model import Model, ModelType
from app_canny import create_demo as create_demo_canny
from app_pose import create_demo as create_demo_pose
from app_text_to_video import create_demo as create_demo_text_to_video
from app_pix2pix_video import create_demo as create_demo_pix2pix_video
from app_canny_db import create_demo as create_demo_canny_db
import argparse


model = Model(device='cuda', dtype=torch.float16)
parser = argparse.ArgumentParser()
parser.add_argument('--public_access', action='store_true',
                    help="if enabled, the app can be access from a public url", default=False)
args = parser.parse_args()


with gr.Blocks(css='style.css') as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
            Text2Video-Zero
        </h1>
        <h2 style="font-weight: 450; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
        We propose <b>Text2Video-Zero, the first zero-shot text-to-video synthesis framework</b>, that also natively supports, Video Instruct Pix2Pix, Pose Conditional, Edge Conditional 
        and, Edge Conditional and DreamBooth Specialized applications.
        </h2>
        <h3 style="font-weight: 450; font-size: 1rem; margin: 0rem">
        Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, <a href="https://www.ece.utexas.edu/people/faculty/atlas-wang">Atlas Wang</a>, Shant Navasardyan
        and <a href="https://www.humphreyshi.com/home">Humphrey Shi</a> 
        [<a href="https://arxiv.org/abs/2303.13439" style="color:blue;">arXiv</a>] 
        [<a href="https://github.com/Picsart-AI-Research/Text2Video-Zero" style="color:blue;">GitHub</a>]
        </h3>
        </div>
        """)

    with gr.Tab('Zero-Shot Text2Video'):
        create_demo_text_to_video(model)
    with gr.Tab('Video Instruct Pix2Pix'):
        create_demo_pix2pix_video(model)
    with gr.Tab('Pose Conditional'):
        create_demo_pose(model)
    with gr.Tab('Edge Conditional'):
        create_demo_canny(model)
    with gr.Tab('Edge Conditional and Dreambooth Specialized'):
        create_demo_canny_db(model)

    gr.HTML(
        """
        <div style="text-align: justify; max-width: 1200px; margin: 20px auto;">
        <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        <b>Version: v1.0</b>
        </h3>
        <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        <b>Caution</b>: 
        We would like the raise the awareness of users of this demo of its potential issues and concerns.
        Like previous large foundation models, Text2Video-Zero could be problematic in some cases, partially we use pretrained Stable Diffusion, therefore Text2Video-Zero can Inherit Its Imperfections.
        So far, we keep all features available for research testing both to show the great potential of the Text2Video-Zero framework and to collect important feedback to improve the model in the future.
        We welcome researchers and users to report issues with the HuggingFace community discussion feature or email the authors.
        </h3>
        <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        <b>Biases and content acknowledgement</b>:
        Beware that Text2Video-Zero may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography, and violence. 
        Text2Video-Zero in this demo is meant only for research purposes.
        </h3>
        </div>
        """)

# demo.launch(share=True)
# demo.launch(debug=True)

_, _, link = demo.queue(api_open=False).launch(
    file_directories=['temporal'], share=args.public_access)
print(link)
