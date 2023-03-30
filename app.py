import gradio as gr
import torch

from model import Model, ModelType

from app_canny import create_demo as create_demo_canny
from app_pose import create_demo as create_demo_pose
from app_text_to_video import create_demo as create_demo_text_to_video
from app_pix2pix_video import create_demo as create_demo_pix2pix_video
from app_canny_db import create_demo as create_demo_canny_db


model = Model(device='cuda', dtype=torch.float16)

with gr.Blocks(css='style.css') as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
            Text2Video-Zero
        </h1>
        <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
        Levon Khachatryan<sup>1*</sup>, Andranik Movsisyan<sup>1*</sup>, Vahram Tadevosyan<sup>1*</sup>, Roberto Henschel<sup>1*</sup>, Zhangyang Wang<sup>1,2</sup>, Shant Navasardyan<sup>1</sup>
        and <a href="https://www.humphreyshi.com/home">Humphrey Shi</a><sup>1,3,4</sup>
        </h2>
        <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
        <sup>1</sup>Picsart AI Resarch (PAIR), <sup>2</sup>UT Austin, <sup>3</sup>U of Oregon, <sup>4</sup>UIUC
        </h2>
        <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
        [<a href="https://arxiv.org/abs/2303.13439" style="color:blue;">arXiv</a>] 
        [<a href="https://github.com/Picsart-AI-Research/Text2Video-Zero" style="color:blue;">GitHub</a>]
        </h2>
        <h2 style="font-weight: 450; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
        We built <b>Text2Video-Zero</b>,  a first zero-shot text-to-video synthesis diffusion framework, that enables low cost yet high-quality and consistent video generation with only pre-trained text-to-image diffusion models without any training on videos or optimization!
        Text2Video-Zero also naturally supports cool extension works of pre-trained text-to-image models such as Instruct Pix2Pix, ControlNet and DreamBooth, and based on which we present Video Instruct Pix2Pix, Pose Conditional, Edge Conditional and, Edge Conditional and DreamBooth Specialized applications.
        We hope our Text2Video-Zero will further democratize AI and empower the creativity of everyone by unleashing the zero-shot video generation and editing capacity of the amazing text-to-image models and encourage future research!
        </h2>
        </div>
        """)

    
    gr.HTML("""
    <p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.
    <br/>
    <a href="https://huggingface.co/spaces/PAIR/Text2Video-Zero?duplicate=true">
    <img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
    </p>""")

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
