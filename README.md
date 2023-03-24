# Text2Video-Zero

This repository is the official implementation of [Text2Video-Zero](https://www.dropbox.com/s/ycudlbby9flehyq/Text2Video-Zero.pdf?dl=0).


**[Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators]()**
</br>
[Levon Khachatryan](),
[Andranik Movsisyan](),
[Vahram Tadevosyan](),
[Roberto Henschel](),
[Zhangyang Wang](https://www.ece.utexas.edu/people/faculty/atlas-wang),
[Shant Navasardyan](),
[Humphrey Shi](https://www.humphreyshi.com)
</br>

[Paper](https://www.dropbox.com/s/ycudlbby9flehyq/Text2Video-Zero.pdf?dl=0) | [Video](https://www.dropbox.com/s/uv90mi2z598olsq/Text2Video-Zero.MP4?dl=0) 

<!---
[comment]: #  [Project Page](https://picsart-ai-research.github.io/Text2Video-Zero) | [arXiv]() | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)]()  
--> 

<p align="center">
<img src="__assets__/github/teaser/teaser_final.png" width="800px"/>  
<br>
<em>Our method Text2Video-Zero enables zero-shot video generation using (i) a textual prompt (see rows 1, 2),  (ii) a prompt combined with guidance from poses or edges (see lower right), and  (iii)  Video Instruct-Pix2Pix, i.e., instruction-guided video editing (see lower left). 
    Results are temporally consistent and follow closely the guidance and textual prompts.</em>
</p>

## Code
Will be released soon!


<!---
## Setup


### Requirements

```shell
pip install -r requirements.txt
```
Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for more efficiency and speed on GPUs. 

### Weights



#### Text-To-Video
Any [Stable Diffusion](https://arxiv.org/abs/2112.10752) v1.5 model weights in huggingface format can be used and must be placed in `models/text-to-video`.
For instance:

```shell
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 model_weights
mv model_weights models/text-to-video
```

#### Video Instruct-Pix2Pix
From [Instruct-Pix2Pix](https://arxiv.org/pdf/2211.09800.pdf) download pretrained model files:
```shell
git lfs install
git clone https://huggingface.co/timbrooks/instruct-pix2pix models/instruct-pix2pix
``` 

#### Text-To-Video with Pose Guidance
From [ControlNet](https://arxiv.org/abs/2302.05543), download the open pose model file:
```shell
mkdir -p models/control
wget -P models/control https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth
```
#### Text-To-Video with Edge Guidance
From [ControlNet](https://arxiv.org/abs/2302.05543), download the Canny edge model file:
```shell
mkdir -p models/control
wget -P models/control https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth 
```


#### Text-To-Video with Edge Guidance and Dreambooth

Integrate a `SD1.5` Dreambooth model into ControlNet using [this](https://github.com/lllyasviel/ControlNet/discussions/12) procedure. Load the model into `models/control_db/`. Dreambooth models can be obtained, for instance, from [CIVITAI](https://civitai.com). 

We provide already prepared model files for `anime` (keyword `1girl`), `arcane style` (keyword `arcane style`) `avatar` (keyword `avatar style`) and `gta-5 style`  (keyword `gtav style`). To this end, download the model files from [google drive]() and extract them into `models/control_db/`.

## Inference API

### Text-To-Video
To directly call our text-to-video generator, run this python command:
```python
from text_to_video.text_to_video_generator import TextToVideo
t2v_generator = TextToVideo()
prompt = "A horse galloping on a street"

# run text-to-video, output format 3x1xFxHxW
# with F = number of frames
# H and W = width and height
video = t2v_generator.inference(prompt)
```
You can create gifs by calling
```python
from app_text_to_video import tensor_to_gif
gif_file_name = tensor_to_gif(video)
print(f"The video has been stored as {gif_file_name}")
```

#### Hyperparameters

You can define the following hyperparameters:
* **motion field strength**:   Define value `motion_field_strength`. Then: `motion_field_strength` = $\delta_x = \delta_y$ (see our paper, Sect. 3.3.1). Default: `motion_field_strength=12`.
* $T$ and $T'$ (see our paper, Sect. 3.3.1). Define values `t0` and `t1`. Default: `t0=881`, `t1=941`.
* **video length**: Define the number of frames `video_length` to be generated. Default: `video_length=8`.

To use these hyperparameters, create a custom `TextToVideo` object:
```python
t2v_generator = TextToVideo(motion_field_strength = motion_field_strength, t0 = t0, t1 = t1, video_length = video_length)
```

#### Ablation Study
In order to replicate the ablation study, `cross-frame attention` can be deactivated as follows:

```python
t2v_generator = TextToVideo(use_cf_attn=False)
```
Enriching latents with motion dynamics can be deactivated as follows:
```python
t2v_generator = TextToVideo(use_motion_field=False)
```

---

### Text-To-Video with Pose Control
To directly call our text-to-video generator with pose control, run this python command:
```python
from text_to_video_generator_pose import TextToVideoPose
t2v_pose_generator = TextToVideoPose()

prompt = 'an astronaut dancing in outer space'
motion_path = '__assets__/poses_videos_corrected/dance1.mp4'

video = t2v_pose_generator.inference(prompt, motion_path)
```
You can create gifs by calling
```python
from app_pose import post_process_gif
out_path = 'out_pose.gif'
gif_file_name = post_process_gif(video, out_path)
print(f"The video has been stored as {gif_file_name}")
```

#### Ablation Study
In order to replicate the ablation study, `cross-frame attention` can be deactivated as follows:

```python
t2v_pose_generator = TextToVideoPose(use_cf_attn=False)
```
Enriching latents with motion dynamics can be deactivated as follows:
```python
t2v_pose_generator = TextToVideoPose(use_motion_field=False)
```

---

### Text-To-Video with Edge Control
To directly call our text-to-video generator with edge control, run this python command:
```python
from text_to_video_generator_canny import TextToVideoCanny
t2v_canny_generator = TextToVideoCanny()

prompt = 'oil painting of a deer, a high-quality, detailed, and professional photo'
motion_path = '__assets__/canny_videos_correct/deer_orig.mp4'

video = t2v_canny_generator.inference(prompt, motion_path)
```
You can create gifs by calling
```python
from app_canny import post_process_gif
out_path = 'out_canny.gif'
gif_file_name = post_process_gif(video, out_path)
```

#### Hyperparameters

You can define the following hyperparameters for Canny edge detection:
* **low threshold**. Define value `low` in the range $(0, 255)$. Default: `low=100`.
* **high threshold**. Define value `high` in the range $(0, 255)$. Default: `high=200`. Make sure that `high` > `low`.

To use these hyperparameters, create a custom `TextToVideoCanny` object:
```python
t2v_canny_generator = TextToVideoCanny(low=low, high=high)
```

#### Ablation Study
In order to replicate the ablation study, `cross-frame attention` can be deactivated as follows:

```python
t2v_canny_generator = TextToVideoCanny(use_cf_attn=False)
```
Enriching latents with motion dynamics can be deactivated as follows:
```python
t2v_canny_generator = TextToVideoCanny(use_motion_field=False)
```

---


### Text-To-Video with Edge Guidance and Dreambooth specialization
Load a dreambooth model then proceed as described in `Text-To-Video with Edge Guidance`

---

### Video Instruct-Pix2Pix

**TODO**

## Inference using Gradio
From the project root folder, run this shell command:
```shell
python app.py
```

Then access the app [locally](http://127.0.0.1:7860) with a browser.



-->




## Results

### Text-To-Video
<table class="center">
<tr>
  <td><img src="__assets__/github/results/t2v/cat_running.gif" raw=true></td>
  <td><img src="__assets__/github/results/t2v/playing.gif"></td>
  <td><img src="__assets__/github/results/t2v/running.gif"></td>              
  <td><img src="__assets__/github/results/t2v/skii.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"A cat is running on the grass"</td>
  <td width=25% style="text-align:center;">"A panda is playing guitar on times square</td>
  <td width=25% style="text-align:center;">"A man is running in the snow"</td>
  <td width=25% style="text-align:center;">"An astronaut is skiing down the hill"</td>
</tr>

<tr>
  <td><img src="__assets__/github/results/t2v/panda_surfing.gif" raw=true></td>
  <td><img src="__assets__/github/results/t2v/bear_dancing.gif"></td>
  <td><img src="__assets__/github/results/t2v/bicycle.gif"></td>              
  <td><img src="__assets__/github/results/t2v/horse_galloping.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"A panda surfing on a wakeboard"</td>
  <td width=25% style="text-align:center;">"A bear dancing on times square</td>
  <td width=25% style="text-align:center;">"A man is riding a bicycle in the sunshine"</td>
  <td width=25% style="text-align:center;">"A horse galloping on a street"</td>
</tr>

<tr>
  <td><img src="__assets__/github/results/t2v/tiger_walking.gif" raw=true></td>
  <td><img src="__assets__/github/results/t2v/panda_surfing_2.gif"></td>
  <td><img src="__assets__/github/results/t2v/horse_galloping_2.gif"></td>              
  <td><img src="__assets__/github/results/t2v/cat_walking.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"A tiger walking alone down the street"</td>
  <td width=25% style="text-align:center;">"A panda surfing on a wakeboard</td>
  <td width=25% style="text-align:center;">"A horse galloping on a street"</td>
  <td width=25% style="text-align:center;">"A cute cat running in a beatiful meadow"</td>
</tr>


<tr>
  <td><img src="__assets__/github/results/t2v/horse_galloping_3.gif" raw=true></td>
  <td><img src="__assets__/github/results/t2v/panda_walking.gif"></td>
  <td><img src="__assets__/github/results/t2v/dog_walking.gif"></td>              
  <td><img src="__assets__/github/results/t2v/astronaut.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"A horse galloping on a street"</td>
  <td width=25% style="text-align:center;">"A panda walking alone down the street</td>
  <td width=25% style="text-align:center;">"A dog is walking down the street"</td>
  <td width=25% style="text-align:center;">"An astronaut is waving his hands on the moon"</td>
</tr>


</table>

### Text-To-Video with Pose Guidance


<table class="center">
<tr>
  <td><img src="__assets__/github/results/pose2v/img_bot_left.gif" raw=true><img src="__assets__/github/results/pose2v/pose_bot_left.gif"></td>
  <td><img src="__assets__/github/results/pose2v/img_bot_right.gif" raw=true><img src="__assets__/github/results/pose2v/pose_bot_right.gif"></td>
  <td><img src="__assets__/github/results/pose2v/img_top_left.gif" raw=true><img src="__assets__/github/results/pose2v/pose_top_left.gif"></td>
  <td><img src="__assets__/github/results/pose2v/img_top_right.gif" raw=true><img src="__assets__/github/results/pose2v/pose_top_right.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"A bear dancing on the concrete"</td>
  <td width=25% style="text-align:center;">"An alien dancing under a flying saucer</td>
  <td width=25% style="text-align:center;">"A panda dancing in Antarctica"</td>
  <td width=25% style="text-align:center;">"An astronaut dancing in the outer space"</td>

</tr>
</table>

### Text-To-Video with Edge Guidance



<table class="center">
<tr>
  <td><img src="__assets__/github/results/edge2v/butterfly.gif" raw=true><img src="__assets__/github/results/edge2v/butterfly_edge.gif"></td>
  <td><img src="__assets__/github/results/edge2v/head.gif" raw=true><img src="__assets__/github/results/edge2v/head_edge.gif"></td>
  <td><img src="__assets__/github/results/edge2v/jelly.gif" raw=true><img src="__assets__/github/results/edge2v/jelly_edge.gif"></td>
  <td><img src="__assets__/github/results/edge2v/mask.gif" raw=true><img src="__assets__/github/results/edge2v/mask_edge.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"White butterfly"</td>
  <td width=25% style="text-align:center;">"Beautiful girl</td>
    <td width=25% style="text-align:center;">"A jellyfish"</td>
  <td width=25% style="text-align:center;">"beautiful girl halloween style"</td>
</tr>

<tr>
  <td><img src="__assets__/github/results/edge2v/fox.gif" raw=true><img src="__assets__/github/results/edge2v/fix_edge.gif"></td>
  <td><img src="__assets__/github/results/edge2v/head_2.gif" raw=true><img src="__assets__/github/results/edge2v/head_2_edge.gif"></td>
  <td><img src="__assets__/github/results/edge2v/santa.gif" raw=true><img src="__assets__/github/results/edge2v/santa_edge.gif"></td>
  <td><img src="__assets__/github/results/edge2v/dear.gif" raw=true><img src="__assets__/github/results/edge2v/dear_edge.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"Wild fox is walking"</td>
  <td width=25% style="text-align:center;">"Oil painting of a beautiful girl close-up</td>
    <td width=25% style="text-align:center;">"A santa claus"</td>
  <td width=25% style="text-align:center;">"A deer"</td>
</tr>

</table>


### Text-To-Video with Edge Guidance and Dreambooth specialization




<table class="center">
<tr>
  <td><img src="__assets__/db/anime_style.gif" raw=true><img src="__assets__/db/anime_edge.gif"></td>
  <td><img src="__assets__/db/arcane_style.gif" raw=true><img src="__assets__/db/arcane_edge.gif"></td>
  <td><img src="__assets__/db/gta-5_man_style.gif" raw=true><img src="__assets__/db/gta-5_man_edge.gif"></td>
  <td><img src="__assets__/github/results/canny_db/img_bot_right.gif" raw=true><img src="__assets__/github/results/canny_db/edge_bot_right.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"anime style"</td>
  <td width=25% style="text-align:center;">"arcane style</td>
    <td width=25% style="text-align:center;">"gta-5 man"</td>
  <td width=25% style="text-align:center;">"avar style"</td>
</tr>

</table>


## Video Instruct Pix2Pix

<table class="center">
<tr>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/up_left.gif" raw=true><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/bot_left.gif"></td>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/up_mid.gif" raw=true><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/bot_mid.gif"></td>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/up_right.gif" raw=true><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/bot_right.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"Replace man with chimpanze"</td>
  <td width=25% style="text-align:center;">"Make it Van Gogh Starry Night style"</td>
    <td width=25% style="text-align:center;">"Make it Picasso style"</td>
</tr>

<tr>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/up_left.gif" raw=true><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/bot_left.gif"></td>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/up_mid.gif" raw=true><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/bot_mid.gif"></td>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/up_right.gif" raw=true><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/bot_right.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;">"Make it Expressionism style"</td>
  <td width=25% style="text-align:center;">"Make it night"</td>
    <td width=25% style="text-align:center;">"Make it autumn"</td>
</tr>
</table>

## BibTeX
If you use our work in your research, please cite our publication:
```
@article{text2video-zero,
    title={Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators},
    author={Khachatryan, Levon and Movsisyan, Andranik and Tadevosyan, Vahram and Henschel, Roberto and Wang, Zhangyang and Navasardyan, Shant and Shi, Humphrey},
    journal={arXiv preprint},
    year={2023}
}
```
