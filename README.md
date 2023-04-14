


# Text2Video-Zero

This repository is the official implementation of [Text2Video-Zero](https://arxiv.org/abs/2303.13439).


**[Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](https://arxiv.org/abs/2303.13439)**
</br>
Levon Khachatryan,
Andranik Movsisyan,
Vahram Tadevosyan,
Roberto Henschel,
[Zhangyang Wang](https://www.ece.utexas.edu/people/faculty/atlas-wang), Shant Navasardyan, [Humphrey Shi](https://www.humphreyshi.com)
</br>

[Paper](https://arxiv.org/abs/2303.13439) | [Video](https://www.dropbox.com/s/uv90mi2z598olsq/Text2Video-Zero.MP4?dl=0) | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PAIR/Text2Video-Zero) | [Project](https://text2video-zero.github.io/)


<p align="center">
<img src="__assets__/github/teaser/teaser_final.png" width="800px"/>  
<br>
<em>Our method Text2Video-Zero enables zero-shot video generation using (i) a textual prompt (see rows 1, 2),  (ii) a prompt combined with guidance from poses or edges (see lower right), and  (iii)  Video Instruct-Pix2Pix, i.e., instruction-guided video editing (see lower left). 
    Results are temporally consistent and follow closely the guidance and textual prompts.</em>
</p>

## News

* [03/23/2023] Paper [Text2Video-Zero](https://arxiv.org/abs/2303.13439) released!
* [03/25/2023] The [first version](https://huggingface.co/spaces/PAIR/Text2Video-Zero) of our huggingface demo (containing `zero-shot text-to-video generation` and  `Video Instruct Pix2Pix`) released!
* [03/27/2023] The [full version](https://huggingface.co/spaces/PAIR/Text2Video-Zero) of our huggingface demo released! Now also included: `text and pose conditional video generation`, `text and edge conditional video generation`, and 
`text, edge and dreambooth conditional video generation`.
* [03/28/2023] Code for all our generation methods released! We added a new low-memory setup. Minimum required GPU VRAM is currently **12 GB**. It will be further reduced in the upcoming releases. 
* [03/29/2023] Improved [Huggingface demo](https://huggingface.co/spaces/PAIR/Text2Video-Zero)! (i) For text-to-video generation, **any base model for stable diffusion** and **any dreambooth model** hosted on huggingface can now be loaded! (ii) We improved the quality of Video Instruct-Pix2Pix. (iii) We added two longer examples for Video Instruct-Pix2Pix.   
* [03/30/2023] New code released! It includes all improvements of our latest huggingface iteration. See the news update from `03/29/2023`. In addition, generated videos (text-to-video) can have **arbitrary length**. 
* [04/06/2023] We integrated [Token Merging](https://github.com/dbolya/tomesd) into our code. When the highest compression is used and chunk size set to `2`, our code can run with **less than 7 GB VRAM**.  
* [04/11/2023] New code and Huggingface demo released! We integrated **depth control**, based on [MiDaS](https://arxiv.org/pdf/1907.01341.pdf).
* [04/13/2023] Our method has been integrad into 🧨 [Diffusers](https://huggingface.co/docs/diffusers/api/pipelines/text_to_video_zero)!

## Contribute
We are on a journey to democratize AI and empower the creativity of everyone, and we believe Text2Video-Zero is a great research direction to unleash the zero-shot video generation and editing capacity of the amazing text-to-image models!

To achieve this goal, all contributions are welcome. Please check out these external implementations and extensions of Text2Video-Zero. We thank the authors for their efforts and contributions:
* https://github.com/JiauZhang/Text2Video-Zero
* https://github.com/camenduru/text2video-zero-colab
* https://github.com/SHI-Labs/Text2Video-Zero-sd-webui






## Setup



1. Clone this repository and enter:

``` shell
git clone https://github.com/Picsart-AI-Research/Text2Video-Zero.git
cd Text2Video-Zero/
```
2. Install requirements using Python 3.9 and CUDA >= 11.6
``` shell
virtualenv --system-site-packages -p python3.9 venv
source venv/bin/activate
pip install -r requirements.txt
```




--- 



## Inference API


To run inferences create an instance of `Model` class

``` python
import torch
from model import Model

model = Model(device = "cuda", dtype = torch.float16)
```


---


### Text-To-Video
To directly call our text-to-video generator, run this python command which stores the result in `tmp/text2video/A_horse_galloping_on_a_street.mp4` :
``` python
prompt = "A horse galloping on a street"
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8}

out_path, fps = f"./text2video_{prompt.replace(' ','_')}.mp4", 4
model.process_text2video(prompt, fps = fps, path = out_path, **params)
```

To use a different stable diffusion base model run this python command:
``` python
from hf_utils import get_model_list
model_list = get_model_list()
for idx, name in enumerate(model_list):
  print(idx, name)
idx = int(input("Select the model by the listed number: ")) # select the model of your choice
model.process_text2video(prompt, model_name = model_list[idx], fps = fps, path = out_path, **params)
```


#### Hyperparameters (Optional)

You can define the following hyperparameters:
* **Motion field strength**:   `motion_field_strength_x` = $\delta_x$  and `motion_field_strength_y` = $\delta_y$ (see our paper, Sect. 3.3.1). Default: `motion_field_strength_x=motion_field_strength_y= 12`.
* $T$ and $T'$ (see our paper, Sect. 3.3.1). Define values `t0` and `t1` in the range `{0,...,50}`. Default: `t0=44`, `t1=47` (DDIM steps). Corresponds to timesteps `881` and `941`, respectively. 
* **Video length**: Define the number of frames `video_length` to be generated. Default: `video_length=8`.


---


### Text-To-Video with Pose Control
To directly call our text-to-video generator with pose control, run this python command:
``` python
prompt = 'an astronaut dancing in outer space'
motion_path = '__assets__/poses_skeleton_gifs/dance1_corr.mp4'
out_path = f"./text2video_pose_guidance_{prompt.replace(' ','_')}.gif"
model.process_controlnet_pose(motion_path, prompt=prompt, save_path=out_path)
```


---



### Text-To-Video with Edge Control
To directly call our text-to-video generator with edge control, run this python command:
``` python
prompt = 'oil painting of a deer, a high-quality, detailed, and professional photo'
video_path = '__assets__/canny_videos_mp4/deer.mp4'
out_path = f'./text2video_edge_guidance_{prompt}.mp4'
model.process_controlnet_canny(video_path, prompt=prompt, save_path=out_path)
```

#### Hyperparameters

You can define the following hyperparameters for Canny edge detection:
* **low threshold**. Define value `low_threshold` in the range $(0, 255)$. Default: `low_threshold=100`.
* **high threshold**. Define value `high_threshold` in the range $(0, 255)$. Default: `high_threshold=200`. Make sure that `high_threshold` > `low_threshold`.

You can give hyperparameters as arguments to `model.process_controlnet_canny`


---


### Text-To-Video with Edge Guidance and Dreambooth specialization
Load a dreambooth model then proceed as described in `Text-To-Video with Edge Guidance`
``` python

prompt = 'your prompt'
video_path = 'path/to/your/video'
dreambooth_model_path = 'path/to/your/dreambooth/model'
out_path = f'./text2video_edge_db_{prompt}.gif'
model.process_controlnet_canny_db(dreambooth_model_path, video_path, prompt=prompt, save_path=out_path)
```

The value `video_path` can be the path to a `mp4` file. To use one of the example videos provided, set `video_path="woman1"`, `video_path="woman2"`, `video_path="woman3"`, or `video_path="man1"`. 
 

The value `dreambooth_model_path` can either be a link to a diffuser model file, or the name of one of the dreambooth models provided. To this end, set `dreambooth_model_path = "Anime DB"`, `dreambooth_model_path = "Avatar DB"`, `dreambooth_model_path = "GTA-5 DB"`, or `dreambooth_model_path = "Arcane DB"`.  The corresponding keywords are: `1girl` (for `Anime DB`), `arcane style` (for `Arcane DB`) `avatar style` (for `Avatar DB`) and `gtav style`  (for `GTA-5 DB`).


#### Custom Dreambooth Models


To load custom Dreambooth models, [transfer](https://github.com/lllyasviel/ControlNet/discussions/12) control to the custom model and  [convert](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py) it to diffuser format. Then, the value of `dreambooth_model_path` must link to the folder containing the diffuser file. Dreambooth models can be obtained, for instance, from [CIVITAI](https://civitai.com). 



---



### Video Instruct-Pix2Pix

To perform pix2pix video editing, run this python command:
``` python
prompt = 'make it Van Gogh Starry Night'
video_path = '__assets__/pix2pix video/camel.mp4'
out_path = f'./video_instruct_pix2pix_{prompt}.mp4'
model.process_pix2pix(video_path, prompt=prompt, save_path=out_path)
```


---


### Text-To-Video with Depth Control

To directly call our text-to-video generator with depth control, run this python command:
``` python
prompt = 'oil painting of a deer, a high-quality, detailed, and professional photo'
video_path = '__assets__/depth_videos/deer.mp4'
out_path = f'./text2video_depth_control_{prompt}.mp4'
model.process_controlnet_depth(video_path, prompt=prompt, save_path=out_path)
```



---




### Low Memory Inference
Each of the above introduced interface can be run in a low memory setup. In the minimal setup, a GPU with **12 GB VRAM** is sufficient. 

To reduce the memory usage, add `chunk_size=k` as additional parameter when calling one of the above defined inference APIs. The integer value `k` must be in the range `{2,...,video_length}`. It defines the number of frames that are processed at once (without any loss in quality). The lower the value the less memory is needed.

When using the gradio app, set `chunk_size` in the `Advanced options`. 

Thanks to the great work of [Token Merging](https://arxiv.org/abs/2303.17604), the memory usage can be further reduced. It can be configured using the  `merging_ratio` parameter with values in `[0,1]`. The higher the value, the more compression is applied (leading to faster inference and less memory requirements). Be aware that too high values will decrease the image quality. 

 
We plan to continue optimizing our code to enable even lower memory consumption.

---


### Ablation Study
To replicate the ablation study, add additional parameters when calling the above defined inference APIs.
*  To deactivate `cross-frame attention`: Add `use_cf_attn=False` to the parameter list.
* To deactivate enriching latent codes with `motion dynamics`: Add `use_motion_field=False` to the parameter list.


Note: Adding `smooth_bg=True` activates background smoothing. However, our  code does not include the salient object detector necessary to run that code.




---



## Inference using Gradio


<details closed>
<summary>Click to see details.</summary>

From the project root folder, run this shell command:
``` shell
python app.py
```

Then access the app [locally](http://127.0.0.1:7860) with a browser.

To access the app remotely, run this shell command:
``` shell
python app.py --public_access
```
For security information about public access we refer to the documentation of [gradio](https://gradio.app/sharing-your-app/#security-and-file-access).

</details>



---  



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
  <td width=25% align="center">"A cat is running on the grass"</td>
  <td width=25% align="center">"A panda is playing guitar on times square"</td>
  <td width=25% align="center">"A man is running in the snow"</td>
  <td width=25% align="center">"An astronaut is skiing down the hill"</td>
</tr>

<tr>
  <td><img src="__assets__/github/results/t2v/panda_surfing.gif" raw=true></td>
  <td><img src="__assets__/github/results/t2v/bear_dancing.gif"></td>
  <td><img src="__assets__/github/results/t2v/bicycle.gif"></td>              
  <td><img src="__assets__/github/results/t2v/horse_galloping.gif"></td>
</tr>
<tr>
  <td width=25% align="center">"A panda surfing on a wakeboard"</td>
  <td width=25% align="center">"A bear dancing on times square"</td>
  <td width=25% align="center">"A man is riding a bicycle in the sunshine"</td>
  <td width=25% align="center">"A horse galloping on a street"</td>
</tr>

<tr>
  <td><img src="__assets__/github/results/t2v/tiger_walking.gif" raw=true></td>
  <td><img src="__assets__/github/results/t2v/panda_surfing_2.gif"></td>
  <td><img src="__assets__/github/results/t2v/horse_galloping_2.gif"></td>              
  <td><img src="__assets__/github/results/t2v/cat_walking.gif"></td>
</tr>
<tr>
  <td width=25% align="center">"A tiger walking alone down the street"</td>
  <td width=25% align="center">"A panda surfing on a wakeboard"</td>
  <td width=25% align="center">"A horse galloping on a street"</td>
  <td width=25% align="center">"A cute cat running in a beautiful meadow"</td>
</tr>


<tr>
  <td><img src="__assets__/github/results/t2v/horse_galloping_3.gif" raw=true></td>
  <td><img src="__assets__/github/results/t2v/panda_walking.gif"></td>
  <td><img src="__assets__/github/results/t2v/dog_walking.gif"></td>              
  <td><img src="__assets__/github/results/t2v/astronaut.gif"></td>
</tr>
<tr>
  <td width=25% align="center">"A horse galloping on a street"</td>
  <td width=25% align="center">"A panda walking alone down the street"</td>
  <td width=25% align="center">"A dog is walking down the street"</td>
  <td width=25% align="center">"An astronaut is waving his hands on the moon"</td>
</tr>


</table>

### Text-To-Video with Pose Guidance


<table class="center">
<tr>
  <td><img src="__assets__/github/results/pose2v/img_bot_left_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/pose2v/img_bot_right_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/pose2v/img_top_left_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/pose2v/img_top_right_merged_with_input.gif" raw=true></td>
</tr>
<tr>
  <td width=25% align="center">"A bear dancing on the concrete"</td>
  <td width=25% align="center">"An alien dancing under a flying saucer"</td>
  <td width=25% align="center">"A panda dancing in Antarctica"</td>
  <td width=25% align="center">"An astronaut dancing in the outer space"</td>

</tr>
</table>

### Text-To-Video with Edge Guidance



<table class="center">
<tr>
  <td><img src="__assets__/github/results/edge2v/butterfly_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/edge2v/head_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/edge2v/jelly_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/edge2v/mask_merged_with_input.gif" raw=true></td>
</tr>
<tr>
  <td width=25% align="center">"White butterfly"</td>
  <td width=25% align="center">"Beautiful girl"</td>
    <td width=25% align="center">"A jellyfish"</td>
  <td width=25% align="center">"beautiful girl halloween style"</td>
</tr>

<tr>
  <td><img src="__assets__/github/results/edge2v/fox_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/edge2v/head_2_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/edge2v/santa_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/edge2v/dear_merged_with_input.gif" raw=true></td>
</tr>
<tr>
  <td width=25% align="center">"Wild fox is walking"</td>
  <td width=25% align="center">"Oil painting of a beautiful girl close-up"</td>
    <td width=25% align="center">"A santa claus"</td>
  <td width=25% align="center">"A deer"</td>
</tr>

</table>


### Text-To-Video with Edge Guidance and Dreambooth specialization




<table class="center">
<tr>
  <td><img src="__assets__/github/results/canny_db/anime_style_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/canny_db/arcane_style_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/canny_db/gta-5_man_style_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/canny_db/img_bot_right_merged_with_input.gif" raw=true></td>
</tr>
<tr>
  <td width=25% align="center">"anime style"</td>
  <td width=25% align="center">"arcane style"</td>
    <td width=25% align="center">"gta-5 man"</td>
  <td width=25% align="center">"avatar style"</td>
</tr>

</table>


## Video Instruct Pix2Pix

<table class="center">
<tr>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/up_left_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/up_mid_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_1/up_right_merged_with_input.gif" raw=true></td>
</tr>
<tr>
  <td width=25% align="center">"Replace man with chimpanze"</td>
  <td width=25% align="center">"Make it Van Gogh Starry Night style"</td>
    <td width=25% align="center">"Make it Picasso style"</td>
</tr>

<tr>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/up_left_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/up_mid_merged_with_input.gif" raw=true></td>
  <td><img src="__assets__/github/results/Video_InstructPix2Pix/frame_2/up_right_merged_with_input.gif" raw=true></td>
</tr>
<tr>
  <td width=25% align="center">"Make it Expressionism style"</td>
  <td width=25% align="center">"Make it night"</td>
    <td width=25% align="center">"Make it autumn"</td>
</tr>
</table>


## Related Links 

* [High-Resolution Image Synthesis with Latent Diffusion Models (a.k.a. LDM & Stable Diffusion)](https://ommer-lab.com/research/latent-diffusion-models/)
* [InstructPix2Pix: Learning to Follow Image Editing Instructions](https://www.timothybrooks.com/instruct-pix2pix/)
* [Adding Conditional Control to Text-to-Image Diffusion Models (a.k.a ControlNet)](https://github.com/lllyasviel/ControlNet)
* [Diffusers](https://github.com/huggingface/diffusers)
* [Token Merging for Stable Diffusion](https://github.com/dbolya/tomesd)

## License
Our code is published under the CreativeML Open RAIL-M license. The license provided in this repository applies to all additions and contributions we make upon the original stable diffusion code. The original stable diffusion code is under the CreativeML Open RAIL-M license, which can found [here](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE).


## BibTeX
If you use our work in your research, please cite our publication:
```
@article{text2video-zero,
    title={Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators},
    author={Khachatryan, Levon and Movsisyan, Andranik and Tadevosyan, Vahram and Henschel, Roberto and Wang, Zhangyang and Navasardyan, Shant and Shi, Humphrey},
    journal={arXiv preprint arXiv:2303.13439},
    year={2023}
}
```



## Alternative ways to use Text2Video-Zero

Text2Video-Zero can alternatively used via 

* 🧨 [Diffusers](https://github.com/huggingface/diffusers) Library.

<details closed>
<summary>Click to see details.</summary>



### Text2Video-Zero in 🧨 Diffusers Library

Text2Video-Zero is [available](https://huggingface.co/docs/diffusers/api/pipelines/text_to_video_zero) in 🧨 Diffusers, starting from version `0.15.0`! 



[Diffusers](https://github.com/huggingface/diffusers) can be installed using the following command:


``` shell
virtualenv --system-site-packages -p python3.9 venv
source venv/bin/activate
pip install diffusers torch imageio
```


To generate a video from a text prompt, run the following command:

``` python
import torch
import imageio
from diffusers import TextToVideoZeroPipeline

# load stable diffusion model weights
model_id = "runwayml/stable-diffusion-v1-5"

# create a TextToVideoZero pipeline
pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# define the text prompt
prompt = "A panda is playing guitar on times square"

# generate the video using our pipeline
result = pipe(prompt=prompt).images
result = [(r * 255).astype("uint8") for r in result]

# save the resulting image
imageio.mimsave("video.mp4", result, fps=4)
```


For more information, including how to run `text and pose conditional video generation`, `text and edge conditional video generation` and `text and edge and dreambooth conditional video generation`, please check the [documentation](https://huggingface.co/docs/diffusers/api/pipelines/text_to_video_zero).  



</details>

