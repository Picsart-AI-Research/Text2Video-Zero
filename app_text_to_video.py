import gradio as gr
from model import Model
from functools import partial
from bs4 import BeautifulSoup
import requests

examples = [
    ["an astronaut waving the arm on the moon"],
    ["a sloth surfing on a wakeboard"],
    ["an astronaut walking on a street"],
    ["a cute cat walking on grass"],
    ["a horse is galloping on a street"],
    ["an astronaut is skiing down the hill"],
    ["a gorilla walking alone down the street"],
    ["a gorilla dancing on times square"],
    ["A panda dancing dancing like crazy on Times Square"],
    ]


def model_url_list():
    url_list = []
    for i in range(0, 5):
        url_list.append(f"https://huggingface.co/models?p={i}&sort=downloads&search=dreambooth")
    return url_list

def data_scraping(url_list):
    model_list = []
    for url in url_list:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        div_class = 'grid grid-cols-1 gap-5 2xl:grid-cols-2'
        div = soup.find('div', {'class': div_class})
        for a in div.find_all('a', href=True):
            model_list.append(a['href'])
    return model_list

model_list = data_scraping(model_url_list())
for i in range(len(model_list)):
    model_list[i] = model_list[i][1:]

best_model_list = [
    "dreamlike-art/dreamlike-photoreal-2.0",
    "dreamlike-art/dreamlike-diffusion-1.0",
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
    "prompthero/openjourney",
]

model_list = best_model_list + model_list


def create_demo(model: Model):

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Text2Video-Zero: Video Generation')
        with gr.Row():
            gr.HTML(
                """
                <div style="text-align: left; auto;">
                <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
                    Description: Simply input <b>any textual prompt</b> to generate videos right away and unleash your creativity and imagination! You can also select from the examples below. For performance purposes, our current preview release by default generates only 8 output frames and output 4s videos, but you can increase it from Advanced Options.
                </h3>
                </div>
                """)

        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    label="Model",
                    choices=model_list,
                    value="dreamlike-art/dreamlike-photoreal-2.0",
                )
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
                with gr.Accordion('Advanced options', open=False):
                    watermark = gr.Radio(["Picsart AI Research", "Text2Video-Zero", "None"], label="Watermark", value='Picsart AI Research')

                    video_length = gr.Number(label="Video length", value=8, min=2, precision=0)
                    chunk_size = gr.Slider(label="Chunk size", minimum=2, maximum=32, value=8, step=1)

                    motion_field_strength_x = gr.Slider(label='Global Translation $\delta_{x}$', minimum=-20, maximum=20, value=12, step=1)
                    motion_field_strength_y = gr.Slider(label='Global Translation $\delta_{y}$', minimum=-20, maximum=20, value=12, step=1)

                    t0 = gr.Slider(label="Timestep t0", minimum=0, maximum=49, value=44, step=1)
                    t1 = gr.Slider(label="Timestep t1", minimum=0, maximum=49, value=47, step=1)

                    n_prompt = gr.Textbox(label="Optional Negative Prompt", value='')
            with gr.Column():
                result = gr.Video(label="Generated Video")
                
        inputs = [
            prompt,
            model_name,
            motion_field_strength_x,
            motion_field_strength_y,
            t0,
            t1,
            n_prompt,
            chunk_size,
            video_length,
            watermark,
        ]

        gr.Examples(examples=examples,
                inputs=inputs,
                outputs=result,
                fn=model.process_text2video,
                # cache_examples=True,
                run_on_click=False,
        )

        run_button.click(fn=model.process_text2video,
                         inputs=inputs,
                         outputs=result,)
    return demo
