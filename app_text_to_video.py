import gradio as gr
from model import Model
from functools import partial

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


def create_demo(model: Model):

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Text2Video-Zero: Video Generation')

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
                with gr.Accordion('Advanced options', open=False):
                    motion_field_strength_x = gr.Slider(label='Global Translation $\delta_{x}$',
                                                        minimum=-20,
                                                        maximum=20,
                                                        value=12,
                                                        step=1)

                    motion_field_strength_y = gr.Slider(label='Global Translation $\delta_{y}$',
                                                        minimum=-20,
                                                        maximum=20,
                                                        value=12,
                                                        step=1)
                    t0 = gr.Slider(label="Timestep t0", minimum=0,
                                   maximum=49, value=44, step=1)
                    t1 = gr.Slider(label="Timestep t1", minimum=0,
                                   maximum=49, value=47, step=1)
                    # inject_noise_to_warp = gr.Checkbox(label="add noise to  warp function")
                    chunk_size = gr.Slider(
                        label="Chunk size", minimum=2, maximum=8, value=8, step=1)

                    n_prompt = gr.Textbox(label="Optional Negative Prompt",
                                          value='')
                    video_length = gr.Number(
                        label="Video length", value=8, min=2, precision=0)
            with gr.Column():
                result = gr.Video(label="Generated Video")
        inputs = [
            prompt,
            motion_field_strength_x,
            motion_field_strength_y,
            t0,
            t1,
            n_prompt,
            chunk_size,
            video_length
        ]

        gr.Examples(examples=examples,
                    inputs=inputs,
                    outputs=result,
                    cache_examples=False,
                    run_on_click=False,
                    )

        run_button.click(fn=model.process_text2video,
                         inputs=inputs,
                         outputs=result,)
    return demo
