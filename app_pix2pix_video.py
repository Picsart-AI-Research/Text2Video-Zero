import gradio as gr
from model import Model


def create_demo(model: Model):
    examples = [
        ['__assets__/pix2pix video/camel.mp4', 'make it Van Gogh Starry Night style'],
        ['__assets__/pix2pix video/mini-cooper.mp4', 'make it Picasso style'],
        ['__assets__/pix2pix video/snowboard.mp4', 'replace man with robot'],
        ['__assets__/pix2pix video/white-swan.mp4', 'replace swan with mallard'],
    ]
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Video Instruct Pix2Pix')
        with gr.Row():
            with gr.Column():
                input_image = gr.Video(label="Input Video",source='upload', type='numpy', format="mp4", visible=True).style(height="auto")
            with gr.Column():
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
                with gr.Accordion('Advanced options', open=False):
                    image_resolution = gr.Slider(label='Image Resolution',
                                                 minimum=256,
                                                 maximum=1024,
                                                 value=512,
                                                 step=64)
                    seed = gr.Slider(label='Seed',
                                     minimum=0,
                                     maximum=65536,
                                     value=0,
                                     step=1)
                    start_t = gr.Slider(label='Starting time in seconds',
                                        minimum=0,
                                        maximum=10,
                                        value=0,
                                        step=1)
                    end_t = gr.Slider(label='End time in seconds (-1 corresponds to uploaded video duration)',
                                      minimum=0,
                                      maximum=10,
                                      value=-1,
                                      step=1)
                    out_fps = gr.Slider(label='Output video fps (-1 corresponds to uploaded video fps)',
                                        minimum=1,
                                        maximum=30,
                                        value=-1,
                                        step=1)
            with gr.Column():
                result = gr.Video(label='Output',
                                    show_label=True)
        inputs = [
            input_image,
            prompt,
            image_resolution,
            seed,
            start_t,
            end_t,
            out_fps
        ]
        gr.Examples(examples, inputs, result)
        # prompt.submit(fn=process, inputs=inputs, outputs=result)
        run_button.click(fn=model.process_pix2pix,
                         inputs=inputs,
                         outputs=result)
    return demo
