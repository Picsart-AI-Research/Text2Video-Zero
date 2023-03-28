import gradio as gr
import os

from model import Model

examples = [
    ['Motion 1', "An astronaut dancing in the outer space"],
    ['Motion 2', "An astronaut dancing in the outer space"],
    ['Motion 3', "An astronaut dancing in the outer space"],
    ['Motion 4', "An astronaut dancing in the outer space"],
    ['Motion 5', "An astronaut dancing in the outer space"],
]

def create_demo(model: Model):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Text and Pose Conditional Video Generation')

        with gr.Row():
            gr.Markdown('Selection: **one motion** and a **prompt**, or use the examples below.')
            with gr.Column():
                gallery_pose_sequence = gr.Gallery(label="Motions", value=[('__assets__/poses_skeleton_gifs/dance1.gif', "Motion 1"), ('__assets__/poses_skeleton_gifs/dance2.gif', "Motion 2"), ('__assets__/poses_skeleton_gifs/dance3.gif', "Motion 3"), ('__assets__/poses_skeleton_gifs/dance4.gif', "Motion 4"), ('__assets__/poses_skeleton_gifs/dance5.gif', "Motion 5")]).style(grid=[2], height="auto")
                input_video_path = gr.Textbox(label="Motion",visible=False,value="Motion 1")
                
                pose_sequence_selector = gr.Markdown('Selection: **Motion 1**')
            with gr.Column():
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
            with gr.Column():
                result = gr.Image(label="Generated Video")

        input_video_path.change(on_video_path_update, None, pose_sequence_selector)
        gallery_pose_sequence.select(pose_gallery_callback, None, input_video_path)
        inputs = [
            input_video_path, 
            prompt,
        ]

        gr.Examples(examples=examples,
                    inputs=inputs,
                    outputs=result,
                    cache_examples=os.getenv('SYSTEM') == 'spaces',
                    fn=model.process_controlnet_pose,
                    run_on_click=False,
                    )

        run_button.click(fn=model.process_controlnet_pose,
                         inputs=inputs,
                         outputs=result,)

    return demo


def on_video_path_update(evt: gr.EventData):
    return f'Selection: **{evt._data}**'

def pose_gallery_callback(evt: gr.SelectData):
    return f"Motion {evt.index+1}"
