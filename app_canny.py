import gradio as gr
from model import Model


def create_demo(model: Model):

    examples = [
        ["__assets__/canny_videos_edge/butterfly.mp4",
            "white butterfly, a high-quality, detailed, and professional photo"],
        ["__assets__/canny_videos_edge/deer.mp4",
            "oil painting of a deer, a high-quality, detailed, and professional photo"],
        ["__assets__/canny_videos_edge/fox.mp4",
            "wild red fox is walking on the grass, a high-quality, detailed, and professional photo"],
        ["__assets__/canny_videos_edge/girl_dancing.mp4",
            "oil painting of a girl dancing close-up, masterpiece, a high-quality, detailed, and professional photo"],
        ["__assets__/canny_videos_edge/girl_turning.mp4",
            "oil painting of a beautiful girl, a high-quality, detailed, and professional photo"],
        ["__assets__/canny_videos_edge/halloween.mp4",
            "beautiful girl halloween style, a high-quality, detailed, and professional photo"],
        ["__assets__/canny_videos_edge/santa.mp4",
            "a santa claus, a high-quality, detailed, and professional photo"],
    ]

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Text and Canny-Edge Conditional Video Generation')

        with gr.Row():
            gr.Markdown('### You can either use one of the below shown examples, or upload your own video from which edge-motions will be extracted. But Take into account that for now If your uploaded video has more than 8 frames, then we will uniformly select them and our method will run only on them.')

        with gr.Row():
            with gr.Column():
                input_video = gr.Video(
                    label="Input Video", source='upload', format="mp4", visible=True).style(height="auto")
            with gr.Column():
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
                with gr.Accordion('Advanced options', open=False):
                    chunk_size = gr.Slider(
                        label="Chunk size", minimum=2, maximum=8, value=8, step=1)
            with gr.Column():
                result = gr.Video(label="Generated Video").style(height="auto")

        inputs = [
            input_video,
            prompt,
            chunk_size
        ]

        gr.Examples(examples=examples,
                    inputs=inputs,
                    outputs=result,
                    fn=model.process_controlnet_canny,
                    # cache_examples=True,
                    # run_on_click=True,
                    )

        run_button.click(fn=model.process_controlnet_canny,
                         inputs=inputs,
                         outputs=result,)
    return demo
