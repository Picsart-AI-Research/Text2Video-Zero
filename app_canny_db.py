import gradio as gr
from model import Model
import gradio_utils

examples = [
    ['Anime DB', "woman1", "Portrait of detailed 1girl, feminine, soldier cinematic shot on canon 5d ultra realistic skin intricate clothes accurate hands Rory Lewis Artgerm WLOP Jeremy Lipking Jane Ansell studio lighting"],
    ['Arcane DB', "woman1", "Oil painting of a beautiful girl arcane style, masterpiece, a high-quality, detailed, and professional photo"],
    ['GTA-5 DB', "man1", "gtav style"],
    ['GTA-5 DB', "woman3", "gtav style"],
    ['Avatar DB', "woman2", "oil painting of a beautiful girl avatar style"],
]


def load_db_model(evt: gr.SelectData):
    db_name = gradio_utils.get_db_name_from_id(evt.index)
    return db_name


def canny_select(evt: gr.SelectData):
    canny_name = gradio_utils.get_canny_name_from_id(evt.index)
    return canny_name


def create_demo(model: Model):

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Text, Canny-Edge and DreamBooth Conditional Video Generation')
        with gr.Row():
            gr.HTML(
                """
                <div style="text-align: left; auto;">
                <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
                    Description: Our current release supports only four predefined DreamBooth models and four "motion edges". So you must choose one DreamBooth model and one "motion edges" shown below, or use the examples. The keywords <b>1girl</b>, <b>arcane style</b>, <b>gtav</b>, and <b>avatar style</b> correspond to the models from left to right.
                </h3>
                </div>
                """)

        with gr.Row():
            with gr.Column():
                # input_video_path = gr.Video(source='upload', format="mp4", visible=False)
                gr.Markdown("## Selection")
                db_text_field = gr.Markdown('DB Model: **Anime DB** ')
                canny_text_field = gr.Markdown('Motion: **woman1**')
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
                with gr.Accordion('Advanced options', open=False):
                    watermark = gr.Radio(["Picsart AI Research", "Text2Video-Zero", "None"], label="Watermark", value='Picsart AI Research')
                    chunk_size = gr.Slider(label="Chunk size", minimum=2, maximum=8, value=8, step=1)
            with gr.Column():
                result = gr.Image(label="Generated Video").style(height=400)

        with gr.Row():
            gallery_db = gr.Gallery(label="Db models", value=[('__assets__/db_files/anime.jpg', "anime"), ('__assets__/db_files/arcane.jpg', "Arcane"), ('__assets__/db_files/gta.jpg', "GTA-5 (Man)"), ('__assets__/db_files/avatar.jpg', "Avatar DB")]).style(grid=[4], height=50)
        with gr.Row():
            gallery_canny = gr.Gallery(label="Motions", value=[('__assets__/db_files/woman1.gif', "woman1"), ('__assets__/db_files/woman2.gif', "woman2"), ('__assets__/db_files/man1.gif', "man1"), ('__assets__/db_files/woman3.gif', "woman3")]).style(grid=[4], height=50)
            predefined_motion = gr.Textbox(visible=False, label='One of the above defined motions')

        db_selection = gr.Textbox(label="DB Model", visible=False)
        canny_selection = gr.Textbox(label="One of the above defined motions", visible=False)

        gallery_db.select(load_db_model, None, db_selection)
        gallery_canny.select(canny_select, None, canny_selection)

        db_selection.change(on_db_selection_update,None,db_text_field)
        canny_selection.change(on_canny_selection_update,None,canny_text_field)

        inputs = [
            db_selection,
            canny_selection,
            prompt,
            chunk_size,
            watermark,
        ]

        gr.Examples(examples=examples,
                    inputs=inputs,
                    outputs=result,
                    fn=model.process_controlnet_canny_db,
                    cache_examples = True,
                    )

        run_button.click(fn=model.process_controlnet_canny_db,
                         inputs=inputs,
                         outputs=result,)
    return demo


def on_db_selection_update(evt : gr.EventData):

    return f"DB model: **{evt._data}**"


def on_canny_selection_update(evt: gr.EventData):
    return f"Motion: **{evt._data}**"
