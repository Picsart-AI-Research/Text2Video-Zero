import os

# App Canny utils


def edge_path_to_video_path(edge_path):
    video_path = edge_path

    vid_name = edge_path.split("/")[-1]
    if vid_name == "butterfly.mp4":
        video_path = "__assets__/canny_videos_mp4/butterfly.mp4"
    elif vid_name == "deer.mp4":
        video_path = "__assets__/canny_videos_mp4/deer.mp4"
    elif vid_name == "fox.mp4":
        video_path = "__assets__/canny_videos_mp4/fox.mp4"
    elif vid_name == "girl_dancing.mp4":
        video_path = "__assets__/canny_videos_mp4/girl_dancing.mp4"
    elif vid_name == "girl_turning.mp4":
        video_path = "__assets__/canny_videos_mp4/girl_turning.mp4"
    elif vid_name == "halloween.mp4":
        video_path = "__assets__/canny_videos_mp4/halloween.mp4"
    elif vid_name == "santa.mp4":
        video_path = "__assets__/canny_videos_mp4/santa.mp4"

    assert os.path.isfile(video_path)
    return video_path


# App Pose utils
def motion_to_video_path(motion):
    videos = [
        "__assets__/poses_skeleton_gifs/dance1_corr.mp4",
        "__assets__/poses_skeleton_gifs/dance2_corr.mp4",
        "__assets__/poses_skeleton_gifs/dance3_corr.mp4",
        "__assets__/poses_skeleton_gifs/dance4_corr.mp4",
        "__assets__/poses_skeleton_gifs/dance5_corr.mp4"
    ]
    if len(motion.split(" ")) > 1 and motion.split(" ")[1].isnumeric():
        id = int(motion.split(" ")[1]) - 1
        return videos[id]
    else:
        return motion


# App Canny Dreambooth utils
def get_video_from_canny_selection(canny_selection):
    if canny_selection == "woman1":
        input_video_path = "__assets__/db_files_2fps/woman1.mp4"

    elif canny_selection == "woman2":
        input_video_path = "__assets__/db_files_2fps/woman2.mp4"

    elif canny_selection == "man1":
        input_video_path = "__assets__/db_files_2fps/man1.mp4"

    elif canny_selection == "woman3":
        input_video_path = "__assets__/db_files_2fps/woman3.mp4"
    else:
        input_video_path = canny_selection

    assert os.path.isfile(input_video_path)
    return input_video_path


def get_model_from_db_selection(db_selection):
    if db_selection == "Anime DB":
        input_video_path = 'PAIR/text2video-zero-controlnet-canny-anime'
    elif db_selection == "Avatar DB":
        input_video_path = 'PAIR/text2video-zero-controlnet-canny-avatar'
    elif db_selection == "GTA-5 DB":
        input_video_path = 'PAIR/text2video-zero-controlnet-canny-gta5'
    elif db_selection == "Arcane DB":
        input_video_path = 'PAIR/text2video-zero-controlnet-canny-arcane'
    else:
        input_video_path = db_selection

    return input_video_path


def get_db_name_from_id(id):
    db_names = ["Anime DB", "Arcane DB", "GTA-5 DB", "Avatar DB"]
    return db_names[id]


def get_canny_name_from_id(id):
    canny_names = ["woman1", "woman2", "man1", "woman3"]
    return canny_names[id]


def logo_name_to_path(name):
    logo_paths = {
        'Picsart AI Research': '__assets__/pair_watermark.png',
        'Text2Video-Zero': '__assets__/t2v-z_watermark.png',
        'None': None
    }
    if name in logo_paths:
        return logo_paths[name]
    return name
