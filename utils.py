'''
Author: jielong.wang jielong.wang@akuvox.com
Date: 2024-06-20 17:20:58
LastEditors: jielong.wang jielong.wang@akuvox.com
LastEditTime: 2024-06-20 17:21:00
FilePath: /yolov8_object_tracking/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import numpy as np
import os

import torch 
import torchvision


def read_json(json_f):
    with open(json_f, "r") as f:
        content = json.load(f)
    return content


def generate_video_from_frames(frames: np.ndarray, output_path:str, fps=30) -> str:
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path
