'''
Author: jielong.wang jielong.wang@akuvox.com
Date: 2024-06-20 17:20:39
LastEditors: jielong.wang jielong.wang@akuvox.com
LastEditTime: 2024-06-20 17:20:44
FilePath: /yolov8_object_tracking/online_demo.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import cv2
import gradio as gr
from pathlib import Path
from typing import List, Generator
import uuid

from ultralytics import YOLO

from utils import read_json, generate_video_from_frames


class YoloV8Track:
    MODEL = [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt",
        "yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt",
        "yolov8n-world.pt", "yolov8s-world.pt", "yolov8m-world.pt", "yolov8l-world.pt", "yolov8x-world.pt",
    ]
    LABEL2CLASSES = read_json("coco_label2class.json")


    def __init__(self, weight: str) -> None:

        assert Path(weight).name in self.MODEL, f"weight must be in {self.MODEL}."
        self.weight = weight
        self.model = self._load_model()
        self.use_yoloworld = "world" in weight
    
    
    def _load_model(self):
        return YOLO(self.weight)
    

    def _reset_yoloworld_model(self, classes: List[str]):
        self.model.set_classes(classes)


    def track(self, video_path: str, targets: str) -> Generator:
        self.model = self._load_model() # reset the model, otherwise will encounter a OpenCV bug. (a cache problem?)
        kwargs = {"persist": True, "device": 0}
        
        if self.use_yoloworld:
            assert len(targets) != 0, "targets is None, please input tracking targets for yolo-world model."
            targets = targets.split(",")
            self._reset_yoloworld_model(targets)
            kwargs.update({"conf": 0.05})

        else:
            if len(targets) != 0:
                targets = targets.split(",")
                for t in targets:
                    assert t in self.LABEL2CLASSES, f"targets must be in {list(self.LABEL2CLASSES.keys())}."
            
                classes = [self.LABEL2CLASSES[t] for t in targets]
                kwargs.update({"classes": classes})

        cap = cv2.VideoCapture(video_path)
        output_frames = []
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = self.model.track(frame, **kwargs)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()[:,:,::-1]
                
                output_frames.append(annotated_frame)
                yield annotated_frame, None

            else:
                # Break the loop if the end of the video is reached
                break
        save_video_file = f"test_dataset/{uuid.uuid4()}.mp4"
        output_video = generate_video_from_frames(output_frames, save_video_file)
        yield annotated_frame, output_video


yolov8_track = YoloV8Track("weights/yolov8x.pt")
yolov8seg_track = YoloV8Track("weights/yolov8x-seg.pt")
yolov8pose_track = YoloV8Track("weights/yolov8x-pose.pt")
yoloworld_track = YoloV8Track("weights/yolov8x-world.pt")


with gr.Blocks() as demo1:
    gr.Markdown("<center><h1>YOLOV8 Tracking</h1></center>")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(scale=2)
            gr.Markdown("Examples")
            gr.Examples(
                examples=["test_dataset/test1.mp4", "test_dataset/test2.mp4"],
                inputs=[
                    video_input
                ],
            )
            targets = gr.Textbox(label="targets", lines=4, placeholder="input like this: person, truck, ...(if do not input, then use coco 80 classes.)")
            text_button = gr.Button("track")

        with gr.Column():
            tracking_output_per_frame = gr.Image(label="video stream")
            video_output = gr.Video(show_label=False)
    
    text_button.click(yolov8_track.track, inputs=[video_input, targets], outputs=[tracking_output_per_frame, video_output]) 

with gr.Blocks() as demo2:
    gr.Markdown("<center><h1>YOLOV8Seg Tracking</h1></center>")
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(scale=2)
            gr.Markdown("Examples")
            gr.Examples(
                examples=["test_dataset/test1.mp4", "test_dataset/test2.mp4"],
                inputs=[
                    video_input
                ],
            )
            targets = gr.Textbox(label="targets", lines=4, placeholder="input like this: person, truck, ...(if do not input, then use coco 80 classes.)")
            text_button = gr.Button("track")

        with gr.Column():
            tracking_output_per_frame = gr.Image(label="video stream")
            video_output = gr.Video(show_label=False)      

    text_button.click(yolov8seg_track.track, inputs=[video_input, targets], outputs=[tracking_output_per_frame, video_output]) 

with gr.Blocks() as demo3:
    gr.Markdown("<center><h1>YOLOV8Pose Tracking</h1></center>")
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(scale=2)
            gr.Markdown("Examples")
            gr.Examples(
                examples=["test_dataset/test1.mp4", "test_dataset/test2.mp4"],
                inputs=[
                    video_input
                ],
            )
            targets = gr.Textbox(label="targets", lines=4, placeholder="input like this: person, truck, ...(if do not input, then use coco 80 classes.)")
            text_button = gr.Button("track")

        with gr.Column():
            tracking_output_per_frame = gr.Image(label="video stream")
            video_output = gr.Video(show_label=False)        

    text_button.click(yolov8pose_track.track, inputs=[video_input, targets], outputs=[tracking_output_per_frame, video_output]) 

with gr.Blocks() as demo4:
    gr.Markdown("<center><h1>YOLOWorld Tracking</h1></center>")
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(scale=2)
            gr.Markdown("Examples")
            gr.Examples(
                examples=["test_dataset/test1.mp4", "test_dataset/test2.mp4"],
                inputs=[
                    video_input
                ],
            )
            targets = gr.Textbox(label="targets", lines=4, placeholder="input (anything you want) like this: person, truck, ...")
            text_button = gr.Button("track")

        with gr.Column():
            tracking_output_per_frame = gr.Image(label="video stream")
            video_output = gr.Video(show_label=False)        

    text_button.click(yoloworld_track.track, inputs=[video_input, targets], outputs=[tracking_output_per_frame, video_output]) 


app = gr.TabbedInterface([demo1, demo2, demo3, demo4], ["YOLOV8 Tracking", "YOLOV8Seg Tracking", "YOLOV8Pose Tracking", "YOLOWorld Tracking"])
app.launch(server_name="0.0.0.0", share=True, show_error=True, debug=True)