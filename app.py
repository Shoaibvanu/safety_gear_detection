import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import os

st.title("Object Detection on Video")

# Load YOLOv5 model
@st.cache(allow_output_mutation=True, hash_funcs={torch.nn.modules.module.Module: id})
def load_model():
    path = 'D:/yolov5safetyhelmet-main/yolov5safetyhelmet-main/final.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)
    return model

# Function to perform object detection on a frame
def detect_objects(model, frame):
    results = model(frame)
    rendered_frame = np.squeeze(results.render())
    return rendered_frame

# Main function to run object detection on the video
def main():
    model = load_model()

    uploaded_file = st.file_uploader("Choose a video file", accept_multiple_files=False)
    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, "temp_video")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_file_path)

        # Get video details
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define codec and VideoWriter object
        output_file_path = os.path.join(temp_dir.name, "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            output_frame_rgb = detect_objects(model, frame_rgb)
            output_frame_bgr = cv2.cvtColor(output_frame_rgb, cv2.COLOR_RGB2BGR)  # Convert back to BGR
            out.write(output_frame_bgr)

        cap.release()
        out.release()

        # Display the output video file
        st.video(output_file_path)

        # Cleanup temporary directory
        temp_dir.cleanup()

if __name__ == "__main__":
    main()

