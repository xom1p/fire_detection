import streamlit as st
import utils
import cv2
import numpy as np
import io
import PIL
from PIL import Image
from camera_input_live import camera_input_live

def play_video(video_source):
    camera = cv2.VideoCapture(video_source)

    st_frame = st.empty()
    while(camera.isOpened()):
        ret, frame = camera.read()

        if ret:
            visualized_image = utils.predict_image(frame, conf_threshold)
            st_frame.image(visualized_image, channels = "BGR")
        else:
            camera.release()
            break

st.set_page_config(
    page_title="Fire/smoke-detection",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="expanded")

st.title("Fire/smoke-detection Project :fire:")
source_radio = st.sidebar.radio("Select Source",["IMAGE","VIDEO","WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 10, 100, 20))/100

input = None
if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg","png"))
    
    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv =cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold = conf_threshold)
        st.image(visualized_image, channels = "BGR")


temporary_location = None
if source_radio == "VIDEO":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an video.", type=("mp4"))

    if input is not None:
        g = io.BytesIO(input.read())
        temporary_location = "upload.mp4"

        with open(temporary_location, "wb") as out:
            out.write(g.read())

        out.close()
    if temporary_location is not None:
        play_video(temporary_location)
        if st.button("Replay", type="primary"):
            pass


def play_live_camera():
    image = camera_input_live()
    uploaded_image = PIL.Image.open(image)
    uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
    visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold)
    st.image(visualized_image, channels = "BGR")

if source_radio == "WEBCAM":
    play_live_camera()
