import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv8 Live", page_icon=" ", layout="wide")

@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)

st.title("YOLOv8 Live Object Detection")
st.write(
    "Runs entirely in your browser tab (webcam required). Press **Stop** to end."
)

col1, col2, col3 = st.columns(3)
model_name = col1.selectbox(
    "Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"], index=0
)
conf = col2.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.01)
device = col3.selectbox("Device hint", ["auto", "cpu", "cuda"], index=0)
classes_text = st.text_input(
    "Limit to classes (comma-separated class names, optional)", value=""
)

model = load_model(model_name)
CLASS_NAME_TO_ID = None
if classes_text.strip():
    # Build a simple name->id mapping from model.names
    name_to_id = {name: idx for idx, name in model.names.items()}
    selected = [s.strip() for s in classes_text.split(",") if s.strip() in name_to_id]
    CLASS_NAME_TO_ID = [name_to_id[s] for s in selected] if selected else None
else:
    CLASS_NAME_TO_ID = None

rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(
            img,
            conf=conf,
            verbose=False,
            device=(None if device == "auto" else device),
            classes=CLASS_NAME_TO_ID
        )
        annotated = results[0].plot()  # BGR with overlays
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="yolov8-live",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration=rtc_config,
)
