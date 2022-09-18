import streamlit as st
from object_detection import *

ACCEPT_IMG_TYPES = ["jpeg", "png", "jpg"]

st.markdown('Author: Runyu Tian')
st.title('Object Detection App')

content_image = st.sidebar.file_uploader("Load your image below:", type = ACCEPT_IMG_TYPES, accept_multiple_files = False, key = None)

with st.spinner("Loading content image..."):
    if content_image is not None:
        st.subheader("Your content image")
        st.image(content_image,use_column_width = True)
        st.subheader("Run the botton on the left to start your detection!")

    else:
        st.text("Image not uploaded yet. This is sample illustration, Please upload images!")
        st.image("example_in.jpg")
        st.subheader("Example detection from above are below")
        st.text("Please upload your own images and  run button on the left to detection your own image!")
        st.image("example_out.png")
        

clicked = st.sidebar.button("Start detection!") and content_image

if clicked:
    with st.spinner("Image detecting, please wait."):
        st.balloons()
        result = predict(content_image)
        st.subheader("Detection result")
        st.image(result)
        st.download_button(label="Download Final Image", data=save_image(result), file_name="detected_image.png", mime="image/png")
