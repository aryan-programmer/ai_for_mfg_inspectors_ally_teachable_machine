import io
from collections.abc import Sequence

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
import keras as k
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


@st.cache_resource
def get_model() -> k.Model:
    return load_model("keras_model.h5", compile=False)


# Get the list of class names from the test loader
@st.cache_data
def get_class_names() -> Sequence[str]:
    return [v.split()[1] for v in open("labels.txt", "r").readlines()]


# Define the functions to load images
def load_uploaded_image(file: str) -> Image.Image:
    return Image.open(file)


def buffer_plot_and_get(fig: matplotlib.figure.Figure):
    buf = io.BytesIO()
    fig.savefig(buf, transparent=True)
    buf.seek(0)
    image = Image.open(buf)
    image = image.crop(image.getbbox())
    return image


def anomaly_detection(
    images_and_names: Sequence[tuple[Image.Image, str]],
) -> tuple[Sequence[tuple[Image.Image, str]], Sequence[tuple[Image.Image, str]]]:
    """
    Given an image path and a trained PyTorch model, returns the predicted class and bounding boxes for any defects detected in the image.
    """

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # Determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(len(images_and_names), 224, 224, 3), dtype=np.float32)

    # Resizing the images to be at least 224x224 and then cropping from the center
    size = (224, 224)
    for i, (image, name) in enumerate(images_and_names):
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[i] = normalized_image_array

    model = get_model()
    class_names = get_class_names()

    predictions = model.predict(data)
    indices = np.argmax(predictions, axis=-1)

    good_res: list[tuple[Image.Image, str]] = []
    bad_res: list[tuple[Image.Image, str]] = []
    for i, (image, name) in enumerate(images_and_names):
        index = indices[i]
        class_name = class_names[index]
        confidence_score = predictions[i][index]

        fig = plt.figure()

        # Set title with predicted label, probability, and true label
        caption = images_and_names[i][1]

        # If anomaly is predicted (negative class)
        if class_name == "Anomaly":
            plt.imshow(image)
            plt.title(
                f"Anomalous image {name} with Probability: {confidence_score:.3f}"
            )
            plt.axis("off")
            plt.tight_layout()
            bad_res.append((buffer_plot_and_get(fig), caption))
        else:
            plt.imshow(image)
            plt.title(f"Good image {name} with Probability {confidence_score:.3f}")
            plt.axis("off")
            plt.tight_layout()
            good_res.append((buffer_plot_and_get(fig), caption))

    return (good_res, bad_res)


# Set up the page layout
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:")

page_bg_img = """
<style>
[data-testid="stImageContainer"] {
    padding: 4px;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


st.title("InspectorsAlly: Watch Your Step")

st.caption(
    "Boost Your Quality Control with InspectorsAlly - The Ultimate AI-Powered Inspection App"
)

st.write(
    "Try uploading a floor tile image and watch how an AI Model will classify it between Good / Anomaly."
)

with st.sidebar:
    img = Image.open("./docs/overview_dataset.jpg")
    st.image(img)
    st.subheader("About InspectorsAlly: Watch Your Step")
    st.write(
        "InspectorsAlly is a powerful AI-powered application designed to help businesses streamline their quality control inspections. With InspectorsAlly, companies can ensure that their products meet the highest standards of quality, while reducing inspection time and increasing efficiency."
    )

    st.write(
        "This advanced inspection app uses state-of-the-art computer vision algorithms and deep learning models to perform visual quality control inspections with unparalleled accuracy and speed. InspectorsAlly is capable of identifying even the slightest defects, such as scratches, dents, discolorations, and more on the Floor Tile Product Images."
    )

    st.write(
        "This particular version is focused on analysing anomalies in floor tiles."
    )


# Set up the sidebar
st.subheader("Select Image Input Method")
input_method = st.radio(
    "options", ["File Uploader", "Camera Input"], label_visibility="collapsed"
)

camera_file_img = None
uploaded_file_images = ()

# Check which input method was selected
if input_method == "File Uploader":
    uploaded_files = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    if uploaded_files is not None and len(uploaded_files) > 0:
        uploaded_file_images = tuple(
            (load_uploaded_image(uploaded_file), uploaded_file.name)
            for uploaded_file in uploaded_files
        )
        images, captions = map(list, zip(*uploaded_file_images))
        st.image(images, caption=captions, width=200)
        st.success("Image(s) uploaded successfully!")
    else:
        st.warning("Please upload at least 1 image file.")

elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file is not None:
        camera_file_img = (
            load_uploaded_image(camera_image_file),
            "Camera Input",
        )
        st.image(camera_file_img[0], caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")
    else:
        st.warning("Please click an image.")


submit = st.button(label="Submit Floor Tile Image(s)")
if submit:
    st.subheader("Output")
    img_file_path = ()
    if input_method == "File Uploader" and uploaded_file_images is not None:
        img_file_path = uploaded_file_images
    elif input_method == "Camera Input" and camera_file_img is not None:
        img_file_path = (camera_file_img,)
    if img_file_path is None or len(img_file_path) == 0:
        st.warning("Please specify at least 1 image")
    else:
        with st.spinner(text="This may take a moment..."):
            good_res, bad_res = anomaly_detection(img_file_path)

            def show_images(res: Sequence[tuple[Image.Image, str]]):
                res_even, res_odd = res[::2], res[1::2]

                col1, col2 = st.columns(2)
                with col1:
                    images_even, captions_even = map(list, zip(*res_even))
                    st.image(images_even, caption=captions_even)
                if len(res_odd) != 0:
                    with col2:
                        images_odd, captions_odd = map(list, zip(*res_odd))
                        st.image(images_odd, caption=captions_odd)

            if len(good_res) == 0:
                st.write(
                    "We're sorry to inform you that our AI-based visual inspection system has detected no good products, i.e. all your products are anomalous."
                )
                show_images(bad_res)
            elif len(bad_res) == 0:
                st.write(
                    "Congratulations! All of your products have been classified as 'Good' items with no anomalies detected in the inspection images."
                )
                show_images(good_res)
            else:
                st.write(
                    "Our AI-based visual inspection system has classified all of the below products as 'Good' items with no anomalies detected:"
                )
                show_images(good_res)
                st.write(
                    "However, we're sorry to inform you that the below products have an anomaly in thier inspection images."
                )
                show_images(bad_res)
