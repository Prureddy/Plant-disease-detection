import tensorflow as tf
import os
import cv2
import streamlit as st
import numpy as np
from keras.preprocessing import image as keras_image
from utils import label_map_util
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
st.set_page_config(page_title="Plant Disease Detection and Classification")


# Set environment variable to use pure Python implementation of protobuf
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Disable oneDNN custom operations to avoid numerical inconsistencies
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ensure TensorFlow eager execution is enabled
tf.compat.v1.enable_eager_execution()

# Set model paths
MODEL_NAME = './object_detection/inference_graph'
IMAGE_NAME = './object_detection/images/out.jpg'

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, 'labelmap.pbtxt')
PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

NUM_CLASSES = 6

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load TensorFlow model into memory
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Function to draw bounding box and labels on the image
def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, thickness=3, display_str_list=()):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)
    im_width, im_height = image_pil.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)

    # Define the font and text box position
    try:
        font = ImageFont.truetype('arial.ttf', 24)  # Replace 'arial.ttf' with your font file
    except IOError:
        font = ImageFont.load_default()

    display_str_heights = [font.getbbox(ds)[3] - font.getbbox(ds)[1] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height

    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getbbox(display_str)[2] - font.getbbox(display_str)[0], font.getbbox(display_str)[3] - font.getbbox(display_str)[1]
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin

    np.copyto(image, np.array(image_pil))
    return image

# Function to perform object detection
def perform_detection(image_path):
    # Load image using OpenCV
    in_image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # Perform detection
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})

    # Visualize detection results using draw_bounding_box_on_image
    for i in range(len(boxes[0])):
        if scores[0][i] > 0.5:
            ymin, xmin, ymax, xmax = tuple(boxes[0][i])
            draw_bounding_box_on_image(
                in_image,
                ymin, xmin, ymax, xmax,
                color=(255, 0, 0),  # Red color for bounding box
                thickness=4,
                display_str_list=['Disease'])  # Example display string list

    return in_image

# List of class names corresponding to the model's output
LABELS = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy',
    'Cherry___healthy','Cherry___Powdery_mildew','Grape___Black_rot','Grape___Esca_Black_Measles','Grape___healthy',
    'Grape___Leaf_blight_Isariopsis_Leaf_Spot','Orange___Haunglongbing','Peach___Bacterial_spot','Peach___healthy',
    'Pepper_bell___Bacterial_spot','Pepper_bell___healthy','Potato___Early_blight','Potato___healthy','Potato___Late_blight',
    'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___healthy','Strawberry___Leaf_scorch'
]

def classify_disease(image_path):
    # Load and preprocess image
    img = keras_image.load_img(image_path, target_size=(150, 150))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Define your classification model (example)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(len(LABELS), activation='softmax')
    ])

    # Load pre-trained weights (example)
    model.load_weights("./object_classification/rps.h5")

    # Predict classes
    predictions = model.predict(x, batch_size=10)

    # Get the class with the highest probability
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = LABELS[predicted_class_index]
    predicted_probability = np.max(predictions)

    return predicted_class_name, predicted_probability

# Streamlit application
import streamlit as st
from PIL import Image

def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Create the navbar
    st.markdown(
        f"""
         <nav style="background-color: #b82f0d; padding: 15px; border-radius: 5px;">
            <a href="#" style="color: white; text-decoration: none; margin-right: 20px; font-weight: bold;">Home</a>
            <a href="#" style="color: white; text-decoration: none; margin-right: 20px; font-weight: bold;">Plant Disease</a>
            <a href="#" style="color: white; text-decoration: none; font-weight: bold;">About</a>
        </nav>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='text-align: left; color: #00bfff;'>Plant Disease Detection & Classification</h1>", unsafe_allow_html=True)
    
    activities = ["Home", "Plant Disease"]
    choice = st.sidebar.selectbox("Select Activity", activities, label_visibility="collapsed")

    if choice == 'Home':
        st.markdown("""
        ## Welcome to the Plant Disease Detection & Classification App!
        
        <p style='color: #fff;'>This application helps in detecting and classifying diseases in plant leaves. It's built using <b>Streamlit</b> and <b>TensorFlow</b>.</p>
        
        ### Key Features:
        - **Detection:** Identify regions in the image where diseases are present.
        - **Classification:** Classify the type of disease present in the leaf.
        
        ### How to Use:
        1. Navigate to the **Plant Disease** section.
        2. Choose either **Detection** or **Classification**.
        3. Upload an image of a plant leaf.
        4. Click **Process** or **Classify** to see the results.
        
        ### About the Project:
        This project leverages advanced machine learning techniques to assist farmers and agricultural experts in identifying and managing plant diseases, ensuring better crop health and yield.
        """, unsafe_allow_html=True)
        st.markdown("* * *")

    elif choice == 'Plant Disease':
        st.markdown("* * *")

        enhance_type = st.sidebar.radio("Type", ["Detection", "Classification"], label_visibility="collapsed")

        if enhance_type == 'Detection':
            st.markdown("<h3 style='color: #00ff7f;'>Plant Disease Detection</h3>", unsafe_allow_html=True)
            image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

            if image_file is not None:
                image_path = './object_detection/images/out.jpg'
                img = Image.open(image_file)
                img.save(image_path)

                if st.button('Process'):
                    processed_image = perform_detection(image_path)
                    st.image(processed_image, use_column_width=True, channels='RGB')

        elif enhance_type == 'Classification':
            st.markdown("<h3 style='color: #00ff7f;'>Plant Disease Classification</h3>", unsafe_allow_html=True)
            image_input = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

            if image_input is not None:
                image_path = './object_classification/images/out.jpg'
                img = Image.open(image_input)
                img.save(image_path)

                if st.button('Classify'):
                    class_name, probability = classify_disease(image_path)
                    st.markdown(f"**Class:** <span style='font-size:24px; color: #ff0000;'>{class_name}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Probability:** <span style='font-size:24px; color: #ff0000;'>{probability:.2f}</span>", unsafe_allow_html=True)
                    st.balloons()

if __name__ == "__main__":
    main()