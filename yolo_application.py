import streamlit as st
import pandas as pd
import os
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image

# Streamlit app title
st.title("YOLO License Plate Detection with OCR")

# File uploader for images or videos
uploaded_file = st.file_uploader("Upload an image or video file", type=['jpg', 'jpeg', 'png', 'mp4', 'mkv'])

# Load the pre-trained YOLO model
model = YOLO("best_license_plate_model.pt")

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Initialize a DataFrame to store vehicle numbers
vehicle_data_file = "vehicle_data.csv"

# Ensure the CSV file exists
if not os.path.exists(vehicle_data_file):
    pd.DataFrame(columns=["Vehicle Number"]).to_csv(vehicle_data_file, index=False)

def store_vehicle_number(vehicle_number):
    """
    Append a detected vehicle number to the CSV file.
    """
    try:
        new_row = pd.DataFrame({"Vehicle Number": [vehicle_number]})
        new_row.to_csv(vehicle_data_file, mode='a', header=False, index=False)
    except Exception as e:
        st.error(f"Error storing vehicle number: {e}")

def predict_and_save_image(path_test, output_image_path):
    """
    Perform object detection on an image, draw bounding boxes, and extract vehicle numbers.
    """
    try:
        results = model.predict(path_test, device='cpu')
        if results is None or len(results) == 0:
            return None

        image = cv2.imread(path_test)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_image = image[y1:y2, x1:x2]

                ocr_results = ocr.ocr(cropped_image, cls=True)
                if not ocr_results:
                    continue

                for line in ocr_results[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    if confidence > 0.0:  # Confidence threshold
                        store_vehicle_number(text)

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image, f"{text} ({confidence * 100:.2f}%)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )

        cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return output_image_path

    except Exception as e:
        # st.error(f"Error processing image: {e}")
        return None

def process_video_and_save(video_path, output_path):
    """
    Process a video frame-by-frame. Every 10th frame is passed for object detection and OCR.
    The output video contains only the processed frames.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error opening video file: {video_path}")
            return None

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_dir = "temp_frames"
        os.makedirs(frame_dir, exist_ok=True)

        frame_count = 0
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 10 == 0:  # Process every 10th frame
                frame_path = os.path.join(frame_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)

                processed_frame_path = os.path.join(frame_dir, f"processed_frame_{frame_count}.jpg")
                result_path = predict_and_save_image(frame_path, processed_frame_path)

                if result_path and os.path.exists(result_path):
                    processed_frame = cv2.imread(result_path)
                    out.write(processed_frame)  # Write only processed frames
                    processed_frames += 1

            frame_count += 1

        cap.release()
        out.release()

        # Delete temporary frames after processing the video
        for file in os.listdir(frame_dir):
            os.remove(os.path.join(frame_dir, file))
        os.rmdir(frame_dir)

        st.success("Processed video")
        # st.info(f"Total frames processed: {processed_frames}")
        return output_path

    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

def process_media(input_path, output_path):
    """
    Process the uploaded media file (image or video) using the YOLO model.
    """
    try:
        file_extension = os.path.splitext(input_path)[1].lower()

        if file_extension in ['.jpg', '.jpeg', '.png']:
            return predict_and_save_image(input_path, output_path)

        elif file_extension in ['.mp4', '.mkv']:
            return process_video_and_save(input_path, output_path)

        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None

    except Exception as e:
        st.error(f"Error processing media: {e}")
        return None

if uploaded_file is not None:
    input_path = f"temp/{uploaded_file.name}"
    output_path = f"output/{uploaded_file.name}"

    try:
        os.makedirs("temp", exist_ok=True)
        os.makedirs("output", exist_ok=True)

        with open(input_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        st.write("Processing...")
        result_path = process_media(input_path, output_path)

        if result_path:
            with open(result_path, "rb") as file:
                st.download_button(
                    label="Download Processed File",
                    data=file,
                    file_name=os.path.basename(result_path),
                    mime="application/octet-stream"
                )

        else:
            st.warning("No file to display or download!")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
