import time

import cv2
import mediapipe as mp
import torch
from PIL import Image
from torchvision import transforms
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import MEDIAPIPE_PATH
from model_loader import load_model


# -----------------------------
# Model
# -----------------------------
model, device = load_model()

# The checkpoint was trained with ImageNet normalization.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# A = 0, B = 1, ..., Z = 25
idx_to_class: dict[int, str] = {
    i: chr(65 + i) for i in range(26)
}


# -----------------------------
# MediaPipe hand detector
# -----------------------------
base_options = python.BaseOptions(
    model_asset_path=MEDIAPIPE_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

hands = vision.HandLandmarker.create_from_options(options)


# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open the webcam. Close other apps using the camera "
        "and check Windows camera permissions."
    )


# MediaPipe VIDEO mode requires monotonically increasing timestamps.
start_time = time.monotonic()


try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read a frame from the webcam.")
            break

        # Mirror webcam for natural preview.
        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGB.
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        # Generate a continuously increasing timestamp.
        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        results = hands.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # --------------------------------
        # Hand detected
        # --------------------------------
        if results.hand_landmarks:

            landmarks = results.hand_landmarks[0]

            h, w = frame.shape[:2]

            x_coords = [
                lm.x * w
                for lm in landmarks
            ]

            y_coords = [
                lm.y * h
                for lm in landmarks
            ]

            # Padding around hand.
            padding = 40

            x_min = max(
                0,
                int(min(x_coords)) - padding
            )

            y_min = max(
                0,
                int(min(y_coords)) - padding
            )

            x_max = min(
                w,
                int(max(x_coords)) + padding
            )

            y_max = min(
                h,
                int(max(y_coords)) + padding
            )

            # Make sure crop is valid.
            if x_max > x_min and y_max > y_min:

                hand_crop = frame[
                    y_min:y_max,
                    x_min:x_max
                ]

                # OpenCV BGR -> RGB -> PIL
                pil_image = Image.fromarray(
                    cv2.cvtColor(
                        hand_crop,
                        cv2.COLOR_BGR2RGB
                    )
                )

                # --------------------------------
                # Preprocess image
                # --------------------------------
                tensor = preprocess(pil_image)

                # Explicitly tell Pylance this is a Tensor.
                tensor = torch.as_tensor(
                    tensor,
                    dtype=torch.float32
                )

                # Add batch dimension.
                tensor = tensor.unsqueeze(0)

                # Move to CPU/GPU used by model.
                tensor = tensor.to(device)

                # --------------------------------
                # Prediction
                # --------------------------------
                with torch.inference_mode():

                    outputs = model(tensor)

                    probabilities = torch.softmax(
                        outputs,
                        dim=1
                    )

                    confidence_tensor, class_index_tensor = torch.max(
                        probabilities,
                        dim=1
                    )

                # --------------------------------
                # Convert PyTorch values to Python
                # types to avoid Pylance errors.
                # --------------------------------

                predicted_index = int(
                    class_index_tensor.item()
                )

                confidence = float(
                    confidence_tensor.item()
                )

                predicted_class = idx_to_class[
                    predicted_index
                ]

                confidence_percent = (
                    confidence * 100.0
                )

                # --------------------------------
                # Draw bounding box
                # --------------------------------
                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    2,
                )

                # --------------------------------
                # Display prediction
                # --------------------------------
                cv2.putText(
                    frame,
                    (
                        f"{predicted_class}: "
                        f"{confidence_percent:.1f}%"
                    ),
                    (
                        x_min,
                        max(30, y_min - 10)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        # --------------------------------
        # Instructions
        # --------------------------------
        cv2.putText(
            frame,
            "Press Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # --------------------------------
        # Show webcam
        # --------------------------------
        cv2.imshow(
            "ASL Recognition",
            frame
        )

        # Q = quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


finally:

    cap.release()

    cv2.destroyAllWindows()

    hands.close()