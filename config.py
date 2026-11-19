from huggingface_hub import hf_hub_download

REPO_ID = "huzaifanasirrr/realtime-sign-language-translator"

# Download the two files required for inference.
MODEL_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename="best_model.pth",
)

MEDIAPIPE_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename="hand_landmarker.task",
)
