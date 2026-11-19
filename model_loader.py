import torch
import torch.nn as nn
from torchvision import models

from config import MODEL_PATH


class SignLanguageModel(nn.Module):
    """ResNet18 + the classification head used by the Hugging Face checkpoint."""

    def __init__(self, num_classes=26):
        super().__init__()

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # This checkpoint is a trusted model file downloaded from the repository.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model = SignLanguageModel(num_classes=26)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Also support a plain state_dict checkpoint.
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    val_acc = checkpoint.get("val_acc") if isinstance(checkpoint, dict) else None
    if val_acc is not None:
        print(f"Model loaded successfully. Validation accuracy: {float(val_acc):.2f}%")
    else:
        print("Model loaded successfully.")

    print(f"Device: {device}")
    return model, device
