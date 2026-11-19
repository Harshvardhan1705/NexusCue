import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from torchvision.models import resnet18, ResNet18_Weights
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# ============================================================
# CONFIGURATION
# ============================================================

BASE_PATH = (
    r"D:\Mini project HHAI\nexus\Dataset\ISL_CSLRT_Corpus"
    r"\ISL_CSLRT_Corpus\Frames_Word_Level"
)

TEST_SIGNS = [
    "YOU",
    "FREE",
    "TODAY",
    "HAPPY",
    "HOW"
]

# Dataset documentation says there are 7 signers.
# We start with 7 groups for this experiment.
NUMBER_OF_SIGNER_GROUPS = 7

OUTPUT_FILE = "signer_groups.txt"


# ============================================================
# LOAD IMAGES
# ============================================================

def get_images():

    image_paths = []

    for sign in TEST_SIGNS:

        folder = os.path.join(
            BASE_PATH,
            sign
        )

        if not os.path.exists(folder):
            print(f"⚠️ Folder not found: {folder}")
            continue

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        files.sort()

        for filename in files:

            image_paths.append(
                (
                    sign,
                    os.path.join(folder, filename)
                )
            )

    return image_paths


# ============================================================
# LOAD RESNET
# ============================================================

def create_model():

    print("\nLoading pretrained ResNet18...")

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(
        weights=weights
    )

    # Remove final classification layer.
    # We want image features, not ImageNet classes.
    model.fc = torch.nn.Identity()

    model.eval()

    transform = weights.transforms()

    print("✅ ResNet18 loaded.")

    return model, transform


# ============================================================
# EXTRACT IMAGE FEATURE
# ============================================================

def extract_feature(
    model,
    transform,
    image_path
):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Convert OpenCV image to PIL.
    from PIL import Image

    image = Image.fromarray(image)

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        feature = model(
            tensor
        )

    feature = feature.squeeze().numpy()

    return feature


# ============================================================
# EXTRACT ALL FEATURES
# ============================================================

def extract_all_features(
    model,
    transform,
    image_paths
):

    features = []
    valid_images = []

    total = len(image_paths)

    print("\nExtracting image features...")
    print("-" * 60)

    for index, (sign, image_path) in enumerate(
        image_paths,
        start=1
    ):

        feature = extract_feature(
            model,
            transform,
            image_path
        )

        if feature is None:
            continue

        features.append(feature)

        valid_images.append(
            (
                sign,
                image_path
            )
        )

        if (
            index % 10 == 0
            or index == total
        ):

            print(
                f"Processed: "
                f"{index}/{total}"
            )

    return (
        np.array(features),
        valid_images
    )


# ============================================================
# CLUSTER IMAGES
# ============================================================

def create_groups(features):

    print("\nNormalizing features...")

    features = normalize(
        features
    )

    print(
        f"\nCreating "
        f"{NUMBER_OF_SIGNER_GROUPS} "
        f"groups..."
    )

    clustering = AgglomerativeClustering(
        n_clusters=NUMBER_OF_SIGNER_GROUPS,
        metric="cosine",
        linkage="average"
    )

    labels = clustering.fit_predict(
        features
    )

    return labels


# ============================================================
# DISPLAY GROUPS
# ============================================================

def display_groups(
    labels,
    image_paths
):

    groups = {}

    for label, item in zip(
        labels,
        image_paths
    ):

        if label not in groups:
            groups[label] = []

        groups[label].append(
            item
        )

    # Sort groups by size.
    groups = dict(
        sorted(
            groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
    )

    print("\n")
    print("=" * 70)
    print("SIGNER GROUPING RESULTS")
    print("=" * 70)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        output.write(
            "SIGNER GROUPING RESULTS\n"
        )

        output.write(
            "=" * 70 + "\n\n"
        )

        for new_number, (
            label,
            items
        ) in enumerate(
            groups.items(),
            start=1
        ):

            print(
                f"\nSIGNER GROUP {new_number}"
            )

            print(
                "-" * 50
            )

            print(
                f"Images: {len(items)}"
            )

            output.write(
                f"\nSIGNER GROUP {new_number}\n"
            )

            output.write(
                "-" * 50 + "\n"
            )

            output.write(
                f"Images: {len(items)}\n\n"
            )

            # Show first 15 images.
            for sign, image_path in items[:15]:

                filename = os.path.basename(
                    image_path
                )

                line = (
                    f"  {sign:<8} "
                    f"{filename}"
                )

                print(line)

                output.write(
                    line + "\n"
                )

            if len(items) > 15:

                remaining = (
                    len(items) - 15
                )

                print(
                    f"  ... "
                    f"{remaining} more"
                )

                output.write(
                    f"  ... "
                    f"{remaining} more\n"
                )

    print("\n" + "=" * 70)

    print(
        f"✅ Results saved to: "
        f"{os.path.abspath(OUTPUT_FILE)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SIGNER GROUPING EXPERIMENT")
    print("=" * 70)

    print("\nDataset signs:")

    for sign in TEST_SIGNS:
        print(f"  • {sign}")

    image_paths = get_images()

    print(
        f"\nTotal images found: "
        f"{len(image_paths)}"
    )

    if not image_paths:

        print(
            "\n❌ No images found."
        )

        return

    model, transform = create_model()

    features, valid_images = (
        extract_all_features(
            model,
            transform,
            image_paths
        )
    )

    print(
        f"\nValid images: "
        f"{len(valid_images)}"
    )

    if len(valid_images) < NUMBER_OF_SIGNER_GROUPS:

        print(
            "\n❌ Not enough images "
            "for clustering."
        )

        return

    labels = create_groups(
        features
    )

    display_groups(
        labels,
        valid_images
    )

    print("\n✅ SIGNER GROUPING COMPLETE")


if __name__ == "__main__":
    main()