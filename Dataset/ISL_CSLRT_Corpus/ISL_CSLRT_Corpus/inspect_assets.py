import os

BASE_PATH = r".\Frames_Word_Level"

folders = sorted(
    folder
    for folder in os.listdir(BASE_PATH)
    if os.path.isdir(os.path.join(BASE_PATH, folder))
)

print(f"Total sign assets: {len(folders)}\n")

for i, folder in enumerate(folders, start=1):
    print(f"{i:03}. {folder}")