from PIL import Image, ImageDraw, ImageFont
import os
import math

BASE = r"D:\Mini project HHAI\nexus\Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"
WORD_DIR = os.path.join(BASE, "Frames_Word_Level")

WORDS = ["YOU", "FREE", "TODAY", "HAPPY", "HOW"]

THUMB_W = 180
THUMB_H = 180
LABEL_H = 35
COLS = 6
PADDING = 10

try:
    FONT = ImageFont.truetype("arial.ttf", 14)
except:
    FONT = ImageFont.load_default()

all_images = []

for word in WORDS:
    folder = os.path.join(WORD_DIR, word)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Natural numerical sorting
    import re

    def number_key(filename):
        numbers = re.findall(r"\d+", filename)
        return int(numbers[-1]) if numbers else 0

    files.sort(key=number_key)

    for filename in files:
        all_images.append((word, filename, os.path.join(folder, filename)))

ROWS = math.ceil(len(all_images) / COLS)

WIDTH = COLS * (THUMB_W + PADDING) + PADDING
HEIGHT = ROWS * (THUMB_H + LABEL_H + PADDING) + PADDING

sheet = Image.new("RGB", (WIDTH, HEIGHT), "white")
draw = ImageDraw.Draw(sheet)

for index, (word, filename, path) in enumerate(all_images):

    row = index // COLS
    col = index % COLS

    x = PADDING + col * (THUMB_W + PADDING)
    y = PADDING + row * (THUMB_H + LABEL_H + PADDING)

    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((THUMB_W, THUMB_H))

        px = x + (THUMB_W - img.width) // 2
        py = y + (THUMB_H - img.height) // 2

        sheet.paste(img, (px, py))

    except Exception as e:
        print("Could not read:", path, e)

    label = f"{word} | {filename}"

    draw.text(
        (x, y + THUMB_H + 5),
        label,
        fill="black",
        font=FONT
    )

output = "signer_comparison_sheet.jpg"
sheet.save(output, quality=95)

print()
print("Created:", output)
print("Total images:", len(all_images))