import cv2
import os
import numpy as np

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

video_path = "input.mp4"
output_folder = "frames"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 24

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with open("video_info.txt", "w") as f:
    f.write(f"{fps}\n{width}\n{height}")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed = apply_clahe(frame)

    filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
    cv2.imwrite(filename, processed)

    frame_count += 1

cap.release()

print("Total frames extracted:", frame_count)