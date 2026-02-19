import cv2
import os

image_folder = "frames"
output_video = "reconstructed.mp4"

# Read FPS from file
with open("fps.txt", "r") as f:
    fps = float(f.read())

print("Using FPS:", fps)

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])

if not images:
    print("No frames found!")
    exit()

first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for image in images:
    frame_path = os.path.join(image_folder, image)
    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    video.write(frame)

video.release()

print("Video reconstructed successfully!")
