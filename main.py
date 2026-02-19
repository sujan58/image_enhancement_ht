import cv2
import os
import numpy as np
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# def apply_msrcr(img, sigmas=[15, 80, 250]):
#     img_float = np.float64(img) + 1.0 
#     retinex = sum(np.log10(img_float) - np.log10(cv2.GaussianBlur(img_float, (0, 0), s)) for s in sigmas) / len(sigmas)
#     return np.uint8(cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX))
# def apply_fusion(img):
#     # Convert to LAB and cast to float32 to prevent numpy casting errors
#     result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
#     a_avg, b_avg = np.average(result[:, :, 1]), np.average(result[:, :, 2])
    
#     # Use standard assignment instead of in-place subtraction (-=)
#     result[:, :, 1] = result[:, :, 1] - ((a_avg - 128) * (result[:, :, 0] / 255.0) * 1.1)
#     result[:, :, 2] = result[:, :, 2] - ((b_avg - 128) * (result[:, :, 0] / 255.0) * 1.1)
    
#     # Clip the values to valid ranges and convert back to uint8
#     result = np.clip(result, 0, 255).astype(np.uint8)
#     img1 = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
#     img2 = apply_clahe(img1)
    
#     # Calculate Weight Maps
#     w1 = np.abs(cv2.Laplacian(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.CV_64F)) + 1e-5
#     w2 = np.abs(cv2.Laplacian(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), cv2.CV_64F)) + 1e-5
    
#     W1, W2 = w1 / (w1 + w2), w2 / (w1 + w2)
    
#     fused = np.zeros_like(img, dtype=np.float64)
#     for i in range(3): 
#         fused[:, :, i] = img1[:, :, i] * W1 + img2[:, :, i] * W2
        
#     return np.uint8(np.clip(fused, 0, 255))
video_path = "input.mp4"
output_folder = "frames"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

# Get original FPS
fps = cap.get(cv2.CAP_PROP_FPS)
print("Original FPS:", fps)

# Save FPS to file
with open("fps.txt", "w") as f:
    f.write(str(fps))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(filename, apply_clahe(frame))
    
    frame_count += 1

cap.release()

print("Total frames extracted:", frame_count)
