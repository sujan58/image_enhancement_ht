import os
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
FRAMES_FOLDER = os.path.join(BASE_DIR, "temp_frames")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

app = FastAPI(title="AquaVision — Underwater Video Enhancement")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------- CLAHE Enhancement (from main.py) ----------

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


# ---------- Video Processing (main.py + reconstruct.py logic) ----------

def process_video(input_path: str, output_path: str) -> int:
    """
    Step 1 (main.py):   Extract frames, apply CLAHE to each, save as JPGs.
    Step 2 (reconstruct.py): Read enhanced frames back and write into a new video.
    """
    job_id = os.path.splitext(os.path.basename(output_path))[0]
    frames_dir = os.path.join(FRAMES_FOLDER, job_id)
    os.makedirs(frames_dir, exist_ok=True)

    # ---- STEP 1: Extract & enhance frames (main.py logic) ----
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        enhanced = apply_clahe(frame)
        filename = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(filename, enhanced)
        frame_count += 1

    cap.release()

    if frame_count == 0:
        raise RuntimeError("No frames could be read from video")

    # ---- STEP 2: Reconstruct video from frames (reconstruct.py logic) ----
    images = sorted([img for img in os.listdir(frames_dir) if img.endswith(".jpg")])
    first_frame = cv2.imread(os.path.join(frames_dir, images[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_name in images:
        frame = cv2.imread(os.path.join(frames_dir, image_name))
        if frame is not None:
            writer.write(frame)

    writer.release()

    # Clean up temp frames
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)

    # Re-encode to H.264 for browser playback (if ffmpeg is available)
    h264_path = output_path.replace(".mp4", "_h264.mp4")
    ret_code = os.system(
        f'ffmpeg -y -i "{output_path}" -c:v libx264 -preset fast -crf 23 '
        f'-c:a aac -movflags +faststart "{h264_path}"'
    )
    if ret_code == 0 and os.path.exists(h264_path):
        os.replace(h264_path, output_path)

    return frame_count


# ---------- API Routes ----------

@app.get("/api/info")
async def info():
    return {"algorithm": "CLAHE", "description": "Contrast Limited Adaptive Histogram Equalization"}


@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...)):
    if not video.filename or video.filename == "":
        raise HTTPException(status_code=400, detail="No file selected")

    if not allowed_file(video.filename):
        raise HTTPException(status_code=400, detail="File type not allowed. Use mp4, avi, mov, mkv, or webm.")

    # Save uploaded file
    job_id = str(uuid.uuid4())[:8]
    ext = video.filename.rsplit(".", 1)[1].lower()
    input_filename = f"{job_id}_input.{ext}"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)

    with open(input_path, "wb") as f:
        content = await video.read()
        f.write(content)

    # Process video
    output_filename = f"{job_id}_enhanced.mp4"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)

    try:
        frame_count = process_video(input_path, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    return {
        "success": True,
        "message": f"Enhanced {frame_count} frames with CLAHE",
        "video_url": f"/api/video/{output_filename}",
        "filename": output_filename,
    }


@app.get("/api/video/{filename}")
async def serve_video(filename: str):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


# Serve frontend (must be last so /api routes take priority)
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")


if __name__ == "__main__":
    print("🚀 AquaVision Server starting...")
    print("   Open http://localhost:5000 in your browser")
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
