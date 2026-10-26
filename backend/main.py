import os
import time
import uuid
import json
import base64
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import torch

from backend.pipeline import ThirdEyeEngine, CLASS_NAMES

app = FastAPI(title="ThirdEye Defense AI Portal", version="2.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DEMO_DIR = STATIC_DIR / "demo"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DEMO_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Initialize ThirdEye Engine once
engine = ThirdEyeEngine(
    thermal_weights=str(BASE_DIR / "models" / "thirdeye_clahe_best.pt"),
    drone_weights=str(BASE_DIR / "models" / "military_convoy_yolo11l.pt")
)

# In-memory job state store
jobs = {}

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>ThirdEye Portal initializing...</h1>")

@app.get("/api/status")
async def get_status():
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU (Host)"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if gpu_available else 0
    
    default_video = BASE_DIR / "Ukraine drone video shows attack on Russian tanks.mp4"
    video_exists = default_video.exists()
    video_size_mb = round(default_video.stat().st_size / (1024**2), 2) if video_exists else 0

    return {
        "status": "online",
        "system": "ThirdEye Tactical Autonomous AI v2.4",
        "hardware": {
            "gpu_acceleration": gpu_available,
            "device": device_name,
            "vram_gb": vram_gb,
            "framework": "PyTorch / Ultralytics YOLO11 + ByteTrack"
        },
        "default_video": {
            "name": "Ukraine drone video shows attack on Russian tanks.mp4",
            "exists": video_exists,
            "size_mb": video_size_mb,
            "resolution": "1280x720 (HD)",
            "framerate": 30.0
        },
        "classes": CLASS_NAMES,
        "models": [
            {"id": "auto", "name": "Auto-Route (Sensor Adaptive: Thermal vs Drone)", "active": True},
            {"id": "drone", "name": "ThirdEye YOLO11L Drone Combat & Convoy (Aerial EO/IR)", "active": False},
            {"id": "thermal", "name": "ThirdEye YOLO11s Night Combat & FLIR (Thermal IR)", "active": False}
        ]
    }

@app.get("/api/ablation")
async def get_ablation():
    csv_path = BASE_DIR / "ablation_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        data = df.to_dict(orient="records")
    else:
        data = [
            {"experiment": "YOLO11 raw", "precision": 0.663, "recall": 0.309, "mAP50": 0.338, "mAP50_95": 0.128},
            {"experiment": "YOLO11 + CLAHE", "precision": 0.291, "recall": 0.397, "mAP50": 0.303, "mAP50_95": 0.125},
            {"experiment": "YOLO11 + CLAHE + TTA", "precision": 0.368, "recall": 0.402, "mAP50": 0.325, "mAP50_95": 0.135}
        ]
    return {
        "results": data,
        "summary": "CLAHE preprocessing increases dark target recall from 30.9% to 40.2% (+30.1% relative increase), crucial for perimeter defense and night combat threat identification."
    }

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Unsupported video format. Please upload MP4, AVI, MOV, or WEBM.")
    
    file_id = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path = UPLOAD_DIR / file_id
    
    with open(save_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
        
    return {
        "status": "success",
        "file_id": file_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "url": f"/uploads/{file_id}"
    }

def run_job_thread(job_id: str, video_path: str, out_path: str, model_choice: str, conf: float, use_clahe: bool, use_tracking: bool,
                   use_sahi: bool, use_tta: bool, apply_nvg: bool, max_frames: Optional[int], start_sec: float):
    job = jobs[job_id]
    job["status"] = "processing"
    
    def on_progress(p_data):
        job["progress"] = p_data["progress"]
        job["frame"] = p_data["frame"]
        job["total_frames"] = p_data["total_frames"]
        job["fps"] = p_data["fps"]
        job["active_count"] = p_data["active_count"]
        job["unique_targets"] = p_data["unique_targets"]
        job["events"] = p_data["events"]
        # Convert thumbnail to base64 for fast frontend display
        if p_data.get("thumbnail_bytes"):
            job["latest_frame_b64"] = base64.b64encode(p_data["thumbnail_bytes"]).decode("utf-8")

    try:
        results = engine.process_video_file(
            video_path=video_path,
            out_video_path=out_path,
            model_choice=model_choice,
            conf=conf,
            use_clahe=use_clahe,
            use_tracking=use_tracking,
            use_sahi=use_sahi,
            use_tta=use_tta,
            apply_nvg=apply_nvg,
            max_frames=max_frames,
            start_sec=start_sec,
            progress_callback=on_progress
        )
        job["status"] = "completed"
        job["progress"] = 1.0
        job["result"] = results
        job["video_url"] = f"/outputs/{os.path.basename(out_path)}"
    except Exception as e:
        print(f"[Job {job_id} Error]: {e}")
        job["status"] = "failed"
        job["error"] = str(e)

@app.post("/api/run")
async def run_model(
    video_source: str = Form("default"),
    uploaded_file_id: Optional[str] = Form(None),
    model_choice: str = Form("auto"),
    conf: float = Form(0.15),
    use_clahe: bool = Form(True),
    use_tracking: bool = Form(True),
    use_sahi: bool = Form(False),
    use_tta: bool = Form(False),
    apply_nvg: bool = Form(False),
    duration_sec: int = Form(15), # 15s default for fast interactive demo
    start_sec: float = Form(0.0)
):
    if video_source == "default":
        video_path = str(BASE_DIR / "Ukraine drone video shows attack on Russian tanks.mp4")
    elif uploaded_file_id:
        video_path = str(UPLOAD_DIR / uploaded_file_id)
    else:
        raise HTTPException(status_code=400, detail="Invalid video source.")

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Selected video file not found.")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    out_filename = f"thirdeye_out_{job_id}.mp4"
    out_path = str(OUTPUT_DIR / out_filename)

    max_frames = int(duration_sec * 30) if duration_sec > 0 else None

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "frame": 0,
        "total_frames": max_frames or 0,
        "fps": 0.0,
        "active_count": 0,
        "unique_targets": {},
        "events": [],
        "latest_frame_b64": None,
        "created_at": time.time()
    }

    t = threading.Thread(
        target=run_job_thread,
        args=(job_id, video_path, out_path, model_choice, conf, use_clahe, use_tracking, use_sahi, use_tta, apply_nvg, max_frames, start_sec),
        daemon=True
    )
    t.start()

    return {"job_id": job_id, "status": "queued", "video_source": video_source}

@app.get("/api/progress/{job_id}")
async def get_progress(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
