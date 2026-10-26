import cv2
import numpy as np
import os
import subprocess
import imageio_ffmpeg
from ultralytics import YOLO
from collections import defaultdict, Counter

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

video_in = r'U.S Army Combat Team Night Operations _ Night Vision Footage.mp4'
model_path = r'models/thirdeye_clahe_best.pt'
out_dir = r'static/demo'
os.makedirs(out_dir, exist_ok=True)

raw_tmp = os.path.join(out_dir, 'raw_tmp.mp4')
hud_tmp = os.path.join(out_dir, 'hud_tmp.mp4')
raw_h264 = os.path.join(out_dir, 'raw_demo.mp4')
hud_h264 = os.path.join(out_dir, 'thirdeye_demo.mp4')

model = YOLO(model_path)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess(frame):
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(clahe.apply(g), cv2.COLOR_GRAY2BGR)

class TrackMemory:
    def __init__(self):
        self.votes = defaultdict(Counter)
        self.hist = defaultdict(list)
        self.seen = Counter()
        
    def update(self, tid, cls, conf, center, frame_idx):
        self.votes[tid][cls] += conf
        self.hist[tid].append((frame_idx, center))
        self.seen[tid] += 1
        
    def label(self, tid):
        if not self.votes[tid]: return 0
        return self.votes[tid].most_common(1)[0][0]
        
    def velocity(self, tid, fps):
        h = self.hist[tid][-10:]
        if len(h) < 2: return 0.0, (0.0, 0.0)
        (f0, (x0, y0)), (f1, (x1, y1)) = h[0], h[-1]
        dt = max(f1 - f0, 1) / fps
        vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
        return float(np.hypot(vx, vy)), (vx, vy)
        
    def counts(self):
        c = Counter()
        for tid, n in self.seen.items():
            if n >= 3: c[self.label(tid)] += 1
        return c

CLASS_NAMES = ['Soldier', 'Tank', 'APC', 'Military_Truck', 'Mobile_Air_Defense']
CLASS_COLORS = {
    0: (0, 255, 128),
    1: (0, 64, 255),
    2: (0, 180, 255),
    3: (255, 200, 0),
    4: (255, 0, 255)
}

def draw_hud(vis, mem, active_count, frame_idx, fps):
    h, w = vis.shape[:2]
    b_len, b_col = 26, (0, 220, 150)
    # Tactical corners
    cv2.line(vis, (15, 15), (15 + b_len, 15), b_col, 1)
    cv2.line(vis, (15, 15), (15, 15 + b_len), b_col, 1)
    cv2.line(vis, (w - 15, 15), (w - 15 - b_len, 15), b_col, 1)
    cv2.line(vis, (w - 15, 15), (w - 15, 15 + b_len), b_col, 1)
    cv2.line(vis, (15, h - 15), (15 + b_len, h - 15), b_col, 1)
    cv2.line(vis, (15, h - 15), (15, h - 15 - b_len), b_col, 1)
    cv2.line(vis, (w - 15, h - 15), (w - 15 - b_len, h - 15), b_col, 1)
    cv2.line(vis, (w - 15, h - 15), (w - 15, h - 15 - b_len), b_col, 1)

    # Reticle
    cx, cy = w // 2, h // 2
    cv2.line(vis, (cx - 15, cy), (cx - 5, cy), (0, 255, 200), 1)
    cv2.line(vis, (cx + 5, cy), (cx + 15, cy), (0, 255, 200), 1)
    cv2.line(vis, (cx, cy - 15), (cx, cy - 5), (0, 255, 200), 1)
    cv2.line(vis, (cx, cy + 5), (cx, cy + 15), (0, 255, 200), 1)

    # HUD Box
    box_w, box_h = 320, 155
    ov = vis.copy()
    cv2.rectangle(ov, (15, 15), (15 + box_w, 15 + box_h), (8, 12, 16), -1)
    vis = cv2.addWeighted(ov, 0.65, vis, 0.35, 0)
    cv2.rectangle(vis, (15, 15), (15 + box_w, 15 + box_h), (0, 200, 130), 1)

    cv2.rectangle(vis, (15, 15), (15 + box_w, 42), (0, 140, 90), -1)
    cv2.putText(vis, "THIRDEYE AI // TACTICAL HUD", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    time_sec = frame_idx / fps
    counts = mem.counts()
    cv2.putText(vis, f"FEED: FLIR / CLAHE IR | T+{time_sec:05.1f}s", (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 200), 1, cv2.LINE_AA)
    cv2.putText(vis, f"ACTIVE DETECTIONS: {active_count}", (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 180), 1, cv2.LINE_AA)

    c_y = 104
    for i, cname in enumerate(CLASS_NAMES):
        c_val = counts.get(i, 0)
        col = CLASS_COLORS.get(i, (200, 200, 200))
        cv2.putText(vis, f"{cname[:8]}: {c_val}", (24 + (i % 2) * 145, c_y + (i // 2) * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    cv2.putText(vis, f"AZM 142.4 DEG | ELEV -03.2 DEG | SENSOR: FLIR LWIR | LAT 34.052 N | LON 43.125 E",
                (w//2 - 270, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 150), 1, cv2.LINE_AA)
    return vis

cap = cv2.VideoCapture(video_in)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Skip the initial 5.5s intro / title card
start_frame = int(5.5 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
clip_frames = 600 # 20 seconds of actual combat night ops

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_raw = cv2.VideoWriter(raw_tmp, fourcc, fps, (W, H))
out_hud = cv2.VideoWriter(hud_tmp, fourcc, fps, (W, H))

mem = TrackMemory()
poster_saved = False

print(f"Rendering demo starting from frame {start_frame} (T+5.5s)...")

for fi in range(clip_frames):
    ret, frame = cap.read()
    if not ret: break
    
    out_raw.write(frame)
    prep = preprocess(frame)
    
    results = model.track(prep, tracker='bytetrack.yaml', persist=True, conf=0.10, imgsz=800, verbose=False)[0]
    vis = prep.copy()
    active = 0
    
    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.int().cpu().tolist()
        classes = results.boxes.cls.int().cpu().tolist()
        confs = results.boxes.conf.cpu().tolist()
        
        for box, tid, cls_idx, conf_val in zip(boxes, track_ids, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            mem.update(tid, cls_idx, conf_val, (cx, cy), start_frame + fi)
            active += 1
            
            lab = mem.label(tid)
            col = CLASS_COLORS.get(lab, (0, 255, 0))
            spd, (vx, vy) = mem.velocity(tid, fps)
            
            d = max(6, min(14, (x2 - x1) // 3, (y2 - y1) // 3))
            cv2.line(vis, (x1, y1), (x1 + d, y1), col, 2)
            cv2.line(vis, (x1, y1), (x1, y1 + d), col, 2)
            cv2.line(vis, (x2, y1), (x2 - d, y1), col, 2)
            cv2.line(vis, (x2, y1), (x2, y1 + d), col, 2)
            cv2.line(vis, (x1, y2), (x1 + d, y2), col, 2)
            cv2.line(vis, (x1, y2), (x1, y2 - d), col, 2)
            cv2.line(vis, (x2, y2), (x2 - d, y2), col, 2)
            cv2.line(vis, (x2, y2), (x2, y2 - d), col, 2)
            
            cv2.circle(vis, (cx, cy), 2, col, -1)
            
            tag = f"#{tid} {CLASS_NAMES[lab]} {spd:.0f}px/s"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (8, 12, 16), -1)
            cv2.putText(vis, tag, (x1 + 3, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
            
            if spd > 5:
                n = max(np.hypot(vx, vy), 1e-6)
                cv2.arrowedLine(vis, (cx, cy), (int(cx + vx / n * 35), int(cy + vy / n * 35)), col, 2, tipLength=0.3)
                
    hud_frame = draw_hud(vis, mem, active, fi, fps)
    out_hud.write(hud_frame)
    
    if fi == 120 and not poster_saved:
        cv2.imwrite(os.path.join(out_dir, 'thirdeye_demo_poster.jpg'), frame)
        poster_saved = True

cap.release()
out_raw.release()
out_hud.release()

print("Transcoding with ffmpeg...")
subprocess.run([
    ffmpeg_exe, '-y', '-i', raw_tmp,
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23',
    raw_h264
], check=True)

subprocess.run([
    ffmpeg_exe, '-y', '-i', hud_tmp,
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23',
    hud_h264
], check=True)

if os.path.exists(raw_tmp): os.remove(raw_tmp)
if os.path.exists(hud_tmp): os.remove(hud_tmp)
print("Updated demo clips cleanly starting past intro card.")
