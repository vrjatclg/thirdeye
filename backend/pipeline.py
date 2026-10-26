import os
import cv2
import numpy as np
import time
import subprocess
import imageio_ffmpeg
from collections import defaultdict, Counter
from ultralytics import YOLO

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# Default Thermal Classes
THERMAL_CLASSES = ['Soldier', 'Tank', 'APC', 'Military_Truck', 'Mobile_Air_Defense']
THERMAL_COLORS = {
    0: (0, 255, 128),   # Soldier: Green
    1: (0, 64, 255),    # Tank: Crimson
    2: (0, 180, 255),   # APC: Amber Gold
    3: (255, 200, 0),   # Military Truck: Cyan
    4: (255, 0, 255)    # Mobile Air Defense: Magenta
}
ARMOR_CLASSES_THERMAL = {1, 2, 4}
CLASS_NAMES = THERMAL_CLASSES

# Drone MilitaryConvoy Colors
DRONE_COLORS = {
    'tank': (0, 64, 255),             # Crimson
    'apc': (0, 180, 255),             # Amber Gold
    'afv': (0, 140, 255),             # Bright Amber
    'artillery': (255, 0, 255),       # Magenta
    'military-vehicle': (255, 200, 0),# Cyan
    'truck': (255, 200, 0),           # Cyan
    'person': (0, 255, 128),          # Green
    'pedestrian': (0, 255, 128)       # Green
}
MILITARY_DRONE_CLASSES = {'tank', 'apc', 'afv', 'artillery', 'military-vehicle', 'truck', 'person', 'pedestrian'}

class TrackMemory:
    """Per-track class voting, trajectory history, temporal smoothing, and speed calculation."""
    def __init__(self, min_track_frames=2, speed_window=10):
        self.votes = defaultdict(Counter)
        self.hist = defaultdict(list)
        self.seen = Counter()
        self.smooth_boxes = {}
        self.min_track_frames = min_track_frames
        self.speed_window = speed_window

    def update(self, tid, cls_key, conf, box, frame_idx):
        self.votes[tid][cls_key] += conf
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        self.hist[tid].append((frame_idx, (cx, cy)))
        self.seen[tid] += 1
        
        # Temporal bounding box smoothing (EMA)
        if tid in self.smooth_boxes:
            prev = self.smooth_boxes[tid]
            smooth = [
                int(0.65 * prev[0] + 0.35 * box[0]),
                int(0.65 * prev[1] + 0.35 * box[1]),
                int(0.65 * prev[2] + 0.35 * box[2]),
                int(0.65 * prev[3] + 0.35 * box[3])
            ]
            self.smooth_boxes[tid] = smooth
        else:
            self.smooth_boxes[tid] = list(box)

    def get_box(self, tid):
        return self.smooth_boxes.get(tid)

    def label(self, tid):
        if not self.votes[tid]:
            return None
        return self.votes[tid].most_common(1)[0][0]

    def velocity(self, tid, fps=30.0):
        h = self.hist[tid][-self.speed_window:]
        if len(h) < 2:
            return 0.0, (0.0, 0.0)
        (f0, (x0, y0)), (f1, (x1, y1)) = h[0], h[-1]
        dt = max(f1 - f0, 1) / max(fps, 1)
        vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
        return float(np.hypot(vx, vy)), (vx, vy)

    def counts(self):
        c = Counter()
        for tid, n in self.seen.items():
            if n >= self.min_track_frames:
                lab = self.label(tid)
                if lab is not None:
                    c[lab] += 1
        return c


class ThirdEyeEngine:
    def __init__(self,
                 thermal_weights=r'models/thirdeye_clahe_best.pt',
                 drone_weights=r'models/military_convoy_yolo11l.pt'):
        self.thermal_weights = thermal_weights
        self.drone_weights = drone_weights
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.tactical_tracker_cfg = os.path.join(os.path.dirname(__file__), 'bytetrack_tactical.yaml')
        if not os.path.exists(self.tactical_tracker_cfg):
            self.tactical_tracker_cfg = 'bytetrack.yaml'

        self.models = {}
        if os.path.exists(self.drone_weights):
            print(f"[ThirdEyeEngine] Initializing YOLO11L Drone-Armor weights: {self.drone_weights}...")
            self.models['drone'] = YOLO(self.drone_weights)
        if os.path.exists(self.thermal_weights):
            print(f"[ThirdEyeEngine] Initializing YOLO11s Thermal-FLIR weights: {self.thermal_weights}...")
            self.models['thermal'] = YOLO(self.thermal_weights)
        print("[ThirdEyeEngine] Ready.")

    def preprocess_thermal(self, frame):
        """Thermal CLAHE Normalization for Infrared/Grayscale imagery"""
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(self.clahe.apply(g), cv2.COLOR_GRAY2BGR)

    def is_thermal_feed(self, frame):
        """Check if image is grayscale/thermal IR vs visible RGB"""
        b, g, r = cv2.split(frame)
        color_diff = float(np.mean(np.abs(r.astype(float) - g.astype(float))) + 
                           np.mean(np.abs(g.astype(float) - b.astype(float))))
        return color_diff < 3.5

    def to_nvg(self, frame):
        """Converts RGB/daylight frame into Gen-3 Phosphor Green Night Vision feed with CLAHE."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        norm = enhanced.astype(np.float32) / 255.0
        nvg = np.zeros_like(frame)
        nvg[:, :, 0] = np.clip(norm * 35.0 + (norm**3) * 160.0, 0, 255).astype(np.uint8)
        nvg[:, :, 1] = np.clip(norm * 245.0 + (norm**2) * 10.0, 0, 255).astype(np.uint8)
        nvg[:, :, 2] = np.clip(norm * 25.0 + (norm**3) * 120.0, 0, 255).astype(np.uint8)
        noise = np.random.normal(0, 2.5, nvg.shape).astype(np.int16)
        return np.clip(nvg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def draw_tactical_hud(self, vis, mem, active_count, frame_idx, fps, sensor_type, model_type):
        h, w = vis.shape[:2]
        
        # Tactical corner brackets
        b_len = 24
        b_col = (0, 220, 150)
        cv2.line(vis, (12, 12), (12 + b_len, 12), b_col, 1)
        cv2.line(vis, (12, 12), (12, 12 + b_len), b_col, 1)
        cv2.line(vis, (w - 12, 12), (w - 12 - b_len, 12), b_col, 1)
        cv2.line(vis, (w - 12, 12), (w - 12, 12 + b_len), b_col, 1)
        cv2.line(vis, (12, h - 12), (12 + b_len, h - 12), b_col, 1)
        cv2.line(vis, (12, h - 12), (12, h - 12 - b_len), b_col, 1)
        cv2.line(vis, (w - 12, h - 12), (w - 12 - b_len, h - 12), b_col, 1)
        cv2.line(vis, (w - 12, h - 12), (w - 12, h - 12 - b_len), b_col, 1)

        # Center Reticle
        cx, cy = w // 2, h // 2
        cv2.line(vis, (cx - 12, cy), (cx - 4, cy), (0, 255, 200), 1)
        cv2.line(vis, (cx + 4, cy), (cx + 12, cy), (0, 255, 200), 1)
        cv2.line(vis, (cx, cy - 12), (cx, cy - 4), (0, 255, 200), 1)
        cv2.line(vis, (cx, cy + 4), (cx, cy + 12), (0, 255, 200), 1)

        # Tactical Status Box
        box_w, box_h = 330, 156
        ov = vis.copy()
        cv2.rectangle(ov, (12, 12), (12 + box_w, 12 + box_h), (8, 12, 16), -1)
        vis = cv2.addWeighted(ov, 0.70, vis, 0.30, 0)
        cv2.rectangle(vis, (12, 12), (12 + box_w, 12 + box_h), (0, 200, 130), 1)

        # Header Bar
        cv2.rectangle(vis, (12, 12), (12 + box_w, 38), (0, 140, 90), -1)
        cv2.putText(vis, "THIRDEYE AI // TACTICAL HUD", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        # Status text
        t_sec = frame_idx / max(fps, 1)
        cv2.putText(vis, f"FEED: {sensor_type} | T+{t_sec:05.1f}s", (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 200), 1, cv2.LINE_AA)
        cv2.putText(vis, f"ACTIVE ENGAGEMENTS: {active_count}", (20, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1, cv2.LINE_AA)

        counts = mem.counts()
        if model_type == 'drone':
            tanks = counts.get('tank', 0)
            apcs = counts.get('apc', 0) + counts.get('afv', 0)
            artillery = counts.get('artillery', 0)
            logistics = counts.get('truck', 0) + counts.get('military-vehicle', 0)

            cv2.putText(vis, f"TANKS: {tanks}", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 64, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"APC/AFV: {apcs}", (180, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"ARTILLERY: {artillery}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"LOGISTICS: {logistics}", (180, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, "MODEL: YOLO11-AERIAL MILITARY", (20, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 150), 1, cv2.LINE_AA)
        else:
            afv_total = sum(counts.get(k, 0) for k in ARMOR_CLASSES_THERMAL)
            trucks = counts.get(3, 0)
            infantry = counts.get(0, 0)
            cv2.putText(vis, f"AFV / ARMOR: {afv_total}", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"TRUCKS: {trucks}", (180, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, f"INFANTRY: {infantry}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 128), 1, cv2.LINE_AA)
            cv2.putText(vis, "MODEL: YOLO11-THERMAL FLIR", (20, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 150), 1, cv2.LINE_AA)

        tot_targets = sum(counts.values())
        cv2.putText(vis, f"SYSTEM: 30 FPS | GRID 36U YA 812 | SENSOR: {sensor_type} | TARGETS CONFIRMED: {tot_targets}",
                    (max(20, w//2 - 290), h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 150), 1, cv2.LINE_AA)

        return vis

    def process_video_file(self, video_path, out_video_path, model_choice="auto", conf=0.15, use_clahe=True,
                           use_tracking=True, use_sahi=False, use_tta=False, apply_nvg=False, max_frames=None, start_sec=0, progress_callback=None):
        """Process video with adaptive sensor routing, SAHI small-target recovery, TTA, and NVG night vision simulation."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(start_sec * fps)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if max_frames:
            num_frames_to_process = min(max_frames, total_video_frames - start_frame)
        else:
            num_frames_to_process = total_video_frames - start_frame

        os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
        tmp_output = out_video_path + ".tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_output, fourcc, fps, (W, H))

        mem = TrackMemory(min_track_frames=2)
        events = []
        target_tracks = {}

        t_start = time.time()
        processed_count = 0

        # Detect feed type from initial frame
        ret_init, frame_init = cap.read()
        if ret_init:
            is_thermal = self.is_thermal_feed(frame_init)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            is_thermal = False

        # Select model based on sensor feed and user choice
        if model_choice == "drone":
            active_model_key = 'drone' if 'drone' in self.models else 'thermal'
        elif model_choice == "thermal":
            active_model_key = 'thermal' if 'thermal' in self.models else 'drone'
        else:
            # Auto-route
            active_model_key = 'thermal' if (is_thermal and 'thermal' in self.models) else ('drone' if 'drone' in self.models else 'thermal')

        active_yolo = self.models.get(active_model_key)
        if apply_nvg:
            sensor_label = "GEN-3 NVG (IIT) // 0-LUX"
        else:
            sensor_label = "FLIR THERMAL IR" if is_thermal else "OPTICAL DRONE RECON"
        inference_imgsz = 1280 if max(W, H) >= 1200 else 800

        print(f"[ThirdEyeEngine] Running inference on {video_path} with model: {active_model_key.upper()} (imgsz={inference_imgsz}, conf={conf}, sahi={use_sahi}, tta={use_tta}, nvg={apply_nvg})")

        try:
            for fi in range(num_frames_to_process):
                ret, frame = cap.read()
                if not ret:
                    break

                processed_count += 1
                curr_frame_idx = start_frame + fi
                curr_sec = curr_frame_idx / fps

                if use_clahe and is_thermal:
                    prep = self.preprocess_thermal(frame)
                else:
                    prep = frame

                if apply_nvg:
                    vis = self.to_nvg(frame)
                else:
                    vis = frame.copy()
                active = 0

                if use_sahi and active_model_key == 'drone':
                    if not hasattr(self, '_sahi_model'):
                        from sahi import AutoDetectionModel
                        self._sahi_model = AutoDetectionModel.from_pretrained(
                            model_type='yolov8',
                            model_path=self.drone_weights,
                            confidence_threshold=conf,
                            device='cuda:0'
                        )
                    from sahi.predict import get_sliced_prediction
                    res_sahi = get_sliced_prediction(
                        prep,
                        self._sahi_model,
                        slice_height=640,
                        slice_width=640,
                        overlap_height_ratio=0.2,
                        overlap_width_ratio=0.2,
                        verbose=0
                    )
                    for p_idx, pred in enumerate(res_sahi.object_prediction_list):
                        raw_name = pred.category.name.lower()
                        if raw_name not in MILITARY_DRONE_CLASSES:
                            continue
                        b = pred.bbox.to_xyxy()
                        int_box = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                        conf_val = float(pred.score.value)
                        tid = p_idx + 1
                        mem.update(tid, raw_name, conf_val, int_box, curr_frame_idx)
                        active += 1

                        display_label = raw_name.upper()
                        col = DRONE_COLORS.get(raw_name, (0, 220, 150))
                        cv2.rectangle(vis, (int_box[0], int_box[1]), (int_box[2], int_box[3]), col, 2)
                        tag = f"#{tid} {display_label} {conf_val*100:.0f}%"
                        cv2.putText(vis, tag, (int_box[0], max(12, int_box[1] - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
                elif use_tracking and active_yolo:
                    results = active_yolo.track(prep, tracker=self.tactical_tracker_cfg, persist=True,
                                                augment=use_tta, conf=conf, imgsz=inference_imgsz, verbose=False)[0]
                    if results.boxes.id is not None:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        tids = results.boxes.id.int().cpu().tolist()
                        classes = results.boxes.cls.int().cpu().tolist()
                        confs = results.boxes.conf.cpu().tolist()

                        for raw_box, tid, c_idx, conf_val in zip(boxes, tids, classes, confs):
                            if active_model_key == 'drone':
                                raw_name = active_yolo.names[c_idx].lower()
                                if raw_name not in MILITARY_DRONE_CLASSES:
                                    continue
                                cls_key = raw_name
                                display_label = raw_name.upper()
                                col = DRONE_COLORS.get(raw_name, (0, 220, 150))
                            else:
                                cls_key = c_idx
                                raw_name = THERMAL_CLASSES[c_idx] if c_idx < len(THERMAL_CLASSES) else f"Target_{c_idx}"
                                display_label = f"AFV [{raw_name}]" if c_idx in ARMOR_CLASSES_THERMAL else raw_name
                                col = (0, 140, 255) if c_idx in ARMOR_CLASSES_THERMAL else THERMAL_COLORS.get(c_idx, (0, 255, 0))

                            int_box = list(map(int, raw_box))
                            bw = int_box[2] - int_box[0]
                            bh = int_box[3] - int_box[1]
                            aspect = bw / max(bh, 1)

                            # Strict civilian infrastructure and house roof rejection:
                            # Real tanks at oblique angles have aspect ratio >= 1.20 (elongated hulls).
                            # Houses and roofs are square or vertical (aspect < 1.15) or oversized (>9,000px).
                            if active_model_key == 'drone':
                                if bw * bh > 9000 or aspect < 1.20 or bw > 140 or bh > 85:
                                    continue
                            else:
                                if bw * bh > 14000 or bw > 180 or bh > 140:
                                    continue

                            mem.update(tid, cls_key, conf_val, int_box, curr_frame_idx)
                            
                            if mem.seen[tid] < 2:
                                continue

                            active += 1
                            smooth_box = mem.get_box(tid)
                            x1, y1, x2, y2 = smooth_box
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            spd, (vx, vy) = mem.velocity(tid, fps)

                            target_tracks[tid] = {
                                "id": tid,
                                "class": display_label,
                                "confidence": round(conf_val, 2),
                                "speed_px_s": round(spd, 1),
                                "last_seen_sec": round(curr_sec, 1)
                            }

                            if mem.seen[tid] == 3:
                                events.append({
                                    "time_sec": round(curr_sec, 1),
                                    "track_id": tid,
                                    "class": display_label,
                                    "confidence": round(conf_val, 2),
                                    "message": f"Target #{tid} confirmed as {display_label} ({round(conf_val*100)}%)"
                                })

                            # Corner brackets
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

                            tag = f"#{tid} {display_label} {spd:.0f}px/s"
                            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                            cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (8, 12, 16), -1)
                            cv2.putText(vis, tag, (x1 + 3, max(10, y1 - 3)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

                            if spd > 3:
                                n = max(np.hypot(vx, vy), 1e-6)
                                cv2.arrowedLine(vis, (cx, cy), (int(cx + vx / n * 35), int(cy + vy / n * 35)),
                                                col, 2, tipLength=0.3)

                hud_frame = self.draw_tactical_hud(vis, mem, active, curr_frame_idx, fps,
                                                   sensor_type=sensor_label, model_type=active_model_key)
                writer.write(hud_frame)

                if progress_callback and (fi % 8 == 0 or fi == num_frames_to_process - 1):
                    elapsed = max(time.time() - t_start, 0.001)
                    current_fps = processed_count / elapsed
                    progress = processed_count / max(num_frames_to_process, 1)

                    thumb = cv2.resize(hud_frame, (640, 360))
                    _, thumb_jpeg = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])

                    progress_callback({
                        "progress": round(progress, 3),
                        "frame": processed_count,
                        "total_frames": num_frames_to_process,
                        "fps": round(current_fps, 1),
                        "active_count": active,
                        "unique_targets": dict(mem.counts()),
                        "thumbnail_bytes": thumb_jpeg.tobytes(),
                        "events": events[-5:]
                    })

        finally:
            cap.release()
            writer.release()

        print(f"[ThirdEyeEngine] Transcoding {tmp_output} to H.264 {out_video_path}...")
        cmd = [
            FFMPEG_EXE, '-y', '-i', tmp_output,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23',
            out_video_path
        ]
        subprocess.run(cmd, check=True)

        if os.path.exists(tmp_output):
            os.remove(tmp_output)

        counts_summary = dict(mem.counts())
        print(f"[ThirdEyeEngine] Video processing finished. Summary: {counts_summary}")

        return {
            "status": "completed",
            "model_used": active_model_key,
            "frames_processed": processed_count,
            "duration_sec": round(processed_count / fps, 1),
            "output_video": out_video_path,
            "target_counts": counts_summary,
            "tracks": list(target_tracks.values())[-25:],
            "events": events
        }
