# ── REAL-TIME PREDICTION + ALERT + REPORT (YOLOv8n-face) — JETSON NANO ──
# Tanpa emotion buffer — prediksi langsung per frame
import cv2
import time
import csv
import os
from datetime import datetime
from ultralytics import YOLO

# ===============================
# LOAD MODEL
# ===============================
emotion_model = YOLO(
    '/home/user/models/best1.pt'
)
face_model = YOLO(
    '/home/user/models/yolov8n-face.pt'
)

EMOTIONS = list(emotion_model.names.values())

COLORS = {
    'Angry'  : (0,   0,   255),
    'Drowsy' : (0,   128, 0  ),
    'Netral' : (0,   255, 0  ),
    'Sad'    : (128, 0,   128),
}

THRESHOLD = {
    "Angry"  : 0.60,
    "Drowsy" : 0.55,
    "Netral" : 0.60,
    "Sad"    : 0.30,
}

min_conf = THRESHOLD.get("Netral", 0.50)

# ===============================
# KAMERA — GStreamer pipeline
# untuk Jetson Nano CSI camera
# Ganti ke index 0 jika pakai USB cam
# ===============================
def open_camera():
    # CSI camera via GStreamer
    gst_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[INFO] CSI camera terbuka via GStreamer")
        return cap

    # Fallback: USB webcam
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("[INFO] USB camera terbuka (index 0)")
        return cap

    raise RuntimeError("[ERROR] Tidak ada kamera yang bisa dibuka.")

cap = open_camera()

# Flush frame awal
for _ in range(5):
    cap.read()

# ===============================
# STATE
# ===============================
prev_time     = time.time()

# ── REPORT STATE ──────────────────────────────────────────────────────
session_start     = datetime.now()
emotion_counts    = {e: 0   for e in EMOTIONS}
emotion_durations = {e: 0.0 for e in EMOTIONS}
conf_accum        = {e: []  for e in EMOTIONS}

last_stable       = None
last_stable_start = time.time()

report_interval   = 5.0
last_report_time  = time.time()
overlay_lines     = []

# ── METRIK PERFORMA ───────────────────────────────────────────────────
fps_history     = []
latency_history = []
total_frames    = 0
detected_frames = 0
skipped_frames  = 0

# ── CSV LOG ───────────────────────────────────────────────────────────
log_dir  = '/home/user/emotion_logs'
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"emotion_log_{session_start.strftime('%Y%m%d_%H%M%S')}.csv")

csv_file   = open(log_path, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'emotion', 'confidence', 'fps', 'latency_ms'])

print(f"[INFO] Log disimpan ke: {log_path}")
print("Tekan 'q' untuk keluar  |  's' untuk simpan snapshot")

# ── HELPER: build overlay ─────────────────────────────────────────────
def build_overlay(emotion_durations, conf_accum, session_start):
    lines   = []
    elapsed = (datetime.now() - session_start).total_seconds()
    lines.append(f"Sesi : {int(elapsed//60):02d}:{int(elapsed%60):02d}")
    lines.append("─────────────────────")
    dominant = max(emotion_durations, key=emotion_durations.get)
    for e in EMOTIONS:
        dur   = emotion_durations[e]
        pct   = (dur / elapsed * 100) if elapsed > 0 else 0
        avg_c = (sum(conf_accum[e]) / len(conf_accum[e]) * 100) if conf_accum[e] else 0
        marker = " ◀" if e == dominant else ""
        lines.append(f"{e:<8} {dur:5.1f}s {pct:4.0f}% c:{avg_c:3.0f}%{marker}")
    lines.append("─────────────────────")
    lines.append(f"Dominan: {dominant}")
    return lines

# ── HELPER: draw overlay ──────────────────────────────────────────────
def draw_report_overlay(frame, lines, x=10, y=75):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness  = 1
    pad        = 6
    line_h     = 18

    max_w = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines)
    box_h = len(lines) * line_h + pad * 2
    box_w = max_w + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        ty    = y + pad + (i + 1) * line_h - 3
        color = (200, 200, 200)
        for emo, col in COLORS.items():
            if line.strip().startswith(emo):
                color = col
                break
        if "Dominan" in line or "Sesi" in line or "─" in line:
            color = (0, 220, 220)
        cv2.putText(frame, line, (x + pad, ty), font, font_scale, color, thickness)

# ── HELPER: ringkasan akhir ───────────────────────────────────────────
def print_summary(session_start, log_path,
                  emotion_durations, conf_accum,
                  fps_history, latency_history,
                  total_frames, detected_frames, skipped_frames):

    elapsed  = (datetime.now() - session_start).total_seconds()
    dominant = max(emotion_durations, key=emotion_durations.get)

    avg_fps  = sum(fps_history)     / len(fps_history)     if fps_history     else 0
    min_fps  = min(fps_history)                             if fps_history     else 0
    max_fps  = max(fps_history)                             if fps_history     else 0
    avg_lat  = sum(latency_history) / len(latency_history)  if latency_history else 0
    min_lat  = min(latency_history)                         if latency_history else 0
    max_lat  = max(latency_history)                         if latency_history else 0
    det_rate = (detected_frames / total_frames * 100)       if total_frames    else 0

    all_conf = [c for lst in conf_accum.values() for c in lst]
    avg_acc  = (sum(all_conf) / len(all_conf) * 100)        if all_conf        else 0

    W = 52

    def sep(char="─"): return char * W

    print("\n" + "═"*W)
    print("        RINGKASAN SESI DETEKSI EMOSI")
    print("═"*W)
    print(f"  Waktu mulai  : {session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Durasi sesi  : {int(elapsed//60):02d}:{int(elapsed%60):02d}  ({elapsed:.1f} detik)")
    print(f"  Mode         : Direct (tanpa buffer) — Jetson Nano")
    print(f"  Log CSV      : {os.path.basename(log_path)}")

    print("\n" + sep())
    print("  PERFORMA SISTEM")
    print(sep())
    print(f"  {'Metrik':<22} {'Rata-rata':>10} {'Min':>8} {'Max':>8}")
    print(sep("·"))
    print(f"  {'FPS':<22} {avg_fps:>10.1f} {min_fps:>8.1f} {max_fps:>8.1f}")
    print(f"  {'Latency (ms)':<22} {avg_lat:>10.1f} {min_lat:>8.1f} {max_lat:>8.1f}")

    print("\n" + sep())
    print("  DETECTION RATE")
    print(sep())
    print(f"  Total frame dibaca      : {total_frames:>6}")
    print(f"  Frame terdeteksi        : {detected_frames:>6}  ({det_rate:.1f}%)")
    print(f"  Frame dilewati (filter) : {skipped_frames:>6}  ({100-det_rate:.1f}%)")
    print(f"  Detection success rate  : {det_rate:>5.1f}%")

    print("\n" + sep())
    print("  AKURASI (RATA-RATA CONFIDENCE PER EMOSI)")
    print(sep())
    print(f"  {'Emosi':<10} {'Conf Rata²':>12} {'Sampel':>8}")
    print(sep("·"))
    for e in EMOTIONS:
        avg_c = (sum(conf_accum[e]) / len(conf_accum[e]) * 100) if conf_accum[e] else 0
        n     = len(conf_accum[e])
        print(f"  {e:<10} {avg_c:>11.1f}% {n:>8}")
    print(sep("·"))
    print(f"  {'KESELURUHAN':<10} {avg_acc:>11.1f}%")

    print("\n" + sep())
    print("  DISTRIBUSI EMOSI")
    print(sep())
    print(f"  {'Emosi':<10} {'Durasi':>8} {'%':>7} {'Bar':<20}")
    print(sep("·"))
    for e in EMOTIONS:
        dur  = emotion_durations[e]
        pct  = (dur / elapsed * 100) if elapsed > 0 else 0
        bar  = "█" * int(pct / 5)
        mark = " ◀ DOMINAN" if e == dominant else ""
        print(f"  {e:<10} {dur:>7.1f}s {pct:>6.1f}% {bar:<20}{mark}")

    print("\n" + "═"*W)
    print("[INFO] Selesai.")

# ═════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════════════════════════════════
while True:
    start_time = time.time()

    ret, frame = cap.read()
    total_frames += 1

    if not ret or frame is None:
        continue

    frame = cv2.resize(frame, (640, 480))

    # ── Deteksi wajah ─────────────────────────────────────────────────
    face_results = face_model.predict(frame, conf=0.5, imgsz=640, verbose=False)
    boxes        = face_results[0].boxes
    frame_detected = False

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            H, W = frame.shape[:2]
            PAD  = 10
            x1c  = max(0, x1 - PAD);  y1c = max(0, y1 - PAD)
            x2c  = min(W, x2 + PAD);  y2c = min(H, y2 + PAD)

            face_crop = frame[y1c:y2c, x1c:x2c]
            if face_crop.size == 0:
                continue

            # Preprocess
            gray_crop  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_crop  = clahe.apply(gray_crop)
            gray_crop  = cv2.resize(gray_crop, (48, 48))
            face_input = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)

            # ── Prediksi langsung — tanpa buffer ──────────────────
            results   = emotion_model.predict(face_input, imgsz=64, verbose=False)
            probs     = results[0].probs
            top1      = probs.top1
            top1_conf = float(probs.top1conf)
            top2_conf = float(probs.top5conf[1]) if len(probs.top5conf) > 1 else 0.0
            emotion   = EMOTIONS[top1]
            conf_gap  = top1_conf - top2_conf

            # Filter confidence
            if emotion == "Sad" and top1_conf < 0.60 and conf_gap < 0.20:
                skipped_frames += 1
                continue
            if emotion != "Sad" and top1_conf < min_conf:
                skipped_frames += 1
                continue

            # Langsung pakai hasil prediksi — tanpa voting
            stable_emotion = emotion

            # Drowsy alert — pakai GPIO jika ada buzzer di Jetson
            if stable_emotion == "Drowsy" and top1_conf > 0.70:
                # winsound tidak tersedia di Linux
                # Ganti dengan GPIO buzzer atau print warning
                print("[ALERT] DROWSY DETECTED!")
                # Contoh pakai GPIO (uncomment jika buzzer terpasang):
                # import Jetson.GPIO as GPIO
                # GPIO.output(buzzer_pin, GPIO.HIGH)
                # time.sleep(0.3)
                # GPIO.output(buzzer_pin, GPIO.LOW)

            # Gambar bbox
            color = COLORS.get(stable_emotion, (255, 255, 255))
            label = f"{stable_emotion} {top1_conf*100:.0f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Statistik
            now = time.time()
            emotion_counts[stable_emotion] += 1
            conf_accum[stable_emotion].append(top1_conf)

            if stable_emotion != last_stable:
                last_stable       = stable_emotion
                last_stable_start = now
            else:
                emotion_durations[stable_emotion] += now - last_stable_start
                last_stable_start = now

            frame_detected = True

            # CSV log
            end_time_log = time.time()
            fps_log      = 1.0 / (end_time_log - prev_time + 1e-9)
            lat_log      = (end_time_log - start_time) * 1000
            csv_writer.writerow([
                datetime.now().strftime('%H:%M:%S.%f')[:-3],
                stable_emotion,
                f"{top1_conf:.4f}",
                f"{fps_log:.1f}",
                f"{lat_log:.1f}"
            ])

    if frame_detected:
        detected_frames += 1

    # ── Refresh overlay ───────────────────────────────────────────────
    now = time.time()
    if now - last_report_time >= report_interval:
        overlay_lines    = build_overlay(emotion_durations, conf_accum, session_start)
        last_report_time = now

    if overlay_lines:
        draw_report_overlay(frame, overlay_lines)

    # ── FPS & Latency ─────────────────────────────────────────────────
    end_time  = time.time()
    latency   = (end_time - start_time) * 1000
    fps       = 1.0 / (end_time - prev_time + 1e-9)
    prev_time = end_time

    fps_history.append(fps)
    latency_history.append(latency)

    cv2.putText(frame, f"FPS: {fps:.1f}",            (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Latency: {latency:.1f} ms", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('Emotion Recognition — Jetson Nano', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        snap_path = os.path.join(
            log_dir, f"snapshot_{datetime.now().strftime('%H%M%S')}.jpg"
        )
        cv2.imwrite(snap_path, frame)
        print(f"[INFO] Snapshot disimpan: {snap_path}")

# ═════════════════════════════════════════════════════════════════════
# RINGKASAN AKHIR
# ═════════════════════════════════════════════════════════════════════
csv_file.close()
cap.release()
cv2.destroyAllWindows()

print_summary(
    session_start, log_path,
    emotion_durations, conf_accum,
    fps_history, latency_history,
    total_frames, detected_frames, skipped_frames
)