import gradio as gr
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import tempfile

def detect_people(video_file):
    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(video_file.read())
    temp_file.close()

    cap = cv2.VideoCapture(temp_file.name)
    if not cap.isOpened():
        return "Cannot open video file", None

    LINE_START = (100, 300)
    LINE_END = (800, 300)
    track_history = defaultdict(list)
    entry_count = 0
    exit_count = 0
    output_frames = []

    def ccw(A, B, C): return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    def intersects(A, B, C, D): return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, verbose=False)[0]
        dets = []
        for *box, score, cls in result.boxes.data.tolist():
            if int(cls) == 0 and score > 0.4:
                x1, y1, x2, y2 = map(int, box)
                dets.append(([x1, y1, x2 - x1, y2 - y1], float(score), 'person'))

        tracks = tracker.update_tracks(dets, frame=frame)
        cv2.line(frame, LINE_START, LINE_END, (0, 255, 0), 2)

        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            l, t, r, b = map(int, tr.to_ltrb())
            cx, cy = (l + r) // 2, (t + b) // 2
            track_history[tid].append((cx, cy))

            if len(track_history[tid]) >= 2:
                prev, curr = track_history[tid][-2], track_history[tid][-1]
                if intersects(prev, curr, LINE_START, LINE_END):
                    if curr[1] < LINE_START[1]:
                        entry_count += 1
                    else:
                        exit_count += 1
                    track_history[tid].clear()

            cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
            cv2.putText(frame, f"ID:{tid}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(frame, f"Entries: {entry_count}  Exits: {exit_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        output_frames.append(frame)

    cap.release()

    if output_frames:
        output_path = "/tmp/output_frame.jpg"
        cv2.imwrite(output_path, output_frames[-1])
        return f"Entries: {entry_count}, Exits: {exit_count}", output_path
    else:
        return "No frames processed.", None


# Gradio interface
interface = gr.Interface(
    fn=detect_people,
    inputs=gr.Video(label="Upload CCTV Video"),
    outputs=[
        gr.Text(label="Result"),
        gr.Image(type="filepath", label="Last Frame Processed")
    ],
    title="People Entry/Exit Detection",
    description="Upload a CCTV video to detect and count people entering or exiting across a virtual line using YOLOv8 + DeepSORT."
)

if __name__ == "__main__":
    interface.launch()

