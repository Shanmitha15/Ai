from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import cv2
from ultralytics import YOLO
from datetime import datetime

app = FastAPI()
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

object_log = {}
object_counter = 0

def gen_frames():
    global object_counter

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]
        current_ids = []

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            obj_id = f"{label}_{object_counter}"
            object_counter += 1

            current_ids.append(obj_id)

            if obj_id not in object_log:
                object_log[obj_id] = {
                    "label": label,
                    "start": datetime.now(),
                    "end": None
                }

        for obj_id in list(object_log.keys()):
            if obj_id not in current_ids:
                if object_log[obj_id]["end"] is None:
                    object_log[obj_id]["end"] = datetime.now()

        annotated = results.plot()

        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/")
def index():
    return HTMLResponse("""
    <html>
    <head>
        <title>AI Surveillance Dashboard</title>
    </head>
    <body style="background:black;color:white;text-align:center;">
        <h1>AI Surveillance Dashboard</h1>
        <img src="/video" width="60%"/>
        <br><br>
        <a href="/logs" style="color:cyan;">View Logs</a>
    </body>
    </html>
    """)


@app.get("/video")
def video():
    return StreamingResponse(gen_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/logs")
def logs():
    output = []

    for obj_id, data in object_log.items():
        if data["end"] is None:
            continue

        duration = (data["end"] - data["start"]).seconds

        output.append({
            "object": data["label"],
            "start_time": data["start"].strftime("%H:%M:%S"),
            "end_time": data["end"].strftime("%H:%M:%S"),
            "duration_sec": duration
        })

    return JSONResponse(output)
