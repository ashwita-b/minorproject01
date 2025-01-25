from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO models
weapon_model = YOLO("D:/threat_page/models/best.pt")  
fight_model = YOLO("D:/threat_page/models/fight.pt")    

# Check class names
print("Weapon Model Classes:", weapon_model.names)
print("Fight Model Classes:", fight_model.names)

# Global variables
cap = None
alert_message = "No threats detected."

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/opencam")
def opencam():
    """Stream webcam feed with real-time detection."""
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Open the camera
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_alert')
def get_alert():
    """Send the current alert status."""
    return jsonify({"alert": alert_message})

@app.route("/shutdown")
def shutdown():
    """Release the webcam."""
    global cap
    if cap:
        cap.release()
        cap = None
    return "Webcam shut down."

def draw_boxes(frame, results, model):
    """Draws bounding boxes with labels on the frame."""
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= 0.7:  # Apply confidence threshold
                    class_id = int(box.cls[0])
                    class_name = model.names.get(class_id, "Unknown")

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label
                    label = f"{class_name} ({conf:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def generate_frames():
    """Capture frames, run YOLO detection, and stream."""
    global cap, alert_message
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run detection with confidence filter
        weapon_results = weapon_model.predict(source=frame, save=False, save_txt=False, conf=0.7, verbose=True)
        fight_results = fight_model.predict(source=frame, save=False, save_txt=False, conf=0.7, verbose=True)

        # Debugging: Print detection results
        print("Weapon Detection Boxes:", weapon_results[0].boxes)
        print("Fight Detection Boxes:", fight_results[0].boxes)

        # Extract detected objects with confidence ≥ 0.7
        detected_weapons = []
        detected_fights = []

        if weapon_results[0].boxes is not None:
            for box in weapon_results[0].boxes:
                if box.conf[0] >= 0.7:
                    class_id = int(box.cls[0])
                    class_name = weapon_model.names.get(class_id, "Unknown")
                    detected_weapons.append(class_name)

        if fight_results[0].boxes is not None:
            for box in fight_results[0].boxes:
                if box.conf[0] >= 0.7:
                    class_id = int(box.cls[0])
                    class_name = fight_model.names.get(class_id, "Unknown")
                    detected_fights.append(class_name)

        # Update alert message
        alerts = []
        if detected_weapons:
            alerts.append(f"Weapon Detected: {', '.join(detected_weapons)}")
        if detected_fights:
            alerts.append(f"Fight Detected: {', '.join(detected_fights)}")

        alert_message = " & ".join(alerts) if alerts else "No Threat Detected"

        # Draw bounding boxes manually
        frame = draw_boxes(frame, fight_results, fight_model)
        frame = draw_boxes(frame, weapon_results, weapon_model)

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == "__main__":
    app.run(debug=True)
