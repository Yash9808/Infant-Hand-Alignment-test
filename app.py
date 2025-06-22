from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import math

app = Flask(__name__)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Finger joints
fingers = {
    "Thumb": [1, 2, 4],
    "Index": [5, 6, 8],
    "Middle": [9, 10, 12],
    "Ring": [13, 14, 16],
    "Pinky": [17, 18, 20]
}

def get_angle(a, b, c):
    def vec(p1, p2):
        return (p2[0]-p1[0], p2[1]-p1[1])
    ba = vec(b, a)
    bc = vec(b, c)
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 0
    cos_angle = max(-1, min(1, dot / mag))
    return math.degrees(math.acos(cos_angle))

def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks.landmark]

                for name, (a, b, c) in fingers.items():
                    angle = get_angle(landmarks[a], landmarks[b], landmarks[c])
                    y = 30 + list(fingers).index(name)*30 + idx*200
                    cv2.putText(frame, f"{name}: {int(angle)}°", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
