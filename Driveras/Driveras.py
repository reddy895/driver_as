import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import csv
from datetime import datetime
import platform
import os
import pygame

# ------------------------------
# 1. USER SETTINGS – you can adjust these!
# ------------------------------
EAR_THRESHOLD = 0.23
PERCLOS_THRESHOLD = 0.5
YAWN_THRESHOLD = 0.6   # 🔥 Adjusted for better yawn detection
HEAD_PITCH_THRESHOLD = 20
HEAD_YAW_THRESHOLD = 30
CLOSED_EYES_TIME = 2.0

WINDOW_DURATION = 3
FPS_ESTIMATE = 30

ALERT_SCORE_THRESHOLD = 50
ALERT_COOLDOWN = 3
last_alert_time = 0

# ------------------------------
# 2. INITIALIZE MEDIAPIPE FACE MESH
# ------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------
# 3. LANDMARK INDICES
# ------------------------------
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [61, 146, 91, 181, 84, 17]
HEAD_POSE_IDX = [1, 152, 33, 263, 61, 291]

# ------------------------------
# 4. 3D MODEL POINTS FOR HEAD POSE
# ------------------------------
model_points = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0]
], dtype=np.float64)

# ------------------------------
# 5. INITIALIZE WEBCAM
# ------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ ERROR: Could not open webcam.")
    exit()

# ------------------------------
# 6. PREPARE AUDIO ALERT
# ------------------------------
pygame.mixer.init()
ALARM_FILE = "beep.wav"
USE_SYSTEM_BEEP = False

if os.path.exists(ALARM_FILE):
    alarm_sound = pygame.mixer.Sound(ALARM_FILE)
else:
    USE_SYSTEM_BEEP = True

def play_alert():
    if USE_SYSTEM_BEEP:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 500)
        else:
            print('\a')
    else:
        if not pygame.mixer.get_busy():
            alarm_sound.play()

# ------------------------------
# 7. DATA STRUCTURES
# ------------------------------
eye_closed_history = deque(maxlen=int(FPS_ESTIMATE * WINDOW_DURATION))
yawn_history = deque(maxlen=int(FPS_ESTIMATE * WINDOW_DURATION))
head_nod_history = deque(maxlen=int(FPS_ESTIMATE * WINDOW_DURATION))

eye_closed_start = None
eye_closed_total = 0.0

prev_time = time.time()
fps = 0

# Logging
log_file = open("drowsiness_log.csv", "a", newline="")
csv_writer = csv.writer(log_file)
if os.path.getsize("drowsiness_log.csv") == 0:
    csv_writer.writerow(["Timestamp", "Event", "Drowsiness_Score", "EAR", "MAR", "Head_Pitch"])

# ------------------------------
# 8. HELPER FUNCTIONS (UNCHANGED)
# ------------------------------
def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth_landmarks):
    A = np.linalg.norm(mouth_landmarks[1] - mouth_landmarks[5])
    B = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[4])
    C = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[3])
    return (A + B) / (2.0 * C)

def get_head_pose(landmarks, image_w, image_h):
    image_points = np.array([
        landmarks[1], landmarks[152], landmarks[33],
        landmarks[263], landmarks[61], landmarks[291]
    ], dtype=np.float64)

    focal_length = image_w
    center = (image_w/2, image_h/2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    if not success:
        return 0, 0, 0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    pose_matrix = cv2.hconcat((rotation_matrix, translation_vec))
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_matrix)

    pitch = angles[0][0]
    yaw = angles[1][0]
    roll = angles[2][0]

    return pitch, yaw, roll

def calculate_drowsiness_score(ear, perclos, yawn_freq, head_pitch, eye_duration):
    score = 0
    if ear < EAR_THRESHOLD:
        score += 20
    score += min(perclos * 40, 40)
    score += min(yawn_freq * 20, 20)

    if head_pitch > HEAD_PITCH_THRESHOLD:
        score += 10
        score += min((head_pitch - HEAD_PITCH_THRESHOLD) * 0.5, 10)

    score += min(eye_duration * 10, 20)
    return min(int(score), 100)

# ------------------------------
# 9. MAIN LOOP
# ------------------------------
print("🚗 Driver Drowsiness Detection System STARTED")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    ear = mar = pitch = yaw = roll = 0.0
    face_detected = False
    drowsiness_score = 0
    perclos = yawn_freq = 0.0
    head_nod_now = False

    if results.multi_face_landmarks:
        face_detected = True
        landmarks = results.multi_face_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            None,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
        )

        landmark_points = np.array(
            [[int(lm.x*w), int(lm.y*h)] for lm in landmarks.landmark]
        )

        left_eye = landmark_points[LEFT_EYE_IDX]
        right_eye = landmark_points[RIGHT_EYE_IDX]
        ear = (eye_aspect_ratio(left_eye) +
               eye_aspect_ratio(right_eye)) / 2.0

        cv2.polylines(frame, [left_eye], True, (0,255,0), 2)
        cv2.polylines(frame, [right_eye], True, (0,255,0), 2)

        mouth = landmark_points[MOUTH_IDX]
        mar = mouth_aspect_ratio(mouth)
        cv2.polylines(frame, [mouth], True, (0,255,255), 2)

        pitch, yaw, roll = get_head_pose(landmark_points, w, h)

        eye_closed_now = ear < EAR_THRESHOLD
        yawning_now = mar > YAWN_THRESHOLD
        head_nod_now = pitch > HEAD_PITCH_THRESHOLD

        eye_closed_history.append(eye_closed_now)
        yawn_history.append(yawning_now)
        head_nod_history.append(head_nod_now)

        perclos = sum(eye_closed_history) / len(eye_closed_history)
        yawn_freq = sum(yawn_history) / len(yawn_history)

        if eye_closed_now:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            eye_closed_total = time.time() - eye_closed_start
        else:
            eye_closed_start = None
            eye_closed_total = 0.0

        drowsiness_score = calculate_drowsiness_score(
            ear, perclos, yawn_freq, pitch, eye_closed_total
        )

    # 🔊 ALERT LOGIC
    if face_detected:
        current_time = time.time()
        if drowsiness_score > ALERT_SCORE_THRESHOLD:
            if current_time - last_alert_time > ALERT_COOLDOWN:
                play_alert()
                last_alert_time = current_time

    # Extra status display
    cv2.putText(frame, f"Score: {drowsiness_score}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(frame, f"Yawning: {'YES' if mar > YAWN_THRESHOLD else 'NO'}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Head Pitch: {pitch:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
    cv2.putText(frame, f"Eye Closed: {eye_closed_total:.1f}s", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

face_mesh.close()
cap.release()
cv2.destroyAllWindows()
log_file.close()
pygame.mixer.quit()
print("✅ System stopped.")