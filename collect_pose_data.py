import cv2
import mediapipe as mp
import csv
import os

# ====== CONFIGURATION ======
ACTIVITY_NAME = "standing"  # change this for each activity
OUTPUT_DIR = "dataset"
MAX_FRAMES = 300  # ~10 seconds
# ===========================

os.makedirs(OUTPUT_DIR, exist_ok=True)
file_path = os.path.join(OUTPUT_DIR, f"{ACTIVITY_NAME}.csv")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

cap = cv2.VideoCapture(0)

with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    # CSV Header
    header = []
    for i in range(33):
        header += [f"x{i}", f"y{i}"]
    writer.writerow(header)

    frame_count = 0

    print("Recording started... Perform the activity.")

    while cap.isOpened() and frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            row = []
            for landmark in result.pose_landmarks.landmark:
                row.append(landmark.x)
                row.append(landmark.y)

            writer.writerow(row)
            frame_count += 1

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("Pose Recording", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
print("Recording finished. Data saved to CSV.")
