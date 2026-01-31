import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

mp_hands = mp.solutions.hands

def extract_hand_joints(video_path, save_path):
    cap = cv2.VideoCapture(str(video_path))
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    all_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            joints = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        else:
            joints = np.zeros((21, 3))  # missing frame

        all_frames.append(joints)

    cap.release()
    all_frames = np.array(all_frames)
    np.save(save_path, all_frames)
    print(f"Saved: {save_path}, shape={all_frames.shape}")


if __name__ == "__main__":
    video_dir = Path("data/raw_videos")
    out_dir = Path("data/joints_clean")
    out_dir.mkdir(parents=True, exist_ok=True)

    for video in video_dir.glob("*.mp4"):
        save_file = out_dir / f"{video.stem}.npy"
        extract_hand_joints(video, save_file)
