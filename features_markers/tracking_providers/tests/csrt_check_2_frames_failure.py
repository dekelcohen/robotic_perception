# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 00:18:07 2025

@author: DEKELCO
"""

import cv2
from PIL import Image
from tracking_providers.csrt_tracker_provider import CSRTTrackerProvider  # replace with actual import path

# ---------- Configuration ----------
video_path = r"C:\Users\dekelco\.cache\huggingface\lerobot\lerobot\svla_so100_pickplace\videos\chunk-000\observation.images.top\episode_000000.mp4"
initial_bbox = [300, 272, 339, 314]  # [x1, y1, x2, y2]

# ---------- Initialize Tracker ----------
tracker = CSRTTrackerProvider(initial_bbox)

# ---------- Open Video ----------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {video_path}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
second_frame_index = frame_count // 2  # pick frame from second half

# ---------- Read First Frame ----------
ret, frame1 = cap.read()
if not ret:
    raise RuntimeError("Failed to read first frame")
pil_frame1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

ok1, bbox1 = tracker.update(pil_frame1)
print("First frame tracking result:", ok1, bbox1)

# ---------- Seek to Second Half ----------
cap.set(cv2.CAP_PROP_POS_FRAMES, second_frame_index)
ret, frame2 = cap.read()
if not ret:
    raise RuntimeError(f"Failed to read frame at index {second_frame_index}")
pil_frame2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

ok2, bbox2 = tracker.update(pil_frame2)
print("Second frame (later) tracking result:", ok2, bbox2)

# ---------- Cleanup ----------
cap.release()
