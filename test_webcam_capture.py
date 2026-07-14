import cv2
import sys
import os

from camera_utils import open_camera

print("--- Webcam Capture Diagnostic ---")
print("Initializing webcam...")
cap, index_used, backend_used = open_camera()

if cap is None:
    print("ERROR: Could not open any webcam device.")
    sys.exit(1)

print(f"Webcam successfully opened at index {index_used} using backend {backend_used}.")
print("Capturing a few frames to warm up...")

for i in range(1, 11):
    success, frame = cap.read()
    if not success or frame is None:
        print(f"ERROR: Failed to read frame {i}")
        cap.release()
        sys.exit(1)
    else:
        print(f"Frame {i} read successfully. Size: {frame.shape}")

# Save the last frame
output_path = "webcam_test.png"
cv2.imwrite(output_path, frame)
print(f"SUCCESS: Saved captured frame to '{output_path}'")

cap.release()
print("Webcam released.")
