import cv2
import sys
import os

print("--- Webcam Capture Diagnostic ---")
print("Initializing cv2.VideoCapture(0)...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam at index 0.")
    print("Let's try index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Could not open webcam at index 1 either.")
        sys.exit(1)

print("Webcam successfully opened!")
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
