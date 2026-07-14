import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

print("--- Initializing Live Webcam Shot Recognition ---")

# Paths
db_path = 'ImagesAttendance'
artifact_dir = r"C:\Users\justi\.gemini\antigravity\brain\60536f6a-d646-4d94-8a32-40072e86a262"
output_image_path = os.path.join(artifact_dir, "webcam_live_test.png")

# Load images
images = []
classNames = []
myList = os.listdir(db_path)

for cl in myList:
    if cl.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        curImg = cv2.imread(os.path.join(db_path, cl))
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])

print(f"Loaded database images: {classNames}")

# Encodings
print("Computing database encodings...")
encodeListKnown = []
for img, name in zip(images, classNames):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(img_rgb)
    if len(encodings) > 0:
        encodeListKnown.append(encodings[0])
        print(f"  Encoded {name}")
    else:
        print(f"  Skipped {name} (no face found)")

# Webcam
print("Opening webcam index 0...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam index 0 could not be opened.")
    exit(1)

# Grab 15 frames for camera to auto-expose and capture a stable image
print("Capturing frames...")
for i in range(15):
    success, img = cap.read()
    if not success or img is None:
        print(f"ERROR: Failed to read frame {i}")
        cap.release()
        exit(1)

cap.release()
print("Webcam released.")

# Process frame
imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

facesCurFrame = face_recognition.face_locations(imgS)
encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

print(f"Found {len(facesCurFrame)} face(s) in webcam frame.")

# Recognition
for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    
    name = "UNKNOWN"
    box_color = (0, 0, 255)
    
    if len(faceDis) > 0:
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex] and faceDis[matchIndex] < 0.55:
            name = classNames[matchIndex].upper()
            box_color = (0, 255, 0)
            
            # Log Attendance
            filename = 'Attendance.csv'
            file_exists = os.path.isfile(filename)
            nameList = []
            if file_exists:
                with open(filename, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[1:]:
                        entry = line.strip().split(',')
                        if entry:
                            nameList.append(entry[0])
            
            if name not in nameList:
                with open(filename, 'a', encoding='utf-8') as f:
                    if not file_exists or os.stat(filename).st_size == 0:
                        f.write("Name,DateTime\n")
                    now = datetime.now()
                    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{name},{dtString}\n")
                    print(f"Attendance marked: {name} at {dtString}")
        
    y1, x2, y2, x1 = faceLoc
    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
    
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), box_color, cv2.FILLED)
    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

# Save result to artifact path
cv2.imwrite(output_image_path, img)
print(f"SUCCESS: Saved live test output image to {output_image_path}")
