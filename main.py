import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to the directory containing images of known people
path = 'ImagesAttendance'
images = []
classNames = []

# Create directory if it doesn't exist
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Created directory: '{path}'")

# Load images from directory
myList = os.listdir(path)
print(f"Scanning directory '{path}'...")
print(f"Found files: {myList}")

for cl in myList:
    # Accept standard image files
    if cl.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        curImg = cv2.imread(os.path.join(path, cl))
        if curImg is not None:
            images.append(curImg)
            # Use filename without extension as the person's name
            classNames.append(os.path.splitext(cl)[0])
        else:
            print(f"Error loading image: {cl}")

print(f"Loaded {len(images)} valid images with names: {classNames}")

# Function to calculate 128-dimensional face encodings for known faces
def findEncodings(images, names):
    encodeList = []
    for img, name in zip(images, names):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if len(encodings) > 0:
            encodeList.append(encodings[0])
            print(f"Encoding successful for: {name}")
        else:
            print(f"WARNING: No face found in image for '{name}'. This image will be skipped.")
    return encodeList

# Function to record recognized faces in a CSV file
def markAttendance(name):
    filename = 'Attendance.csv'
    file_exists = os.path.isfile(filename)
    
    # Read existing names to avoid duplicate logging in the same day/session
    nameList = []
    if file_exists:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                # Read headers and entries
                lines = f.readlines()
                for line in lines[1:]: # Skip header
                    entry = line.strip().split(',')
                    if entry:
                        nameList.append(entry[0])
        except Exception as e:
            print(f"Error reading attendance file: {e}")

    # If the person has not been logged yet, write entry
    if name not in nameList:
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                # If file is empty or just created, add headers
                if not file_exists or os.stat(filename).st_size == 0:
                    f.write("Name,DateTime\n")
                
                now = datetime.now()
                dtString = now.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{name},{dtString}\n")
                print(f"Attendance marked for {name} at {dtString}")
        except Exception as e:
            print(f"Error writing to attendance file: {e}")

# Calculate encodings for all known faces
print("Calculating face encodings for database...")
encodeListKnown = findEncodings(images, classNames)
print("Encoding computation completed.")

# Initialize webcam
print("Initializing webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Please check your camera connection or settings.")
    print("If your webcam index is different, modify `cv2.VideoCapture(0)` with the correct index.")
    exit(1)

print("\n--- Running Face Recognition ---")
print("Press 'q' in the video window to quit.")

while True:
    success, img = cap.read()
    if not success or img is None:
        print("ERROR: Failed to capture image from webcam. Exiting...")
        break

    # Resize frame to 1/4 size for faster face recognition processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Process each face detected in the webcam stream
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # Compare with known database
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        name = "UNKNOWN"
        box_color = (0, 0, 255) # Red for unknown faces
        
        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)
            # Threshold distance: smaller means closer match. Usually 0.55 or 0.6 is good.
            if matches[matchIndex] and faceDis[matchIndex] < 0.55:
                name = classNames[matchIndex].upper()
                box_color = (0, 255, 0) # Green for recognized faces
                markAttendance(name)

        # Scale coordinates back up to original image size
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        # Draw bounding box and name text
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), box_color, cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

    # Display the result
    cv2.imshow('Face Recognition Attendance System', img)
    
    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("System shut down successfully.")
