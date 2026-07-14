import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

print("--- Running Offline Face Recognition Verification Test ---")

# 1. Verify known images exist
known_dir = 'ImagesAttendance'
image_files = [f for f in os.listdir(known_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
if not image_files:
    print("FAILURE: No test images found in ImagesAttendance directory.")
    exit(1)

test_name = os.path.splitext(image_files[0])[0]
test_image_path = os.path.join(known_dir, image_files[0])
print(f"Using test image: {test_image_path} (Name: {test_name})")

# 2. Load the known image
known_image = cv2.imread(test_image_path)
if known_image is None:
    print(f"FAILURE: Could not read image {test_image_path}")
    exit(1)

# 3. Calculate known encodings
known_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
known_encodings = face_recognition.face_encodings(known_rgb)

if len(known_encodings) == 0:
    print("FAILURE: Could not find any face in the training image.")
    exit(1)

print(f"SUCCESS: Successfully calculated face encoding for {test_name}.")
known_encoding = known_encodings[0]

# 4. Simulate a webcam frame using the same image
# To make it realistic, we'll slightly blur it to simulate frame capture noise
simulated_frame = cv2.GaussianBlur(known_image, (3, 3), 0)
simulated_rgb = cv2.cvtColor(simulated_frame, cv2.COLOR_BGR2RGB)

# 5. Detect face in simulated frame
face_locations = face_recognition.face_locations(simulated_rgb)
face_encodings = face_recognition.face_encodings(simulated_rgb, face_locations)

if len(face_encodings) == 0:
    print("FAILURE: Could not find face in simulated frame.")
    exit(1)

print(f"SUCCESS: Found {len(face_encodings)} face(s) in simulated frame.")

# 6. Compare face encodings
for face_encoding in face_encodings:
    matches = face_recognition.compare_faces([known_encoding], face_encoding)
    face_distances = face_recognition.face_distance([known_encoding], face_encoding)
    
    distance = face_distances[0]
    match = matches[0]
    
    print(f"Distance calculated: {distance:.4f} (Match: {match})")
    
    if match and distance < 0.55:
        print(f"SUCCESS: Identified face as {test_name.upper()}")
        
        # 7. Test Attendance Logging
        attendance_file = 'Attendance.csv'
        if os.path.exists(attendance_file):
            os.remove(attendance_file) # Clear file to start fresh for test
            
        print("Logging attendance...")
        
        # Call logging logic
        def markAttendance(name):
            file_exists = os.path.isfile(attendance_file)
            with open(attendance_file, 'a', encoding='utf-8') as f:
                if not file_exists or os.stat(attendance_file).st_size == 0:
                    f.write("Name,DateTime\n")
                now = datetime.now()
                dtString = now.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{name},{dtString}\n")
                print(f"Logged to CSV: {name} at {dtString}")
        
        markAttendance(test_name.upper())
        
        # Verify CSV content
        if os.path.exists(attendance_file):
            with open(attendance_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print("CSV Contents:")
                for line in lines:
                    print("  ", line.strip())
                if len(lines) >= 2 and test_name.upper() in lines[1]:
                    print("\n--- TEST COMPLETED SUCCESSFULLY ---")
                    exit(0)
                else:
                    print("FAILURE: CSV logging verify check failed.")
                    exit(1)
        else:
            print("FAILURE: CSV file not created.")
            exit(1)
    else:
        print("FAILURE: Face distance too high, face not recognized.")
        exit(1)
