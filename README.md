# Face Recognition & Attendance System 

A real-time, automated face recognition attendance system built using **Python**, **OpenCV**, and the **dlib-based `face_recognition` library**. It captures video from a webcam, matches detected faces against a known database, and logs attendance into a CSV file with date and time.

This project is inspired by the popular Murtaza's Workshop face recognition tutorial, with several robust enhancements such as:
*   Safe face encoding computation (preventing crashes on startup if a database photo has no detectable face).
*   Automatic CSV header initialization.
*   Support for multiple image formats (`.jpg`, `.jpeg`, `.png`, `.webp`).
*   Offline diagnostic verification suite.

---

##  Requirements & Installation

This project is optimized for **Windows** (x64) and **Python 3.11**. Python 3.11 is highly recommended to avoid complex Visual Studio build errors when installing `dlib`.

### 1. Clone the repository
```bash
git clone https://github.com/JustinRheyDavid/facerecognistionproj.git
cd facerecognistionproj
```

### 2. Setup Virtual Environment
Create and activate a virtual environment to isolate the project dependencies:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
To install the dependencies successfully without C++ compilers (like CMake and MSVC), install the precompiled Windows wheel for `dlib` first:
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install precompiled dlib wheel
python -m pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp311-cp311-win_amd64.whl

# Install other packages (OpenCV 4.9.x is used for compatibility with NumPy 1.x)
python -m pip install face-recognition opencv-python==4.9.0.80 numpy<2
```

---

##  Running the Project

### 1. Add Known Faces
Place images of people you want the system to recognize into the `ImagesAttendance/` directory. 
*   Name the image files exactly as the person's name (e.g., `Justin.jpg`, `Elon Musk.png`).
*   Ensure each photo has a clear, well-lit, front-facing view of the person's face.

### 2. Run Diagnostics (Recommended)
Before starting the webcam, run the import diagnostics to check if all libraries are working:
```bash
python test_imports.py
```

Then, run the offline verification test which simulates recognition on the images inside `ImagesAttendance/` and logs attendance to `Attendance.csv`:
```bash
python test_face_rec.py
```

### 3. Run Live Attendance System
Launch the webcam recognition script:
```bash
python main.py
```
*   **Webcam Bounding Boxes**:
    *   🟢 **Green Box**: Face recognized! Bounding box displays the name, and attendance is saved in `Attendance.csv`.
    *   🔴 **Red Box**: Unknown face.
*   **Quit**: Focus the video frame and press **`q`** to close.

---

##  Project Structure

```
facerecognistionproj/
│
├── ImagesAttendance/          # Directory containing images of known individuals (e.g., "Justin.jpg")
│
├── Attendance.csv             # Output file logging Name and DateTime (generated automatically)
├── test_imports.py            # Diagnostic script to verify packages load correctly
├── test_face_rec.py           # Offline verification script to test recognition logic
├── main.py                    # Main script running webcam detection and attendance logging
├── .gitignore                 # Excludes venv, cached files, and local logs from git tracking
└── README.md                  # Project documentation
```

---

