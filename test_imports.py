import sys

try:
    print("Python version:", sys.version)
    
    import numpy as np
    print("Numpy version:", np.__version__)
    
    import cv2
    print("OpenCV version:", cv2.__version__)
    
    import dlib
    print("Dlib version:", dlib.__version__)
    
    import face_recognition
    print("Face Recognition version:", face_recognition.__version__)
    
    print("\n--- Diagnostic Check ---")
    print("SUCCESS: All packages imported successfully!")
    
except ImportError as e:
    print("\n--- Diagnostic Check ---")
    print("FAILURE: Could not import a package.")
    print("Error details:", e)
    sys.exit(1)
