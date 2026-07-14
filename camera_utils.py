import cv2


def open_camera(indexes=(0, 1, 2)):
    """Open a webcam using a Windows-friendly backend fallback."""
    backend_attempts = []
    if hasattr(cv2, "CAP_DSHOW"):
        backend_attempts.append((cv2.CAP_DSHOW, "CAP_DSHOW"))
    backend_attempts.append((None, "default"))

    for index in indexes:
        for backend, label in backend_attempts:
            try:
                if backend is None:
                    cap = cv2.VideoCapture(index)
                else:
                    cap = cv2.VideoCapture(index, backend)
            except Exception as exc:
                print(f"Camera index {index} with backend {label} raised: {exc}")
                continue

            if cap.isOpened():
                print(f"Opened camera index {index} using backend {label}.")
                return cap, index, label

            print(f"Camera index {index} with backend {label} was not available.")

    return None, None, None
