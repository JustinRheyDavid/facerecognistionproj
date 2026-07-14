import unittest
from unittest.mock import patch

import camera_utils


class FakeCapture:
    def __init__(self, opened):
        self._opened = opened

    def isOpened(self):
        return self._opened


class CameraUtilsTests(unittest.TestCase):
    def test_open_camera_falls_back_to_next_index(self):
        calls = []

        def fake_video_capture(index, backend):
            calls.append((index, backend))
            if index == 1:
                return FakeCapture(True)
            return FakeCapture(False)

        with patch("camera_utils.cv2.VideoCapture", side_effect=fake_video_capture):
            cap, used_index, used_backend = camera_utils.open_camera(indexes=(0, 1))

        self.assertIsNotNone(cap)
        self.assertEqual(used_index, 1)
        self.assertIsNotNone(used_backend)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
