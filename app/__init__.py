"""TIER CHECK — a meme face-tier scanner built with FastAPI and OpenCV."""

import os

# OpenCV 5 prints a dnn backend warning on every detector creation. Silence it
# before cv2 is imported anywhere in the package.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

__version__ = "1.0.0"
