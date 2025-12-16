import numpy as np
import cv2
import torch
from ultralytics import YOLO
import pandas as pd
import scipy
import matplotlib
import sklearn

print("Environment check:")
print("numpy:", np.__version__)
print("opencv:", cv2.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("ultralytics:", YOLO)
print("pandas:", pd.__version__)
print("scipy:", scipy.__version__)
print("matplotlib:", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)

print("✅ YOLO environment looks good")