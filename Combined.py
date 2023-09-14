import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import matplotlib.pyplot as plt
from Attendance import *
#from eyedetection import *
#from emotiondetection import *

cap = cv2.VideoCapture(0)
attendance(cap)
#DetectEye(cap) 
#emotions(cap)
