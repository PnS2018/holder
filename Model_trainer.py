import cv2
import numpy as np
from Classification_model import Classification_model as Cm


# Script for training different models

cm2 = Cm("d(17_5)_s(64)_b(100)_e(400)_r(0.7)_w(0_1)_h(0_1)_z(0_25)_m8")
cm2.set_parameter(64,100,400,0.7,0.1,0.1,0.25,8)
cm2.load_model()
cm2.train_model()