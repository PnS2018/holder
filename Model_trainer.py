import cv2
import numpy as np
from Classification_model import Classification_model as Cm


# Script for training different models

cm0 = Cm("d(17_5)_s(64)_b(100)_e(400)_r(0.7)_w(0_1)_h(0_1)_z(0_25)_m5")
cm0.set_parameter(64,100,400,0.7,0.1,0.1,0.25,5)
cm0.load_model()
cm0.train_model()

cm1 = Cm("d(17_5)_s(64)_b(100)_e(400)_r(0.7)_w(0_1)_h(0_1)_z(0_25)_m6")
cm1.set_parameter(64,100,400,0.7,0.1,0.1,0.25,6)
cm1.load_model()
cm1.train_model()

cm2 = Cm("d(17_5)_s(64)_b(100)_e(400)_r(0.7)_w(0_1)_h(0_1)_z(0_25)_m7")
cm2.set_parameter(64,100,400,0.7,0.1,0.1,0.25,7)
cm2.load_model()
cm2.train_model()

cm3 = Cm("d(17_5)_s(64)_b(50)_e(600)_r(0.7)_w(0_1)_h(0_1)_z(0_25)_m7")
cm3.set_parameter(64,50,600,0.7,0.1,0.1,0.25,7)
cm3.load_model()
cm3.train_model()
