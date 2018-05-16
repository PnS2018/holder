import cv2
import numpy as np
from Classification_model import Classification_model as Cm


# Script for training different models

cm0 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(1)_w(0_1)_h(0_1)_z(0_25)_m2")
cm0.set_parameter(64,100,400,1,0.1,0.1,0.25,2)
cm0.load_model()
cm0.train_model()

cm1 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(0_5)_w(0.1)_h(0.1)_z(0_25)_m2")
cm1.set_parameter(64,100,400,0.5,0.1,0.1,0.25,2)
cm1.load_model()
cm1.train_model()

cm2 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(1)_w(0_2)_h(0_2)_z(0_25)_m2")
cm2.set_parameter(64,100,400,1,0.2,0.2,0.25,2)
cm2.load_model()
cm2.train_model()

cm3 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(0_5)_w(0.2)_h(0.2)_z(0_25)_m2")
cm3.set_parameter(64,100,400,0.5,0.2,0.2,0.25,2)
cm3.load_model()
cm3.train_model()

cm4 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(0_3)_w(0.1)_h(0.1)_z(0_25)_m2")
cm4.set_parameter(64,100,400,0.3,0.1,0.1,0.25,2)
cm4.load_model()
cm4.train_model()

cm5 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(0_3)_w(0.2)_h(0.2)_z(0_25)_m2")
cm5.set_parameter(64,100,400,0.3,0.2,0.2,0.25,2)
cm5.load_model()
cm5.train_model()

cm6 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(1)_w(0_1)_h(0_1)_z(0_25)_m3")
cm6.set_parameter(64,100,400,1,0.1,0.1,0.25,3)
cm6.load_model()
cm6.train_model()

cm7 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(0_5)_w(0.1)_h(0.1)_z(0_25)_m3")
cm7.set_parameter(64,100,400,0.5,0.1,0.1,0.25,3)
cm7.load_model()
cm7.train_model()

cm8 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(1)_w(0_2)_h(0_2)_z(0_25)_m3")
cm8.set_parameter(64,100,400,1,0.2,0.2,0.25,3)
cm8.load_model()
cm8.train_model()

cm9 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(0_5)_w(0.2)_h(0.2)_z(0_25)_m3")
cm9.set_parameter(64,100,400,0.5,0.2,0.2,0.25,3)
cm9.load_model()
cm9.train_model()

cm10 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(0_3)_w(0.1)_h(0.1)_z(0_25)_m3")
cm10.set_parameter(64,100,400,0.3,0.1,0.1,0.25,3)
cm10.load_model()
cm10.train_model()

cm11 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(0_3)_w(0.2)_h(0.2)_z(0_25)_m3")
cm11.set_parameter(64,100,400,0.3,0.2,0.2,0.25,3)
cm11.load_model()
cm11.train_model()

cm12 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(1)_w(0_1)_h(0_1)_z(0_25)_m4")
cm12.set_parameter(64,100,400,1,0.1,0.1,0.25,4)
cm12.load_model()
cm12.train_model()

cm13 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(0_5)_w(0.1)_h(0.1)_z(0_25)_m4")
cm13.set_parameter(64,100,400,0.5,0.1,0.1,0.25,4)
cm13.load_model()
cm13.train_model()

cm14 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(1)_w(0_2)_h(0_2)_z(0_25)_m4")
cm14.set_parameter(64,100,400,1,0.2,0.2,0.25,4)
cm14.load_model()
cm14.train_model()

cm15 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(0_5)_w(0.2)_h(0.2)_z(0_25)_m4")
cm15.set_parameter(64,100,400,0.5,0.2,0.2,0.25,4)
cm15.load_model()
cm15.train_model()

cm16 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(0_3)_w(0.1)_h(0.1)_z(0_25)_m4")
cm16.set_parameter(64,100,400,0.3,0.1,0.1,0.25,4)
cm16.load_model()
cm16.train_model()

cm17 = Cm("d(10_5)_s(64)_b(100)_e(400)_r(0_3)_w(0.2)_h(0.2)_z(0_25)_m4")
cm17.set_parameter(64,100,400,0.3,0.2,0.2,0.25,4)
cm17.load_model()
cm17.train_model()
