import cv2
import numpy as np
from Classification_model import Classification_model as Cm
from Classification_model2 import Classification_model as Cm2


# Script for training different models


cm = Cm("conv_net_not_real_time")
cm.load_model()
cm.train_model()

cm2 = Cm2("conv_net_real_time")
cm2.load_model()
cm2.train_model()
