import cv2
import numpy as np
from Classification_model import Classification_model as Cm


# Script for training different models


cm = Cm("conv_net_not_real_time")
cm.load_model()
cm.train_model()
