import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from transform_utils import *

quat = np.array([-0.013, 0.923, 0.382, -0.032])
euler = quat2euler(quat)/np.pi*180
print(euler)


