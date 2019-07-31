import sys
import os
import datetime
import time
from pyangel import angelps
import numpy as np
ps = angelps.AngelPs()
grad = np.ones((500, 10), np.float)

ps.create_tensor("tensor", (500, 10), np.float)
res = ps.pull(["tensor"])
print("create tensor: " + str(res))
ps.push(["tensor"], [grad])
ps.update()
res = ps.pull(['tensor'])
print("update tensor: " + str(res))
