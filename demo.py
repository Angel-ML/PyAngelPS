import sys
import os
import datetime
import time
from pyangel import angelps
import numpy as np
import grpc
os.environ['jvm_port']='9005'
os.environ['plasma_name']='/tmp/plasma'
ps = angelps.AngelPs()
ps.batch_size = 1
grad = np.ones((500, 10), np.float)
key = '1246'
ps.create_tensor(key, (500, 10), np.float)
ps.init()
print(ps.key_matid.items())
print(ps.task_id)

res = ps.pull([key])[0]
print("create tensor: " + str(res))
for i in range(3):
    ps.push([key], [grad])
    res = ps.pull([key])[0]
    print("update tensor: " + str(res))
    print("shape: " + str(res.shape))

