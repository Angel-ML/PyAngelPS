import sys
import os
import datetime
import time
idx = os.environ.get('python_id', 0)
assert idx != 0
print('begin ' + str(idx))
time.sleep(3)
print('end ' + str(idx))
