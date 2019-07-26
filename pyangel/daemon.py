import sys
import asyncio
import os


async def start_process(path, *args, **kwds):
    while True:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, path, *args, **kwds)
        res = await proc.wait()
        if res == 0:
            break


async def daemon():
    _jvm_port = int(os.environ.get('jvm_port', '0'))
    # assert _jvm_port != 0

    _pool = {}
    job_id = 0
    msg = ''
    while True:
        await asyncio.sleep(2)
        job_id += 1
        num_worker = 3
        if msg is None:
            break
        for i in range(num_worker):
            new_env = os.environ.copy()
            new_env['python_id'] = str(job_id) + '_' + str(i)
            asyncio.create_task(start_process('./demo.py', env=new_env))


asyncio.run(daemon())
