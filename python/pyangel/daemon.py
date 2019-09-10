import sys
import asyncio
import os
import sys
import uvloop


user_script = os.environ.get('user_script', '/home/uuirs/dev/python_test/demo.py')


class DaemonProtocol(asyncio.Protocol):

    def connection_made(self, transport):
        print('pipe opened', file=sys.stderr, flush=True)
        super(DaemonProtocol, self).connection_made(transport=transport)

    def data_received(self, data):
        print('received: {!r}'.format(data), file=sys.stderr, flush=True)
        print(data.decode(), file=sys.stderr, flush=True)
        print(len(data), file=sys.stdout, flush=True)
        for i in range(len(data)):
            asyncio.create_task(start_process(user_script, env=os.environ.copy()))
        super(DaemonProtocol, self).data_received(data)

    def connection_lost(self, exc):
        print('pipe closed', file=sys.stderr, flush=True)
        super(DaemonProtocol, self).connection_lost(exc)


async def start_process(path, *args, **kwds):
    while True:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, path, *args, **kwds)
        res = await proc.wait()
        if res == 0:
            break


if __name__ == "__main__":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()
    try:
        stdin_pipe_reader = loop.connect_read_pipe(DaemonProtocol, sys.stdin)
        loop.run_until_complete(stdin_pipe_reader)
        loop.run_forever()
    finally:
        loop.close()
