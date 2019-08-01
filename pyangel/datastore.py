import pyarrow.plasma as plasma
import pyarrow
from .serde import DataHead
import concurrent.futures

data_head = DataHead()


def _set_data():
    data_head.write_to_buffer()


class DataStore:
    def __init__(self, loop, plasma_name="/tmp/test"):
        self.plasma_client = plasma.connect(plasma_name)
        self._loop = loop

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.plasma_client.disconnect()

    @staticmethod
    def get_rand_id():
        return plasma.ObjectID.from_random()

    async def aget(self, object_id):
        if not isinstance(object_id, pyarrow._plasma.ObjectID):
            object_id = plasma.ObjectID(object_id)
        buffer = self.plasma_client.get_buffers([object_id])[0]
        buffer = memoryview(buffer)
        # data_head = DataHead()
        data_head.from_buffer(buffer)
        return data_head.parse_data(buffer)

    async def aset(self, object_id, data):
        if not isinstance(object_id, pyarrow._plasma.ObjectID):
            object_id = plasma.ObjectID(object_id)
        # data_head = DataHead()
        data_head.from_data(data)

        object_size = data_head.nbytes

        buffer = self.plasma_client.create(object_id, object_size)
        buffer = memoryview(buffer)
        data_head.buffer = buffer

        with concurrent.futures.ProcessPoolExecutor() as pool:
            await self._loop.run_in_executor(pool, _set_data)
        self.plasma_client.seal(object_id)

    def get_all(self, object_ids):
        if not isinstance(object_ids[0], pyarrow._plasma.ObjectID):
            object_ids = list(map(lambda x: plasma.ObjectID(x), object_ids))
        buffers = self.plasma_client.get_buffers(object_ids)

        def get_res(buffer):
            buffer = memoryview(buffer)
            # data_head = DataHead()
            data_head.from_buffer(buffer)
            return data_head.parse_data(buffer)

        return list(map(get_res, buffers))

    def set_all(self, object_ids, data):
        from multiprocessing import Pool
        if not isinstance(object_ids[0], pyarrow._plasma.ObjectID):
            object_ids = list(map(lambda x: plasma.ObjectID(x), object_ids))

        def set_data(idx):
            # data_head = DataHead()
            data_head.from_data(data[idx])

            object_size = data_head.nbytes

            buffer = self.plasma_client.create(object_ids[idx], object_size)
            buffer = memoryview(buffer)
            data_head.write_to_buffer(data[idx], buffer)
        p = Pool(5)
        p.map(set_data, [x for x in range(len(object_ids))])




