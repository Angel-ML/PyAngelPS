import pyarrow.plasma as plasma
from .serde import DataHead
import concurrent.futures
import asyncio


class DataStore:
    def __init__(self, plasma_name="/tmp/test"):
        self.plasma_client = plasma.connect(plasma_name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.plasma_client.disconnect()

    async def aget(self, object_id):
        buffer = self.plasma_client.get_buffers([object_id])
        buffer = memoryview(buffer)
        data_head = DataHead()
        data_head.from_buffer(buffer)
        return data_head.parse_data(buffer)

    async def aset(self, object_id, data):
        data_head = DataHead()
        data_head.from_data(data)

        object_size = data_head.nbytes

        buffer = self.plasma_client.create(object_id, object_size)
        buffer = memoryview(buffer)
        loop = asyncio.get_event_loop()
        with concurrent.futures.ProcessPoolExecutor() as pool:
            await loop.run_in_executor(pool, data_head.write_to_buffer(data, buffer))
        self.plasma_client.seal(object_id)

    def get_all(self, object_ids):
        buffers = self.plasma_client.get_buffers(object_ids)

        def get_res(buffer):
            buffer = memoryview(buffer)
            data_head = DataHead()
            data_head.from_buffer(buffer)
            return data_head.parse_data(buffer)

        return list(map(get_res, buffers))

    def set_all(self, object_ids, data):
        from multiprocessing import Pool

        def set_data(idx):
            data_head = DataHead()
            data_head.from_data(data[idx])

            object_size = data_head.nbytes

            buffer = self.plasma_client.create(object_ids[idx], object_size)
            buffer = memoryview(buffer)
            data_head.write_to_buffer(data[idx], buffer)
        p = Pool(5)
        p.map(set_data, [x for x in range(len(object_ids))])




