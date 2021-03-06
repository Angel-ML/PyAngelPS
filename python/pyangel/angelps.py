import sys
import os
import asyncio
from .datastore import DataStore
from pyarrow import plasma
from grpclib.client import Channel
from .serde import _DTYPE_NP_TO_PROTO
from .client_worker_grpc import ClientWorkerStub
from .client_worker_pb2 import *
import threading
import time

def _loop_mgr(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def async_map(func, argvs):
    tasks = []
    for argv in argvs:
        if isinstance(argv, tuple):
            task = asyncio.create_task(func(*argv))
        else:
            task = asyncio.create_task(func(argv))
        tasks.append(task)
    await asyncio.gather(*tasks)


class AngelPs:

    def __init__(self):
        self.jvm_port = int(os.environ.get("jvm_port"))
        self.plasma_name = os.environ.get("plasma_name")
        self.task_id = -1

        self.key_matid = {}
        self.key_objectid = {}
        self.key_param = {}
        self.key_grad = {}
        self.key_indice = {}
        self.keys = set()

        self._loop = asyncio.get_event_loop()
        self._thread = threading.Thread(target=_loop_mgr, args=(self._loop,), daemon=True)
        self._thread.start()

        self.data_store = DataStore(self._loop, self.plasma_name)

        self.channel = Channel('127.0.0.1', self.jvm_port, loop=self._loop)
        self.client_worker = ClientWorkerStub(self.channel)

        self.epoch = 0
        self.batch = 0
        self.batch_size = 0

    def close(self):
        self.channel.close()

    def create_tensor(self, key, shape, dtype, init_params=None):

        res = asyncio.run_coroutine_threadsafe(
            self.client_worker.CreateTensor(RPCTensor(taskId=self.task_id, name=key, dim=len(shape), shape=shape,
                                                      dtype=_DTYPE_NP_TO_PROTO[dtype])), self._loop).result()

        if self.task_id == -1:
            self.task_id = res.taskId
        self.key_matid[key] = res.matId
        self.keys.add(key)

    def create_variable(self, key, shape, dtype, init_params=None, updater_params=None):
        print('create variable:' + str(dtype) + ' '+ str(_DTYPE_NP_TO_PROTO[dtype]))
        if len(shape) == 1:
            shape = (1, *shape)
        if init_params is None:
            init_params = {}
        if updater_params is None:
            updater_params = {}

        res = asyncio.run_coroutine_threadsafe(
            self.client_worker.CreateVariable(RPCVariable(taskId=self.task_id, name=key, dim=len(shape), shape=shape,
                                                          dtype=_DTYPE_NP_TO_PROTO[dtype],
                                                          initializerParams=init_params,
                                                          updaterParams=updater_params)), self._loop).result()
        if self.task_id == -1:
            self.task_id = res.taskId
        self.key_matid[key] = res.matId
        self.keys.add(key)

    def create_embedding(self, key, num_feats, embedding_size, dtype, init_params=None, updater_params=None):
        print('create embedding:' + str(dtype) + ' '+ str(_DTYPE_NP_TO_PROTO[dtype]))

        res = asyncio.run_coroutine_threadsafe(
            self.client_worker.CreateEmbedding(RPCEmbedding(taskId=self.task_id, name=key, numFeats=num_feats,
                                                            embeddingSize=embedding_size,
                                                            dtype='lk_'+_DTYPE_NP_TO_PROTO[dtype])), self._loop).result()
        if self.task_id == -1:
            self.task_id = res.taskId
        self.key_matid[key] = res.matId
        self.keys.add(key)

    def init(self, keys=None):
        if keys is None:
            keys = self.keys

        asyncio.run_coroutine_threadsafe(async_map(self.ainit, keys), self._loop).result()

    def pull(self, keys=None, indices=None):
        tmpt = time.time()
        if keys is None:
            keys = self.keys

        if indices is None:
            indices = [None] * len(keys)
        print('pull 1', time.time() - tmpt, flush=True)
        tmpt = time.time()
        asyncio.run_coroutine_threadsafe(async_map(self.apull, zip(keys, indices)), self._loop).result()
        print('pull 2', time.time() - tmpt, flush=True)
        return list(map(lambda x: self.key_param[x], keys))

    def push(self, keys=None, values=None):
        if keys is None:
            keys = self.keys
        if values is not None:
            for key, value in zip(keys, values):
                object_id = DataStore.get_rand_id()
                self.key_objectid[key] = object_id
                self.key_grad[key] = value
        asyncio.run_coroutine_threadsafe(async_map(self.apush, keys), self._loop).result()

    def load(self, keys, pathes):
        asyncio.run_coroutine_threadsafe(async_map(self.aload, zip(keys,pathes))).result()

    def save(self, keys, pathes):
        asyncio.run_coroutine_threadsafe(async_map(self.aload, zip(keys,pathes))).result()

    def update(self, keys=None):
        if keys is None:
            keys = self.keys

        asyncio.run_coroutine_threadsafe(async_map(self.aupdate, keys), self._loop).result()

    def sync(self):
        asyncio.run_coroutine_threadsafe(async_map(self.asyncc, [self.task_id]), self._loop).result()

    async def apull(self, key, indices = None):
        if indices is not None:
            self.key_indice[key] = indices
            object_id = self.data_store.get_rand_id()
            await self.data_store.aset(object_id, indices, update=False)

            res = await self.client_worker.Pull(
                PullRequest(taskId=self.task_id, matId=self.key_matid[key], epoch=self.epoch, batch=self.batch,
                            objectId=object_id.binary()))
        else:
            tmpt = time.time()
            res = await self.client_worker.Pull(
                PullRequest(taskId=self.task_id, matId=self.key_matid[key], epoch=self.epoch, batch=self.batch))
            print('apull 1', time.time() - tmpt, flush=True)

        object_id = res.objectId
        tmpt = time.time()

        data = await self.data_store.aget(object_id, update=True)
        print('apull 2', time.time() - tmpt, flush=True)

        self.key_param[key] = data

    def async_pull(self, key, indices = None):
        asyncio.run_coroutine_threadsafe(self.apull(key, indices), self._loop)


    async def apush(self, key):
        tmpt = time.time()
        object_id = self.key_objectid[key]
        data = self.key_grad[key]
        if key in self.key_param and len(self.key_param[key]) == 3:
            data = (self.key_param[key][0], data, self.key_param[key][2])


        await self.data_store.aset(object_id, data, update=True)

        #print('apush data ' + str(time.time() - tmpt))
        await self.client_worker.Push(
            PushRequest(taskId=self.task_id, matId=self.key_matid[key], epoch=self.epoch, batch=self.batch,
                        batchSize=self.batch_size, objectId=object_id.binary()))
        #print('apush ' + str(time.time() - tmpt))

    async def aupdate(self, key):
        matid = self.key_matid[key]
        await self.client_worker.Update(TensorLike(taskId=self.task_id, name=key, matId=matid))

    async def ainit(self, key):
        matid = self.key_matid[key]
        await self.client_worker.Init(TensorLike(taskId=self.task_id, name=key, matId=matid))

    async def aload(self, key, path):
        matid = self.key_matid[key]
        await self.client_worker.Load(LoadTensorLike(taskId=self.task_id, matId=matid, path=path, conf={}))

    async def asave(self, key, path):
        matid = self.key_matid[key]
        await self.client_worker.Save(SaveTensorLike(taskId=self.task_id, matId=matid, path=path, formatClassName=""))

    async def asyncc(self, task_id):
        await self.client_worker.Sync(SyncRequest(taskId=task_id))










