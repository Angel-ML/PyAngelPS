import sys
import os
import asyncio
from .datastore import DataStore
from pyarrow import plasma
from grpclib.client import Channel
from .serde import _DTYPE_NP_TO_JVM
from .client_worker_grpc import ClientWorkerStub
from .client_worker_pb2 import *


class AngelPs:

    def __init__(self):
        self.jvm_port = int(os.environ.get("jvm_port"))
        self.plasma_name = os.environ.get("plasma_name")
        self.task_id = -1

        self.data_store = DataStore(self.plasma_name)

        self.key_matid = {}
        self.key_objectid = {}
        self.key_param = {}
        self.key_grad = {}
        self.keys = []

        self.channel = Channel('127.0.0.1', self.jvm_port)
        self.client_worker = ClientWorkerStub(self.channel)

        self.epoch = 0
        self.batch = 0
        self.batch_size = 0

    def init(self, keys, values):
        pass

    def create_tensor(self, key, shape, dtype):
        res = self.client_worker.CreateTensor(RPCTensor(taskId=self.task_id, name=key, dim=len(shape), shape=shape,
                                                  dtype= _DTYPE_NP_TO_JVM[dtype]))
        if self.task_id == -1:
            self.task_id = res.
    def create(self, keys, values):
        def create(key, value):


    def pull(self, keys=None):
        tasks = []
        for key in keys:
            task = asyncio.create_task(self.apull(key))
            tasks.append(task)
        asyncio.gather(*tasks)


    def push(self, keys=None):
        tasks = []
        for key in keys:
            task = asyncio.create_task(self.apush(key))
            tasks.append(task)
        asyncio.gather(*tasks)

    def load(self, keys=None, values=None):
        pass

    def save(self, keys=None, values=None):
        pass

    def update(self, keys=None, values=None):
        pass

    async def apull(self, key):

        if key not in self.key_matid:
            num = len(self.key_matid) + 1
            self.key_matid[key] = num

        res = await self.client_worker.Pull(
            PullRequest(taskId=self.task_id, matId=self.key_matid[key], epoch=self.epoch, batch=self.batch,
                        objectId=0))

        object_id = res.objectId

        data = await self.data_store.aget(object_id)

        self.key_param[key] = data

    async def apush(self, key):
        object_id = self.key_objectid[key]
        data = self.key_grad[key]

        await self.data_store.aset(object_id, data)

        await self.client_worker.Push(
            PushRequest(taskId=self.task_id, matId=self.key_matid[key], epoch=self.epoch, batch=self.batch,
                        batchSize=self.batch_size, objectId=object_id))

    async def aupdate(self, key):
        object_id = self.key_objectid[key]
        matid = self.key_matid[key]
        await self.client_worker.Update(TensorLike(taskId=self.task_id, name='', matId=matid))

    async def acreate(self, key):
        param = self.key_param[key]


        #await self.client_worker.CreateEmbedding(RPCEmbedding(taskId=self.task_id, name=key, ))
        await self.client_worker.CreateTensor(RPCTensor(taskId=self.task_id, name=key, dim=len(param.shape),
                                                        shape=param.shape, dtype=_DTYPE_NP_TO_JVM[param.dtype]))









