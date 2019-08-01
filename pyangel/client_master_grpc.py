# Generated by the Protocol Buffers compiler. DO NOT EDIT!
# source: client_master.proto
# plugin: grpclib.plugin.main
import abc

import grpclib.const
import grpclib.client

import common_pb2
import client_master_pb2


class AngelCleintMasterBase(abc.ABC):

    @abc.abstractmethod
    async def RegisterWorker(self, stream):
        pass

    @abc.abstractmethod
    async def RegisterTask(self, stream):
        pass

    @abc.abstractmethod
    async def SetAngelLocation(self, stream):
        pass

    @abc.abstractmethod
    async def GetAngelLocation(self, stream):
        pass

    @abc.abstractmethod
    async def HeartBeat(self, stream):
        pass

    @abc.abstractmethod
    async def Clock(self, stream):
        pass

    @abc.abstractmethod
    async def GetClockMap(self, stream):
        pass

    @abc.abstractmethod
    async def GetGlobalBatchSize(self, stream):
        pass

    @abc.abstractmethod
    async def CompleteTask(self, stream):
        pass

    def __mapping__(self):
        return {
            '/ClientMaster.AngelCleintMaster/RegisterWorker': grpclib.const.Handler(
                self.RegisterWorker,
                grpclib.const.Cardinality.UNARY_UNARY,
                client_master_pb2.RegisterWorkerReq,
                client_master_pb2.RegisterWorkerResp,
            ),
            '/ClientMaster.AngelCleintMaster/RegisterTask': grpclib.const.Handler(
                self.RegisterTask,
                grpclib.const.Cardinality.UNARY_UNARY,
                client_master_pb2.RegisterTaskReq,
                client_master_pb2.RegisterTaskResp,
            ),
            '/ClientMaster.AngelCleintMaster/SetAngelLocation': grpclib.const.Handler(
                self.SetAngelLocation,
                grpclib.const.Cardinality.UNARY_UNARY,
                client_master_pb2.SetAngelLocationReq,
                common_pb2.VoidResp,
            ),
            '/ClientMaster.AngelCleintMaster/GetAngelLocation': grpclib.const.Handler(
                self.GetAngelLocation,
                grpclib.const.Cardinality.UNARY_UNARY,
                common_pb2.VoidReq,
                client_master_pb2.GetAngelLocationResp,
            ),
            '/ClientMaster.AngelCleintMaster/HeartBeat': grpclib.const.Handler(
                self.HeartBeat,
                grpclib.const.Cardinality.UNARY_UNARY,
                client_master_pb2.HeartBeatReq,
                client_master_pb2.HeartBeatResp,
            ),
            '/ClientMaster.AngelCleintMaster/Clock': grpclib.const.Handler(
                self.Clock,
                grpclib.const.Cardinality.UNARY_UNARY,
                client_master_pb2.ClockReq,
                client_master_pb2.ClockResp,
            ),
            '/ClientMaster.AngelCleintMaster/GetClockMap': grpclib.const.Handler(
                self.GetClockMap,
                grpclib.const.Cardinality.UNARY_UNARY,
                client_master_pb2.GetClockMapReq,
                client_master_pb2.GetClockMapResp,
            ),
            '/ClientMaster.AngelCleintMaster/GetGlobalBatchSize': grpclib.const.Handler(
                self.GetGlobalBatchSize,
                grpclib.const.Cardinality.UNARY_UNARY,
                common_pb2.VoidReq,
                client_master_pb2.GetGlobalBatchResp,
            ),
            '/ClientMaster.AngelCleintMaster/CompleteTask': grpclib.const.Handler(
                self.CompleteTask,
                grpclib.const.Cardinality.UNARY_UNARY,
                common_pb2.CompleteTaskReq,
                common_pb2.VoidResp,
            ),
        }


class AngelCleintMasterStub:

    def __init__(self, channel: grpclib.client.Channel) -> None:
        self.RegisterWorker = grpclib.client.UnaryUnaryMethod(
            channel,
            '/ClientMaster.AngelCleintMaster/RegisterWorker',
            client_master_pb2.RegisterWorkerReq,
            client_master_pb2.RegisterWorkerResp,
        )
        self.RegisterTask = grpclib.client.UnaryUnaryMethod(
            channel,
            '/ClientMaster.AngelCleintMaster/RegisterTask',
            client_master_pb2.RegisterTaskReq,
            client_master_pb2.RegisterTaskResp,
        )
        self.SetAngelLocation = grpclib.client.UnaryUnaryMethod(
            channel,
            '/ClientMaster.AngelCleintMaster/SetAngelLocation',
            client_master_pb2.SetAngelLocationReq,
            common_pb2.VoidResp,
        )
        self.GetAngelLocation = grpclib.client.UnaryUnaryMethod(
            channel,
            '/ClientMaster.AngelCleintMaster/GetAngelLocation',
            common_pb2.VoidReq,
            client_master_pb2.GetAngelLocationResp,
        )
        self.HeartBeat = grpclib.client.UnaryUnaryMethod(
            channel,
            '/ClientMaster.AngelCleintMaster/HeartBeat',
            client_master_pb2.HeartBeatReq,
            client_master_pb2.HeartBeatResp,
        )
        self.Clock = grpclib.client.UnaryUnaryMethod(
            channel,
            '/ClientMaster.AngelCleintMaster/Clock',
            client_master_pb2.ClockReq,
            client_master_pb2.ClockResp,
        )
        self.GetClockMap = grpclib.client.UnaryUnaryMethod(
            channel,
            '/ClientMaster.AngelCleintMaster/GetClockMap',
            client_master_pb2.GetClockMapReq,
            client_master_pb2.GetClockMapResp,
        )
        self.GetGlobalBatchSize = grpclib.client.UnaryUnaryMethod(
            channel,
            '/ClientMaster.AngelCleintMaster/GetGlobalBatchSize',
            common_pb2.VoidReq,
            client_master_pb2.GetGlobalBatchResp,
        )
        self.CompleteTask = grpclib.client.UnaryUnaryMethod(
            channel,
            '/ClientMaster.AngelCleintMaster/CompleteTask',
            common_pb2.CompleteTaskReq,
            common_pb2.VoidResp,
        )
