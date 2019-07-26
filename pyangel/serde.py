import struct
import numpy as np
import scipy as sp

def read_int(buffer):
    return struct.unpack("<i", buffer)


def write_int(value, buffer):
    buffer = struct.pack("<i", value)


def read_long(buffer):
    return struct.unpack("<q", buffer)


def write_long(value, buffer):
    buffer = struct.pack("<q", value)


DataHeadLen = 128

_DTYPE_NP_TO_JVM = {
    np.int: 1,
    np.long: 2,
    np.float: 4,
    np.double: 8
}

_DTYPE_JVM_TO_NP = {
    1: np.int,
    2: np.long,
    4: np.float,
    8: np.double
}



class DataHead:
    def __init__(self):
        self.sparse_dim = 0
        self.dense_dim = 0
        self.shape = []
        self.nnz = 0
        self.dtype = 0
        self.length = 0

        self.nbytes = 0

    def from_buffer(self, buffer):
        offset = 0
        self.sparse_dim = read_int(buffer[offset:offset+4])
        offset += 4
        self.dense_dim = read_int(buffer[offset:offset+4])
        offset += 4
        self.shape = []
        for i in range(8):
            self.shape.append(read_long(buffer[offset:offset+8]))
            offset += 8
        self.nnz = read_int(buffer[offset:offset+4])
        offset += 4
        self.dtype = read_int(buffer[offset:offset+4])
        offset += 4
        self.length = read_int(buffer[offset:offset+4])
        offset += 4

        self.nbytes = len(buffer)

    def from_data(self, data):
        if isinstance(data, np.array):
            self.dense_dim = len(data.shape)
            self.shape = list(data.shape)
            self.nnz = data.size
            self.dtype = _DTYPE_NP_TO_JVM[data.dtype]
            self.length = data.size

            self.nbytes = data.nbytes + DataHeadLen
        elif isinstance(data, sp.sparse.coo_matrixx):
            pass

    def write_to_buffer(self, data, buffer):
        offset = 0
        write_int(self.sparse_dim, buffer[offset:offset+4])
        offset += 4
        write_int(self.dense_dim, buffer[offset:offset+4])
        offset += 4
        for i in range(8):
            write_long(self.shape[i], buffer[offset:offset+8])
            offset += 8
        write_int(self.nnz, buffer[offset:offset+4])
        offset += 4
        write_int(self.dtype, buffer[offset:offset+4])
        offset += 4
        write_int(self.length, buffer[offset:offset+4])
        offset += 4
        buffer[:DataHeadLen] = memoryview(data)

    def parse_data(self, buffer):
        buffer = buffer[DataHeadLen:]
        if self.sparse_dim == 0:
            return np.frombuffer(buffer, dtype=_DTYPE_JVM_TO_NP[self.dtype]).reshape(self.shape)


