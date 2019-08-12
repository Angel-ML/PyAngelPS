import struct
import numpy as np
import scipy as sp

def read_int(buffer):
    return struct.unpack("<i", buffer)[0]


def write_int(value, buffer):
    buffer[:] = memoryview(struct.pack("<i", value)).cast("b")


def read_long(buffer):
    return struct.unpack("<q", buffer)[0]


def write_long(value, buffer):
    buffer[:] = memoryview(struct.pack("<q", value)).cast("b")


DataHeadLen = 128

_DTYPE_NP_TO_JVM = {
    np.dtype("int32"): 1,
    np.dtype("int64"): 2,
    np.dtype("float32"): 4,
    np.dtype("float64"): 8,
}

_DTYPE_JVM_TO_NP = {
    1: np.int32,
    2: np.int64,
    4: np.float32,
    8: np.float64
}


_DTYPE_NP_TO_PROTO = {
    np.int: 'int',
    np.dtype("int32"): 'int',
    np.long: 'long',
    np.dtype("int64"): 'long',
    np.float32: 'float',
    np.dtype("float32"): 'float',
    np.float: 'double',
    np.float64: 'double',
    np.dtype("float64"): 'double',

}


class DataHead:
    def __init__(self):
        self.sparse_dim = 0
        self.dense_dim = 0
        self.raw_shape = []
        self.nnz = 0
        self.raw_dtype = 0
        self.length = 0

        self.data = None
        self.buffer = None
        self.update = False

    @property
    def shape(self):
        return list(filter(lambda x: x != 0, self.raw_shape))

    @property
    def nbytes(self):
        return self.length + DataHeadLen

    def from_buffer(self, buffer, update):
        self.update = update
        offset = 0
        self.sparse_dim = read_int(buffer[offset:offset+4])
        offset += 4
        self.dense_dim = read_int(buffer[offset:offset+4])
        offset += 4
        self.raw_shape = []
        for i in range(8):
            self.raw_shape.append(read_long(buffer[offset:offset+8]))
            offset += 8
        self.nnz = read_int(buffer[offset:offset+4])
        offset += 4
        self.raw_dtype = read_int(buffer[offset:offset+4])
        offset += 4
        self.length = read_int(buffer[offset:offset+4])
        offset += 4

        self.buffer = buffer

    def from_data(self, data, update):
        self.update = update
        if isinstance(data, np.ndarray):
            if self.update is True and len(data.shape) == 1:
                data = data.reshape(1, *data.shape)
            self.sparse_dim = -1
            self.dense_dim = len(data.shape)
            self.raw_shape = list(data.shape)
            for i in range(len(data.shape), 8):
                self.raw_shape.append(0)
            self.nnz = data.size
            self.raw_dtype = _DTYPE_NP_TO_JVM[data.dtype]
            self.length = data.data.nbytes

            self.data = data

        elif isinstance(data, tuple):
            self.sparse_dim = 1
            indices, values, valid_idxs = data
            if len(indices.shape) == 1:
                if values.shape == (1,) and np.all(values == 1):
                    self.dense_dim = -1
                    self.raw_shape = list(valid_idxs)
                    self.length = indices.data.nbytes
                elif values.shape == (1,):
                    self.dense_dim = 0
                    self.raw_shape = list(valid_idxs)
                    self.length = indices.data.nbytes + values.data.nbytes
                else:
                    self.dense_dim = len(values.shape) - 1
                    self.raw_shape = list(valid_idxs)
                    self.raw_shape.extend(values.shape[1:])
                    self.length = indices.data.nbytes + values.data.nbytes
                for i in range(len(self.raw_shape), 8):
                    self.raw_shape.append(0)
                self.nnz = values.size
                self.raw_dtype = _DTYPE_NP_TO_JVM[values.dtype]
            else:
                self.dense_dim = 0
                self.sparse_dim = 2
                self.raw_shape = valid_idxs
                for i in range(len(self.raw_shape), 8):
                    self.raw_shape.append(0)
                self.length = indices.data + values.data.nbytes
                self.nnz = values.size
                self.raw_dtype = _DTYPE_NP_TO_JVM[values.dtype]
        self.data = data

    def write_to_buffer(self):
        buffer = self.buffer
        data = self.data
        offset = 0
        write_int(self.sparse_dim, buffer[offset:offset+4])
        offset += 4
        write_int(self.dense_dim, buffer[offset:offset+4])
        offset += 4
        for i in range(8):
            write_long(self.raw_shape[i], buffer[offset:offset+8])
            offset += 8
        write_int(self.nnz, buffer[offset:offset+4])
        offset += 4
        write_int(self.raw_dtype, buffer[offset:offset+4])
        offset += 4
        write_int(self.length, buffer[offset:offset+4])
        offset += 4
        if self.sparse_dim < 0:
            data_buffer = memoryview(data.data).cast("b", shape=(data.data.nbytes,))
            buffer[DataHeadLen:] = memoryview(data_buffer)
        elif self.sparse_dim == 1 and self.dense_dim == 0:
            indices = data[0]
            data_buffer = memoryview(indices.data).cast("b", shape=(indices.data.nbytes,))
            buffer[DataHeadLen:] = memoryview(data_buffer)
        elif self.sparse_dim == 1 and self.dense_dim >= 1:
            offset = DataHeadLen
            indices = data[0]
            data_buffer = memoryview(indices.data).cast("b", shape=(indices.data.nbytes,))
            buffer[offset:offset + self.nnz*8] = memoryview(data_buffer)
            offset += self.nnz * 8
            values = data[1]
            data_buffer = memoryview(values.data).cast("b", shape=(values.data.nbytes,))
            buffer[offset:] = memoryview(data_buffer)
        elif self.sparse_dim == 2 and self.dense_dim == 0:
            offset = DataHeadLen
            indices = data[0]
            data_buffer = memoryview(indices.data).cast("b", shape=(indices.data.nbytes,))
            buffer[offset:offset + self.nnz*8] = memoryview(data_buffer)
            offset += self.nnz * 8
            values = data[1]
            data_buffer = memoryview(values.data).cast("b", shape=(values.data.nbytes,))
            buffer[offset:] = memoryview(data_buffer)
        else:
            raise Exception('from wrong buffer')

    def parse_data(self, buffer=None):
        if buffer is None:
            buffer = self.buffer
        buffer = buffer[DataHeadLen:]
        if self.sparse_dim < 0:
            if self.update is True and self.shape[0] == 1:
                return np.frombuffer(buffer, dtype=_DTYPE_JVM_TO_NP[self.raw_dtype]).reshape(self.shape[1:])
            else:
                return np.frombuffer(buffer, dtype=_DTYPE_JVM_TO_NP[self.raw_dtype]).reshape(self.shape)
        elif self.sparse_dim == 1 and self.dense_dim == -1:
            indices = np.frombuffer(buffer, dtype=_DTYPE_JVM_TO_NP[self.raw_dtype]).reshape(self.nnz)
            values = np.ones((self.nnz,), dtype= _DTYPE_JVM_TO_NP[self.raw_dtype])
            return indices, values, self.shape
        elif self.sparse_dim == 1 and self.dense_dim == 0:
            offset = 0
            indices = np.frombuffer(buffer[offset:offset + self.nnz * 8], dtype=np.int64).reshape(self.nnz)
            offset += self.nnz * 8
            values = np.frombuffer(buffer[offset:], dtype=_DTYPE_JVM_TO_NP[self.raw_dtype]).reshape(self.nnz)
            return indices, values, self.shape
        elif self.sparse_dim == 1 and self.dense_dim >= 1:
            offset = 0
            indices = np.frombuffer(buffer[offset:offset + self.nnz * 8], dtype=np.int64).reshape(self.nnz)
            new_shape = (self.nnz, *(self.shape[1:]))
            values = np.frombuffer(buffer[offset:], dtype=_DTYPE_JVM_TO_NP[self.raw_dtype]).reshape(new_shape)
            return indices, values, self.shape[0]
        elif self.sparse_dim == 2 and self.dense_dim == 0:
            offset = 0
            row = np.frombuffer(buffer[offset:offset + self.nnz * 8], dtype=np.int64).reshape(self.nnz)
            offset += self.nnz * 8
            col = np.frombuffer(buffer[offset:offset + self.nnz * 8], dtype=np.int64).reshape(self.nnz)
            offset += self.nnz * 8
            values = np.frombuffer(buffer[offset:], dtype=_DTYPE_JVM_TO_NP[self.raw_dtype]).reshape(self.nnz)
            return np.array([row, col]), values, self.shape
        else:
            raise Exception('parse wrong data')







