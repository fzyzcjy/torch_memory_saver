"""Two-node numeric reproduction of NCCL graph replay across TMS remapping."""

import argparse
import ctypes
import datetime
import importlib.metadata
import json
import os
import struct
import time

import torch
import torch.distributed as dist
from torch_memory_saver import torch_memory_saver as tms


def emit(event, **fields):
    print(json.dumps(dict(event=event, time=time.time(), rank=args.rank, **fields)), flush=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--rank', type=int, required=True)
parser.add_argument('--world-size', type=int, default=2)
parser.add_argument('--master-addr', required=True)
parser.add_argument('--master-port', type=int, required=True)
parser.add_argument('--mode', choices=['remap', 'keep', 'stable', 'copy-only'], default='remap')
parser.add_argument('--elements', type=int, default=1048576)
parser.add_argument('--cycles', type=int, default=3)
parser.add_argument('--observe-old-backing', action='store_true')
args = parser.parse_args()
if args.observe_old_backing and args.mode != 'remap':
    parser.error('--observe-old-backing requires --mode remap')
torch.cuda.set_device(0)
store = dist.TCPStore(args.master_addr, args.master_port, args.world_size,
                      args.rank == 0, timeout=datetime.timedelta(seconds=90))


def barrier(label):
    store.set(f'{label}/{args.rank}', b'1')
    store.wait([f'{label}/{rank}' for rank in range(args.world_size)])


lib_path = importlib.metadata.distribution('nvidia-nccl-cu13').locate_file('nvidia/nccl/lib/libnccl.so.2')
nccl = ctypes.CDLL(str(lib_path))


class UniqueId(ctypes.Structure):
    _fields_ = [('data', ctypes.c_char * 128)]


for name, signature in {
    'ncclGetUniqueId': [ctypes.POINTER(UniqueId)],
    'ncclCommInitRank': [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UniqueId, ctypes.c_int],
    'ncclAllGather': [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p],
    'ncclCommDestroy': [ctypes.c_void_p],
}.items():
    func = getattr(nccl, name)
    func.argtypes = signature
    func.restype = ctypes.c_int
nccl.ncclGetErrorString.argtypes = [ctypes.c_int]
nccl.ncclGetErrorString.restype = ctypes.c_char_p


def checked(code):
    if code:
        raise RuntimeError(nccl.ncclGetErrorString(code).decode())


uid = UniqueId()
if args.rank == 0:
    checked(nccl.ncclGetUniqueId(ctypes.byref(uid)))
    store.set('nccl-id', bytes(uid))
else:
    uid = UniqueId.from_buffer_copy(store.get('nccl-id'))
comm = ctypes.c_void_p()
checked(nccl.ncclCommInitRank(ctypes.byref(comm), args.world_size, uid, args.rank))
stream = torch.cuda.Stream()


def gather(x, out):
    checked(nccl.ncclAllGather(x.data_ptr(), out.data_ptr(), x.numel(), 7, comm,
                              torch.cuda.current_stream().cuda_stream))


driver = ctypes.CDLL('libcuda.so.1')
driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
driver.cuPointerGetAttribute.restype = ctypes.c_int


def mapping(tensor):
    buffer_id = ctypes.c_uint64()
    result = driver.cuPointerGetAttribute(ctypes.byref(buffer_id), 7, tensor.data_ptr())
    return dict(ptr=hex(tensor.data_ptr()), bytes=tensor.numel() * tensor.element_size(),
                buffer_id=buffer_id.value, buffer_id_result=result)


class MemoryLocation(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('id', ctypes.c_int)]


class MemoryAccess(ctypes.Structure):
    _fields_ = [('location', MemoryLocation), ('flags', ctypes.c_ulonglong)]


class OldBackingAlias:
    """Keep the original physical allocation observable at a separate address."""

    def __init__(self, x, out):
        for name, signature in {
            'cuMemRetainAllocationHandle': [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p],
            'cuMemAddressReserve': [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint64],
            'cuMemMap': [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint64],
            'cuMemSetAccess': [ctypes.c_uint64, ctypes.c_size_t, ctypes.POINTER(MemoryAccess), ctypes.c_size_t],
            'cuMemsetD32_v2': [ctypes.c_uint64, ctypes.c_uint, ctypes.c_size_t],
            'cuMemcpyDtoH_v2': [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
            'cuMemUnmap': [ctypes.c_uint64, ctypes.c_size_t],
            'cuMemAddressFree': [ctypes.c_uint64, ctypes.c_size_t],
            'cuMemRelease': [ctypes.c_uint64],
        }.items():
            func = getattr(driver, name)
            func.argtypes = signature
            func.restype = ctypes.c_int
        base = ctypes.c_uint64()
        size = ctypes.c_size_t()
        self.check(driver.cuPointerGetAttribute(ctypes.byref(base), 11, x.data_ptr()))
        self.check(driver.cuPointerGetAttribute(ctypes.byref(size), 12, x.data_ptr()))
        assert base.value <= out.data_ptr() < out.data_ptr() + out.numel() * 4 <= base.value + size.value
        self.handle = ctypes.c_uint64()
        self.ptr = ctypes.c_uint64()
        self.size = size.value
        self.check(driver.cuMemRetainAllocationHandle(ctypes.byref(self.handle), x.data_ptr()))
        self.check(driver.cuMemAddressReserve(ctypes.byref(self.ptr), self.size, 0, 0, 0))
        self.check(driver.cuMemMap(self.ptr.value, self.size, 0, self.handle.value, 0))
        access = MemoryAccess(MemoryLocation(1, torch.cuda.current_device()), 3)
        self.check(driver.cuMemSetAccess(self.ptr.value, self.size, ctypes.byref(access), 1))
        self.x_ptr = self.ptr.value + x.data_ptr() - base.value
        self.out_ptr = self.ptr.value + out.data_ptr() - base.value
        self.count = x.numel()
        buffer_id = ctypes.c_uint64()
        self.check(driver.cuPointerGetAttribute(ctypes.byref(buffer_id), 7, self.ptr.value))
        emit('old_backing_alias', ptr=hex(self.ptr.value), size=self.size,
             original_ptr=hex(base.value), buffer_id=buffer_id.value)

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'CUDA Driver API error {code} in old-backing probe')

    def fill(self, cycle):
        self.sentinel = 10000 + cycle * 10 + args.rank + 1
        for ptr, value, count in [(self.x_ptr, self.sentinel, self.count),
                                  (self.out_ptr, -555, self.count * args.world_size)]:
            bits = struct.unpack('I', struct.pack('f', value))[0]
            self.check(driver.cuMemsetD32_v2(ptr, bits, count))

    def inspect(self, cycle):
        data = ctypes.create_string_buffer(self.count * args.world_size * 4)
        self.check(driver.cuMemcpyDtoH_v2(data, self.out_ptr, len(data)))
        actual = torch.frombuffer(data, dtype=torch.float32).view(args.world_size, self.count)
        expected = torch.arange(1, args.world_size + 1, dtype=torch.float32) + 10000 + cycle * 10
        expected[args.rank] = -555
        mismatches = (actual != expected[:, None]).sum().item()
        emit('old_backing_observation', cycle=cycle, mismatches=mismatches,
             first_values=actual[:, :4].tolist(), expected=expected.tolist())

    def close(self):
        self.check(driver.cuMemUnmap(self.ptr.value, self.size))
        self.check(driver.cuMemAddressFree(self.ptr.value, self.size))
        self.check(driver.cuMemRelease(self.handle.value))


emit('environment', torch=torch.__version__, nccl=torch.cuda.nccl.version(),
     tms_module=__import__('torch_memory_saver').__file__, preload=os.environ.get('LD_PRELOAD'),
     nccl_library=str(lib_path), mode=args.mode, elements=args.elements,
     nccl_env={k: v for k, v in os.environ.items() if k.startswith('NCCL_')})
with torch.cuda.stream(stream):
    warm_send = torch.ones(args.elements, device='cuda', dtype=torch.float32)
    warm_recv = torch.empty(args.elements * args.world_size, device='cuda')
    for _ in range(3):
        gather(warm_send, warm_recv)
stream.synchronize()
barrier('warmup')

with tms.region(tag='payload', enable_cpu_backup=True, cpu_backup_backend='pinned'):
    x = torch.full((args.elements,), args.rank + 1, device='cuda', dtype=torch.float32)
    out = torch.full((args.world_size * args.elements,), -999., device='cuda')
if args.mode == 'stable':
    with tms.region(tag='resident', enable_cpu_backup=True, cpu_backup_backend='pinned'):
        stable_x = torch.empty_like(x)
        stable_out = torch.empty_like(out)

torch.cuda.synchronize()
emit('before_capture', x=mapping(x), out=mapping(out))
if args.mode == 'stable':
    emit('resident_buffers', x=mapping(stable_x), out=mapping(stable_out))
barrier('capture')
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    if args.mode == 'stable':
        stable_x.copy_(x)
        gather(stable_x, stable_out)
        out.copy_(stable_out)
    elif args.mode == 'copy-only':
        for rank in range(args.world_size):
            out[rank * args.elements:(rank + 1) * args.elements].copy_(x)
    else:
        gather(x, out)


old_alias = None


def replay_check(cycle):
    if old_alias is not None:
        old_alias.fill(cycle)
    x.fill_(100 * cycle + args.rank + 1)
    out.fill_(-999)
    torch.cuda.synchronize()
    barrier(f'replay-{cycle}')
    emit('replay_start', cycle=cycle)
    graph.replay()
    torch.cuda.synchronize()
    if old_alias is not None:
        barrier(f'alias-read-{cycle}')
        old_alias.inspect(cycle)
    actual = out.cpu().view(args.world_size, args.elements)
    expected = torch.arange(1, args.world_size + 1, dtype=torch.float32) + 100 * cycle
    if args.mode == 'copy-only':
        expected.fill_(100 * cycle + args.rank + 1)
    mismatches = (actual != expected[:, None]).sum().item()
    emit('replay_result', cycle=cycle, mismatches=mismatches,
         first_values=actual[:, :4].tolist(), expected=expected.tolist())
    return mismatches == 0


passed = replay_check(0)
if args.observe_old_backing:
    old_alias = OldBackingAlias(x, out)
for cycle in range(1, args.cycles + 1):
    torch.cuda.synchronize()
    barrier(f'pause-{cycle}')
    before_x, before_out = mapping(x), mapping(out)
    if args.mode != 'keep':
        tms.pause(tag='payload')
        emit('paused', cycle=cycle)
        tms.resume(tag='payload')
        torch.cuda.synchronize()
    preserved = bool((x.cpu() == 100 * (cycle - 1) + args.rank + 1).all())
    emit('resumed', cycle=cycle, x_before=before_x, x_after=mapping(x),
         out_before=before_out, out_after=mapping(out), input_preserved=preserved)
    passed = replay_check(cycle) and preserved and passed
    if args.mode == 'remap':
        barrier(f'eager-{cycle}')
        out.fill_(-777)
        gather(x, out)
        torch.cuda.synchronize()
        expected = torch.arange(1, args.world_size + 1, dtype=torch.float32) + 100 * cycle
        mismatches = (out.cpu().view(args.world_size, args.elements) != expected[:, None]).sum().item()
        emit('eager_after_resume', cycle=cycle, mismatches=mismatches)
        passed = passed and mismatches == 0

barrier('finished')
graph.reset()
checked(nccl.ncclCommDestroy(comm))
if old_alias is not None:
    old_alias.close()
emit('complete', passed=passed)
raise SystemExit(0 if passed else 1)
