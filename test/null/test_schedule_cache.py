import os, pathlib, subprocess, sys, tempfile, unittest
from unittest.mock import patch
from tinygrad import Tensor, Variable, Context
from tinygrad import dtypes
from tinygrad.codegen import _compact_disk_program, _disk_program_get, _disk_program_payload_safe, _disk_uop_fingerprint
from tinygrad.helpers import cpu_events
from tinygrad.schedule import _disk_schedule_get, schedule_cache
from tinygrad.uop.ops import CallInfo, KernelInfo, Ops, ProgramInfo, UOp

def schedule_one():
  Tensor([1]).schedule_linear()

class TestScheduleCache(unittest.TestCase):
  def test_bound_variable_var_vals(self):
    v = Variable('pos', 1, 100)
    x = Tensor.ones(10).contiguous().realize()

    t = x + Tensor(v.bind(42))
    _, var_vals = t.linear_with_vars()
    self.assertEqual(var_vals, {'pos': 42})

  def test_disable_schedule_cache(self):
    schedule_cache.clear()

    # test write
    with Context(SCACHE=0): schedule_one()
    self.assertEqual(len(schedule_cache), 0)
    with Context(SCACHE=1):
      schedule_one()
      schedule_one()
    self.assertEqual(len(schedule_cache), 1)

    # test read
    with Context(PROFILE=1):
      cpu_events.clear()
      with Context(SCACHE=0): schedule_one()
      num_events_no_cache = len(cpu_events)

      cpu_events.clear()
      with Context(SCACHE=1): schedule_one()
      num_events_cache = len(cpu_events)
    self.assertLess(num_events_cache, num_events_no_cache)

  def test_disk_schedule_and_program_cache_across_processes(self):
    root = pathlib.Path(__file__).parents[2]
    with tempfile.TemporaryDirectory() as tmp:
      env = {**os.environ, "CACHEDB": str(pathlib.Path(tmp) / "cache.db"), "CACHELEVEL": "3", "DEV": "PYTHON", "PYTHONPATH": str(root)}
      fill = """from tinygrad import Tensor
from examples.airbench_cifar10 import airbench_conv2d
Tensor.manual_seed(123)
assert Tensor([1, 2]).sum().item() == 3
x = Tensor.ones(1, 1, 4, 4).contiguous().realize().is_param_()
w = Tensor.ones(1, 1, 3, 3).contiguous().realize().is_param_()
airbench_conv2d(x, w).sum().backward()
x.grad.realize(w.grad)
"""
      hit = """import tinygrad.schedule as sched, tinygrad.codegen as codegen
def fail(*args, **kwargs): raise RuntimeError('persistent cache miss')
sched.create_schedule = codegen.do_to_program = fail
from tinygrad import Tensor
from examples.airbench_cifar10 import airbench_conv2d
Tensor.manual_seed(123)
assert Tensor([1, 2]).sum().item() == 3
x = Tensor.ones(1, 1, 4, 4).contiguous().realize().is_param_()
w = Tensor.ones(1, 1, 3, 3).contiguous().realize().is_param_()
airbench_conv2d(x, w).sum().backward()
x.grad.realize(w.grad)
"""
      subprocess.run([sys.executable, "-c", fill], cwd=root, env=env, check=True)
      subprocess.run([sys.executable, "-c", hit], cwd=root, env=env, check=True)

      inspect_cache = """import pickle, sqlite3
from tinygrad.helpers import CACHEDB
from tinygrad.codegen import _disk_cache_uop_safe
from tinygrad.uop.ops import UOp, Ops, buffers
conn = sqlite3.connect(CACHEDB)
tables = [x[0] for x in conn.execute("SELECT name FROM sqlite_master WHERE type='table'") if x[0].startswith(('schedule_', 'program_'))]
assert {x.split('_')[0] for x in tables} == {'schedule', 'program'}
for table in tables:
  for val, in conn.execute(f"SELECT val FROM '{table}'"):
    root = pickle.loads(val)
    assert isinstance(root, UOp)
    assert _disk_cache_uop_safe(root)
    assert not any(u in buffers for u in root.toposort())
    assert not any(u.op is Ops.CUSTOM_FUNCTION and u.arg in {'graph', 'hcq'} for u in root.toposort())
    if table.startswith('program_'): assert not any(u.op is Ops.BINARY for u in root.toposort())
"""
      subprocess.run([sys.executable, "-c", inspect_cache], cwd=root, env=env, check=True)

      miss = """import tinygrad.schedule as sched, tinygrad.codegen as codegen
def fail(*args, **kwargs): raise RuntimeError('persistent cache miss')
sched.create_schedule = codegen.do_to_program = fail
from tinygrad import Tensor
Tensor([1, 2]).prod().realize()
"""
      proc = subprocess.run([sys.executable, "-c", miss], cwd=root, env=env, text=True, capture_output=True)
      self.assertNotEqual(proc.returncode, 0)
      self.assertIn("persistent cache miss", proc.stderr)

  def test_disk_cache_keys_include_schedule_and_codegen_config(self):
    root = pathlib.Path(__file__).parents[2]
    with tempfile.TemporaryDirectory() as tmp:
      base_env = {**os.environ, "CACHEDB": str(pathlib.Path(tmp) / "cache.db"), "CACHELEVEL": "3", "DEV": "PYTHON", "PYTHONPATH": str(root)}
      codegen = """from tinygrad import Tensor
from tinygrad.codegen import to_program
from tinygrad.device import Device
x = Tensor.empty(64).contiguous().realize()
ast = (x + 1).schedule_linear().src[-1].src[0]
print(len(to_program(ast, Device['PYTHON'].renderer).src[1].src))
"""
      codegen_results = [subprocess.check_output([sys.executable, "-c", codegen], cwd=root, env={**base_env, "DMC": value}, text=True).strip()
                         for value in ("0", "1")]
      self.assertNotEqual(*codegen_results)

      schedule = """from tinygrad import Tensor
devices = tuple(f'PYTHON:{i}' for i in range(4))
print(len(Tensor.ones(1024).contiguous().shard(devices, axis=0).sum().schedule_linear().src))
"""
      schedule_results = [subprocess.check_output([sys.executable, "-c", schedule], cwd=root,
                          env={**base_env, "LATE_ALLREDUCE": value}, text=True).strip() for value in ("0", "1")]
      self.assertNotEqual(*schedule_results)

  def test_disk_cache_connection_reopens_after_pid_change(self):
    root = pathlib.Path(__file__).parents[2]
    with tempfile.TemporaryDirectory() as tmp:
      env = {**os.environ, "CACHEDB": str(pathlib.Path(tmp) / "cache.db"), "PYTHONPATH": str(root)}
      code = """import os
from unittest.mock import patch
import tinygrad.helpers as helpers
first = helpers.db_connection()
with patch('tinygrad.helpers.os.getpid', return_value=os.getpid() + 1):
  second = helpers.db_connection()
  assert second is not first
  helpers.diskcache_put('fork_test', 'child', 1)
assert helpers.diskcache_get('fork_test', 'child') == 1
"""
      subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True)

  def test_disk_uop_fingerprint_includes_tags_and_rejects_callbacks(self):
    a, b = UOp(Ops.CONST, dtypes.int, arg=1, tag="a"), UOp(Ops.CONST, dtypes.int, arg=1, tag="b")
    self.assertEqual(a.key, b.key)
    self.assertNotEqual(_disk_uop_fingerprint(a), _disk_uop_fingerprint(b))
    callback = UOp(Ops.CALL, src=(a,), arg=CallInfo(lambda: None, "opaque"))
    self.assertIsNone(_disk_uop_fingerprint(callback))
    custom = UOp(Ops.CUSTOM_FUNCTION, src=(a,), arg="third_party")
    self.assertIsNone(_disk_uop_fingerprint(custom))

  def test_disk_cache_rejects_invalid_and_assembler_payloads(self):
    sink = UOp(Ops.SINK, arg=KernelInfo(name="assembler"))
    assembler = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(UOp(Ops.INS, arg="instruction"),)),
                                            UOp(Ops.SOURCE, arg="instruction")), arg=ProgramInfo(name="assembler"))
    self.assertFalse(_disk_program_payload_safe(assembler))
    with patch("tinygrad.codegen.diskcache_get", return_value=assembler): self.assertIsNone(_disk_program_get("bad"))
    with patch("tinygrad.schedule.diskcache_get", return_value=UOp.const(dtypes.int, 1)): self.assertIsNone(_disk_schedule_get("bad"))

  def test_compact_disk_program_drops_lowered_graph(self):
    sink = UOp(Ops.SINK, src=(UOp.const(dtypes.int, 1),), arg=KernelInfo(name="compact"))
    prg = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(UOp.const(dtypes.int, 2),)), UOp(Ops.SOURCE, arg="source")),
              arg=ProgramInfo(name="compact"))
    compact = _compact_disk_program(prg)
    self.assertEqual(compact.src[0].src, ())
    self.assertEqual(compact.src[1].src, ())
    self.assertEqual(compact.src[2].arg, "source")
    self.assertTrue(_disk_program_payload_safe(compact))

  def test_partial_uop_cleanup(self):
    # Pickle can discard a UOp before __init__ has populated its fields.
    partial = object.__new__(UOp)
    partial.__del__()

if __name__ == "__main__":
  unittest.main()
