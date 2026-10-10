import pytest

from compiler.graph import Node
from compiler.memory import MemoryConfig, plan_memory
from python.edge_npu.tensor import TensorSpec


def _chain():
    order = [Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="bias_add", inputs=("r", "b"), output="d"),
             Node(op="requantize", inputs=("d",), output="q")]
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "r": TensorSpec(name="r", dtype="int32", shape=(8, 8)),
             "b": TensorSpec(name="b", dtype="int32", shape=(8,)),
             "d": TensorSpec(name="d", dtype="int32", shape=(8, 8)),
             "q": TensorSpec(name="q", dtype="int8", shape=(8, 8), scale=0.02)}
    aliases = {"r": "c", "d": "r"}
    return order, specs, aliases


def test_plan_worked_example_slots_and_peak():
    order, specs, aliases = _chain()
    plan = plan_memory(order, specs, aliases, MemoryConfig())
    got = {n: (slot.address, slot.size) for n, slot in plan.slots.items()}
    assert got == {"a": (0x1000, 64), "w": (0x1040, 64), "c": (0x1080, 256),
                   "r": (0x1080, 256), "b": (0x1000, 32), "d": (0x1080, 256),
                   "q": (0x1000, 64)}
    assert plan.peak_bytes == 0x180


def test_plan_alias_keeps_slot_alive_until_last_tenant():
    order, specs, aliases = _chain()
    plan = plan_memory(order, specs, aliases, MemoryConfig())
    assert plan.slots["r"] == plan.slots["c"] == plan.slots["d"]
    # q is born while d (a tenant of the c slot) is still live, so the c slot
    # is not reusable; b's freed block wins first-fit by address.
    assert plan.slots["q"].address == 0x1000


def test_plan_budget_exact_then_one_byte_short():
    order, specs, aliases = _chain()
    plan_memory(order, specs, aliases, MemoryConfig(size=0x180))  # exactly fits
    with pytest.raises(ValueError, match="384"):
        plan_memory(order, specs, aliases, MemoryConfig(size=0x17F))


def test_plan_aligns_every_address():
    order = [Node(op="matmul", inputs=("a", "w"), output="c")]
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(10, 10)),
             "w": TensorSpec(name="w", dtype="int8", shape=(10, 10)),
             "c": TensorSpec(name="c", dtype="int32", shape=(10, 10))}
    plan = plan_memory(order, specs, {}, MemoryConfig())
    assert plan.slots["a"].address == 0x1000
    assert plan.slots["w"].address == 0x1080
    assert plan.slots["c"].address == 0x1100
    assert all(s.address % 64 == 0 for s in plan.slots.values())


def test_plan_skips_unreferenced_tensors():
    order, specs, aliases = _chain()
    specs = dict(specs, ghost=TensorSpec(name="ghost", dtype="int8", shape=(8,)))
    plan = plan_memory(order, specs, aliases, MemoryConfig())
    assert "ghost" not in plan.slots


def test_plan_deterministic():
    order, specs, aliases = _chain()
    first = plan_memory(order, specs, aliases, MemoryConfig())
    second = plan_memory(order, specs, aliases, MemoryConfig())
    assert first == second


def test_config_validation():
    with pytest.raises(ValueError):
        MemoryConfig(align=0)
    with pytest.raises(ValueError):
        MemoryConfig(base=0x1001, align=64)
    with pytest.raises(ValueError):
        MemoryConfig(size=0)
