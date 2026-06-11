"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.ops = []  # flat list of (engine, slot) in program order
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def emit(self, engine, slot):
        self.ops.append((engine, slot))

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        val = val % (2**32)
        if val not in self.const_map:
            addr = self.alloc_scratch(name or f"c_{val}")
            self.emit("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def vconst(self, val, name=None):
        """A vector of VLEN copies of a constant."""
        val = val % (2**32)
        if val not in self.vconst_map:
            src = self.scratch_const(val)
            addr = self.alloc_scratch(name or f"vc_{val}", VLEN)
            self.emit("valu", ("vbroadcast", addr, src))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def lower(self, ops):
        """Lower (engine, slot) ops to instruction bundles.

        Trivial version: one slot per instruction bundle.
        """
        return [{engine: [slot]} for engine, slot in ops]

    def emit_hash_v(self, val, t1, t2):
        """Vectorized myhash on the vector register `val` (in place)."""
        for op1, c1, op2, op3, c3 in HASH_STAGES:
            self.emit("valu", (op1, t1, val, self.vconst(c1)))
            self.emit("valu", (op3, t2, val, self.vconst(c3)))
            self.emit("valu", (op2, val, t1, t2))

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        debug: bool = False,
    ):
        """
        Vectorized implementation: the whole batch lives in scratch as
        batch_size//VLEN vector registers for values and indices; memory is
        only touched for the initial value load, the per-round node-value
        gathers, and the final value store.

        Memory layout (from build_mem_image) is statically known given the
        problem sizes, so header loads are unnecessary.
        """
        assert batch_size % VLEN == 0
        n_vec = batch_size // VLEN
        forest_p = 7
        indices_p = forest_p + n_nodes
        values_p = indices_p + batch_size

        vc_zero = self.vconst(0)
        vc_one = self.vconst(1)
        vc_two = self.vconst(2)
        vc_forest_p = self.vconst(forest_p)
        vc_n_nodes = self.vconst(n_nodes)
        # Materialize hash constants up front
        for _, c1, _, _, c3 in HASH_STAGES:
            self.vconst(c1)
            self.vconst(c3)

        zero_const = self.scratch_const(0)

        # Per-vector persistent state
        val = [self.alloc_scratch(f"val{v}", VLEN) for v in range(n_vec)]
        idx = [self.alloc_scratch(f"idx{v}", VLEN) for v in range(n_vec)]

        # Rotating temp pools so independent vectors don't share temps
        N_POOLS = 16
        pools = []
        for p in range(N_POOLS):
            pools.append(
                {
                    name: self.alloc_scratch(f"{name}_{p}", VLEN)
                    for name in ("vaddr", "nv", "t1", "t2", "b", "lt")
                }
            )

        # Initial state: values come from memory, indices are all zero
        val_addr_const = []
        for v in range(n_vec):
            val_addr_const.append(self.scratch_const(values_p + v * VLEN))
            self.emit("load", ("vload", val[v], val_addr_const[v]))
            self.emit("valu", ("vbroadcast", idx[v], zero_const))

        # First pause: matches the first yield of reference_kernel2 (memory
        # is still unmodified at this point).
        self.emit("flow", ("pause",))

        for r in range(rounds):
            for v in range(n_vec):
                P = pools[v % N_POOLS]
                vaddr, nv, t1, t2, b, lt = (
                    P["vaddr"], P["nv"], P["t1"], P["t2"], P["b"], P["lt"],
                )
                # node_val = mem[forest_values_p + idx]  (gather)
                self.emit("valu", ("+", vaddr, idx[v], vc_forest_p))
                for k in range(VLEN):
                    self.emit("load", ("load_offset", nv, vaddr, k))
                # val = myhash(val ^ node_val)
                self.emit("valu", ("^", val[v], val[v], nv))
                self.emit_hash_v(val[v], t1, t2)
                if debug:
                    keys = tuple((r, v * VLEN + k, "hashed_val") for k in range(VLEN))
                    self.emit("debug", ("vcompare", val[v], keys))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                self.emit("valu", ("%", t1, val[v], vc_two))
                self.emit("valu", ("==", t1, t1, vc_zero))
                self.emit("flow", ("vselect", b, t1, vc_one, vc_two))
                self.emit("valu", ("*", idx[v], idx[v], vc_two))
                self.emit("valu", ("+", idx[v], idx[v], b))
                # idx = 0 if idx >= n_nodes else idx
                self.emit("valu", ("<", lt, idx[v], vc_n_nodes))
                self.emit("flow", ("vselect", idx[v], lt, idx[v], vc_zero))
                if debug:
                    keys = tuple((r, v * VLEN + k, "wrapped_idx") for k in range(VLEN))
                    self.emit("debug", ("vcompare", idx[v], keys))

        # Write final values back to memory (indices aren't checked).
        for v in range(n_vec):
            self.emit("store", ("vstore", val_addr_const[v], val[v]))

        # Final pause: matches the last yield of reference_kernel2.
        self.emit("flow", ("pause",))

        self.instrs = self.lower(self.ops)


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
    debug: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds, debug=debug)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_debug(self):
        # Checks intermediate values against the reference trace
        do_kernel_test(10, 16, 256, debug=True)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
