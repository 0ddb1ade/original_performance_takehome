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
import os
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

    def emit(self, engine, slot, mem=None):
        """Queue a slot. `mem` is an optional region tag for memory ops;
        ops with different tags are assumed not to alias (regions here are
        the read-only forest and the per-vector value slices, which are all
        disjoint)."""
        self.ops.append((engine, slot, mem))

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

    def slot_accesses(self, engine, slot):
        """Return (reads, writes, mem_read, mem_write, barrier) for a slot."""
        vec = range(VLEN)
        match engine, slot:
            case "alu", (_, dest, a, b):
                return [a, b], [dest], False, False, False
            case "valu", ("vbroadcast", dest, src):
                return [src], [dest + i for i in vec], False, False, False
            case "valu", ("multiply_add", dest, a, b, c):
                reads = [x + i for x in (a, b, c) for i in vec]
                return reads, [dest + i for i in vec], False, False, False
            case "valu", (_, dest, a, b):
                reads = [x + i for x in (a, b) for i in vec]
                return reads, [dest + i for i in vec], False, False, False
            case "load", ("const", dest, _):
                return [], [dest], False, False, False
            case "load", ("load", dest, addr):
                return [addr], [dest], True, False, False
            case "load", ("load_offset", dest, addr, off):
                return [addr + off], [dest + off], True, False, False
            case "load", ("vload", dest, addr):
                return [addr], [dest + i for i in vec], True, False, False
            case "store", ("store", addr, src):
                return [addr, src], [], False, True, False
            case "store", ("vstore", addr, src):
                return [addr] + [src + i for i in vec], [], False, True, False
            case "flow", ("select", dest, cond, a, b):
                return [cond, a, b], [dest], False, False, False
            case "flow", ("vselect", dest, cond, a, b):
                reads = [x + i for x in (cond, a, b) for i in vec]
                return reads, [dest + i for i in vec], False, False, False
            case "flow", ("add_imm", dest, a, _):
                return [a], [dest], False, False, False
            case "flow", ("pause",):
                return [], [], False, False, True
            case "debug", ("compare", loc, _):
                return [loc], [], False, False, False
            case "debug", ("vcompare", loc, _):
                return [loc + i for i in vec], [], False, False, False
            case "debug", _:
                return [], [], False, False, False
            case _:
                raise NotImplementedError(f"Unknown slot {engine} {slot}")

    def lower(self, ops):
        """Greedy list scheduler: place each slot in the earliest cycle that
        respects dependencies and per-engine slot limits.

        Semantics of the machine: all reads in a cycle see the state from the
        end of the previous cycle, writes land at the end of the cycle. So a
        consumer must be placed strictly after its producer (RAW), a write
        strictly after a previous write (WAW), but a write may share a cycle
        with earlier program-order reads of the old value (WAR).
        """
        last_write = {}  # scratch addr -> cycle of last write
        last_read = {}  # scratch addr -> latest cycle of any read
        mem_last_load = {}  # region tag -> latest load cycle
        mem_last_store = {}  # region tag -> latest store cycle
        max_cycle = -1
        floor = 0  # barrier floor
        bundles = []

        def place(engine, slot, earliest):
            c = max(earliest, floor)
            while True:
                while c >= len(bundles):
                    bundles.append({})
                slots = bundles[c].setdefault(engine, [])
                if len(slots) < SLOT_LIMITS[engine]:
                    slots.append(slot)
                    return c
                c += 1

        for engine, slot, mem in ops:
            reads, writes, mem_read, mem_write, barrier = self.slot_accesses(
                engine, slot
            )
            if barrier:
                c = place(engine, slot, max_cycle + 1)
                floor = c + 1
                max_cycle = max(max_cycle, c)
                continue
            earliest = 0
            for a in reads:
                earliest = max(earliest, last_write.get(a, -1) + 1)
            for a in writes:
                earliest = max(
                    earliest, last_write.get(a, -1) + 1, last_read.get(a, -1)
                )
            if mem_read:
                earliest = max(earliest, mem_last_store.get(mem, -1) + 1)
            if mem_write:
                # Stores may share a cycle with loads (loads see old memory)
                # and with other stores (all our stores are to disjoint addrs).
                earliest = max(
                    earliest,
                    mem_last_load.get(mem, -1),
                    mem_last_store.get(mem, -1),
                )
            c = place(engine, slot, earliest)
            for a in reads:
                if last_read.get(a, -1) < c:
                    last_read[a] = c
            for a in writes:
                last_write[a] = c
            if mem_read:
                mem_last_load[mem] = max(mem_last_load.get(mem, -1), c)
            if mem_write:
                mem_last_store[mem] = max(mem_last_store.get(mem, -1), c)
            max_cycle = max(max_cycle, c)

        return [b for b in bundles if b]

    def emit_hash_v(self, val, t1, t2):
        """Vectorized myhash on the vector register `val` (in place).

        Stages of the form (a + C) + (a << k) collapse to a single
        multiply_add: a * (1 + 2**k) + C. The xor-based stages need the full
        three ops.
        """
        for op1, c1, op2, op3, c3 in HASH_STAGES:
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mult = self.vconst(1 + (1 << c3))
                self.emit("valu", ("multiply_add", val, val, mult, self.vconst(c1)))
            else:
                self.emit("valu", (op3, t1, val, self.vconst(c3)))
                self.emit("valu", (op1, t2, val, self.vconst(c1)))
                self.emit("valu", (op2, val, t1, t2))

    def emit_select_tree(self, d, bits, bcast, diff, free):
        """Compute the node value at depth d from the path bits, without
        touching memory: select among the 2**d possible nodes.

        Bottom-level selects between two known node values are a single
        multiply_add with precomputed (right-left) broadcast diffs; the
        merging selects between per-element vectors go to the flow engine
        (vselect), which is otherwise idle.
        """

        def rec(j0, j1, bi):
            if j1 - j0 == 2:
                t = free.pop()
                self.emit(
                    "valu",
                    ("multiply_add", t, bits[d - 1], diff[d][j0 // 2], bcast[d][j0]),
                )
                return t
            mid = (j0 + j1) // 2
            left = rec(j0, mid, bi + 1)
            right = rec(mid, j1, bi + 1)
            self.emit("flow", ("vselect", left, bits[bi], right, left))
            free.append(right)
            return left

        return rec(0, 2**d, 0)

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        debug: bool = False,
    ):
        """
        Vectorized implementation exploiting the static traversal structure.

        All indices start at 0 and the traversal descends one level per
        round; reaching the leaves the next index always wraps to the root
        (2*idx+1 >= n_nodes for every leaf). So the tree depth of every
        element is known statically per round:

            depth(r) = r % (forest_height + 1)

        Consequences: no index state, no wrap compare. For rounds at depth
        d <= SELECT_DEPTH the node value is computed from the path bits by
        a select tree over preloaded broadcast node values (no memory
        traffic); only deeper rounds gather node values with scalar loads.
        The gather address is reconstructed from the path bits once per
        descent and then updated incrementally.

        Values live in rotating per-vector register pools; memory is only
        touched for the initial vload, deep-round gathers and the final
        vstore. The memory layout from build_mem_image is statically known
        given the problem sizes, so there are no header loads.
        """
        assert batch_size % VLEN == 0
        assert n_nodes == 2 ** (forest_height + 1) - 1
        n_vec = batch_size // VLEN
        forest_p = 7
        indices_p = forest_p + n_nodes
        values_p = indices_p + batch_size
        period = forest_height + 1
        D = min(4, forest_height)  # max depth resolved by select trees

        vc_one = self.vconst(1)
        vc_two = self.vconst(2)
        for op1, c1, op2, op3, c3 in HASH_STAGES:
            if op1 == "+" and op2 == "+" and op3 == "<<":
                self.vconst(1 + (1 << c3))
                self.vconst(c1)
            else:
                self.vconst(c1)
                self.vconst(c3)
        # addr_next = 2*addr + bit + (1 - forest_p)
        vc_step = self.vconst(1 - forest_p)
        # addr at depth D+1 = horner(path bits) + 2^(D+1) - 1 + forest_p
        vc_horner = self.vconst(2 ** (D + 1) - 1 + forest_p)

        # Load the top D+1 levels of the tree and broadcast each node value.
        n_top = 2 ** (D + 1) - 1
        n_top_pad = (n_top + VLEN - 1) // VLEN * VLEN
        nodes_s = self.alloc_scratch("top_nodes", n_top_pad)
        for k in range(0, n_top_pad, VLEN):
            self.emit(
                "load",
                ("vload", nodes_s + k, self.scratch_const(forest_p + k)),
                mem="forest",
            )
        root_b = self.alloc_scratch("root_b", VLEN)
        self.emit("valu", ("vbroadcast", root_b, nodes_s))
        bcast = {}
        diff = {}
        for d in range(1, D + 1):
            lo = 2**d - 1
            bcast[d] = []
            for j in range(2**d):
                a = self.alloc_scratch(f"nb_{d}_{j}", VLEN)
                self.emit("valu", ("vbroadcast", a, nodes_s + lo + j))
                bcast[d].append(a)
            diff[d] = []
            for k in range(2 ** (d - 1)):
                a = self.alloc_scratch(f"nd_{d}_{k}", VLEN)
                self.emit("valu", ("-", a, bcast[d][2 * k + 1], bcast[d][2 * k]))
                diff[d].append(a)

        # Rotating register pools: each in-flight vector owns its values,
        # gather address, path bits and temporaries.
        N_POOLS = int(os.environ.get("KB_POOLS", "10"))
        pools = []
        for p in range(N_POOLS):
            pools.append(
                {
                    "val": self.alloc_scratch(f"val_{p}", VLEN),
                    "addr": self.alloc_scratch(f"addr_{p}", VLEN),
                    "bits": [
                        self.alloc_scratch(f"b{i}_{p}", VLEN) for i in range(D)
                    ],
                    "tmp": [self.alloc_scratch(f"m{i}_{p}", VLEN) for i in range(4)],
                }
            )

        val_addr_c = [self.scratch_const(values_p + v * VLEN) for v in range(n_vec)]

        # First pause: matches the first yield of reference_kernel2 (memory
        # is still unmodified at this point).
        self.emit("flow", ("pause",))

        for v in range(n_vec):
            P = pools[v % N_POOLS]
            val, addr, bits, tmp = P["val"], P["addr"], P["bits"], P["tmp"]
            self.emit("load", ("vload", val, val_addr_c[v]), mem=("values", v))
            for r in range(rounds):
                d = r % period
                free = list(tmp)
                if d == 0:
                    nv = root_b
                elif d <= D:
                    nv = self.emit_select_tree(d, bits, bcast, diff, free)
                else:
                    nv = free.pop()
                    for k in range(VLEN):
                        self.emit("load", ("load_offset", nv, addr, k), mem="forest")
                # val = myhash(val ^ node_val)
                self.emit("valu", ("^", val, val, nv))
                ht = [t for t in tmp if t != nv]
                self.emit_hash_v(val, ht[0], ht[1])
                if debug:
                    keys = tuple((r, v * VLEN + k, "hashed_val") for k in range(VLEN))
                    self.emit("debug", ("vcompare", val, keys))
                # Branch bit; not needed at the leaves (wrap to root) or on
                # the last round.
                if r + 1 < rounds and d != forest_height:
                    b = bits[d] if d < D else ht[0]
                    self.emit("valu", ("&", b, val, vc_one))
                    if d == D:
                        # Next round gathers: build the address from the path
                        # bits via Horner.
                        if D == 0:
                            self.emit("valu", ("+", addr, b, vc_horner))
                        else:
                            src = bits[0]
                            for x in bits[1:] + [b]:
                                self.emit(
                                    "valu", ("multiply_add", ht[1], src, vc_two, x)
                                )
                                src = ht[1]
                            self.emit("valu", ("+", addr, src, vc_horner))
                    elif d > D:
                        self.emit("valu", ("+", ht[1], b, vc_step))
                        self.emit("valu", ("multiply_add", addr, addr, vc_two, ht[1]))
            # Write final values back to memory (indices aren't checked).
            self.emit("store", ("vstore", val_addr_c[v], val), mem=("values", v))

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
