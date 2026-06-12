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
        """A vector of VLEN copies of a constant. When a staging slot is
        set, the scalar constant is bounced through it instead of keeping a
        dedicated scalar slot per value (saves scratch; the scheduler
        serializes the staging writes)."""
        val = val % (2**32)
        if val not in self.vconst_map:
            addr = self.alloc_scratch(name or f"vc_{val}", VLEN)
            staging = getattr(self, "vconst_staging", None)
            if staging is not None:
                self.emit("load", ("const", staging, val))
                self.emit("valu", ("vbroadcast", addr, staging))
            else:
                self.emit("valu", ("vbroadcast", addr, self.scratch_const(val)))
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
        """Lower the flat op list to instruction bundles."""
        sched = os.environ.get("KB_SCHED", "deadline")
        if sched == "greedy":
            return self.lower_greedy(ops)
        if sched == "prio":
            return self.lower_priority(ops)
        return self.lower_deadline(ops)

    def lower_greedy(self, ops):
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
                # Pauses share a bundle with regular work where possible: the
                # whole bundle executes, then the core pauses. Subsequent ops
                # may also share the pause cycle (they run in the same step).
                c = place(engine, slot, max(max_cycle, 0))
                floor = c
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

    def build_graph(self, ops):
        """Build the dependency graph over the flat op list.

        Hazards become edges: RAW/WAW latency 1 (consumer strictly after
        producer), WAR and store-store latency 0 (may share a cycle, since
        reads see start-of-cycle state). Pauses are zero-latency barriers
        against everything emitted before/after them. Returns
        (pred_count, succs) with succs[i] = [(consumer, latency), ...].
        """
        n = len(ops)
        accesses = [self.slot_accesses(e, s) for e, s, _ in ops]
        pred_count = [0] * n
        succs = [[] for _ in range(n)]

        last_write = {}  # addr -> writer op index
        readers = {}  # addr -> reader op indices since last write
        mem_store = {}  # region -> last store index
        mem_loads = {}  # region -> load indices since last store
        barrier_idx = None
        since_barrier = []

        for i, (engine, slot, mem) in enumerate(ops):
            reads, writes, mem_read, mem_write, barrier = accesses[i]
            preds = {}

            def addp(j, lat):
                if j is not None and preds.get(j, -1) < lat:
                    preds[j] = lat

            if barrier:
                for j in since_barrier:
                    addp(j, 0)
                addp(barrier_idx, 0)
                since_barrier = []
                barrier_idx = i
            else:
                addp(barrier_idx, 0)
                for a in reads:
                    addp(last_write.get(a), 1)
                for a in writes:
                    addp(last_write.get(a), 1)
                    for j in readers.get(a, ()):
                        addp(j, 0)
                if mem_read:
                    addp(mem_store.get(mem), 1)
                if mem_write:
                    addp(mem_store.get(mem), 0)
                    for j in mem_loads.get(mem, ()):
                        addp(j, 0)
                since_barrier.append(i)
                for a in reads:
                    readers.setdefault(a, []).append(i)
                for a in writes:
                    last_write[a] = i
                    readers[a] = []
                if mem_read:
                    mem_loads.setdefault(mem, []).append(i)
                if mem_write:
                    mem_store[mem] = i
                    mem_loads[mem] = []
            for j, lat in preds.items():
                succs[j].append((i, lat))
                pred_count[i] += 1

        return pred_count, succs

    def schedule_pass(self, ops, pred_count, succs, prio):
        """One resource-constrained list-scheduling pass.

        Cycle by cycle, issues the ready op with the smallest `prio` key per
        engine, subject to SLOT_LIMITS. Returns (cycle per op, makespan).
        Zero-latency successors freed by an op may issue in the same cycle.
        """
        from heapq import heappush, heappop

        n = len(ops)
        earliest = [0] * n
        pred_left = list(pred_count)
        avail = {eng: [] for eng in SLOT_LIMITS}
        future = defaultdict(list)
        for i in range(n):
            if pred_left[i] == 0:
                heappush(avail[ops[i][0]], (prio[i], i))

        cycles = [0] * n
        remaining = n
        c = 0
        while remaining:
            for i in future.pop(c, ()):
                heappush(avail[ops[i][0]], (prio[i], i))
            counts = dict.fromkeys(SLOT_LIMITS, 0)
            progress = True
            while progress:
                progress = False
                for eng, lim in SLOT_LIMITS.items():
                    heap = avail[eng]
                    while counts[eng] < lim and heap:
                        _, i = heappop(heap)
                        cycles[i] = c
                        counts[eng] += 1
                        remaining -= 1
                        progress = True
                        for j, lat in succs[i]:
                            if earliest[j] < c + lat:
                                earliest[j] = c + lat
                            pred_left[j] -= 1
                            if pred_left[j] == 0:
                                if earliest[j] <= c:
                                    heappush(avail[ops[j][0]], (prio[j], j))
                                else:
                                    future[earliest[j]].append(j)
            c += 1
        return cycles, c

    def to_bundles(self, ops, cycles, n_cycles):
        bundles = [{} for _ in range(n_cycles)]
        for i, (engine, slot, _) in enumerate(ops):
            bundles[cycles[i]].setdefault(engine, []).append(slot)
        return [b for b in bundles if b]

    def lower_priority(self, ops):
        """Critical-path list scheduler.

        Schedules cycle by cycle, always issuing the ready op with the
        greatest height (longest dependency chain below it, counting
        latency-1 edges). Pool-reuse WAR edges chain one vector's ops to the
        next user of its registers, so heights see through register reuse
        and the scheduler staggers the generations by true criticality.

        Raw height priority makes all in-flight chains advance in lockstep
        (equal heights), so their gather phases collide on the 2-slot load
        engine while select phases leave it idle. Quantizing the height
        keeps macro-criticality decisions while breaking ties by emission
        order, which staggers the chains' phases.
        """
        n = len(ops)
        pred_count, succs = self.build_graph(ops)

        height = [0] * n
        for i in range(n - 1, -1, -1):
            h = 0
            for j, lat in succs[i]:
                v = height[j] + lat
                if v > h:
                    h = v
            height[i] = h

        shift = int(os.environ.get("KB_PRIO_SHIFT", "8"))
        tie = os.environ.get("KB_TIE", "emit")
        if tie == "exact":
            prio = [(-(h >> shift), -h) for h in height]
        else:
            prio = [(-(h >> shift),) for h in height]

        cycles, n_cycles = self.schedule_pass(ops, pred_count, succs, prio)
        return self.to_bundles(ops, cycles, n_cycles)

    def lower_deadline(self, ops):
        """Globally deadline-aware scheduler.

        A backward pass schedules the *reversed* dependency graph under the
        same engine capacities (equivalent to placing every op as late as a
        resource-feasible schedule allows). That yields a deadline per op
        that accounts for both dependencies and downstream engine capacity:
        at most SLOT_LIMITS[e] ops of engine e share any deadline cycle, so
        deadlines are inherently phase-spread - the property raw
        critical-path heights lacked. The forward pass then issues by
        earliest deadline first. Optionally iterates, re-deriving deadlines
        from the previous forward schedule, keeping the best result.
        """
        n = len(ops)
        pred_count, succs = self.build_graph(ops)
        rsuccs = [[] for _ in range(n)]
        rpred = [0] * n
        for u in range(n):
            for v, lat in succs[u]:
                rsuccs[v].append((u, lat))
                rpred[u] += 1

        # Forward depth = critical path from the sources; the backward
        # pass's priority (it is the "height" of the reversed graph).
        depth = [0] * n
        for u in range(n):
            for v, lat in succs[u]:
                if depth[v] < depth[u] + lat:
                    depth[v] = depth[u] + lat

        shift = int(os.environ.get("KB_PRIO_SHIFT", "8"))
        iters = int(os.environ.get("KB_SCHED_ITERS", "8"))
        bprio = [(-(depth[i] >> shift), -i) for i in range(n)]

        best = None
        for _ in range(max(iters, 1)):
            bcycles, lb = self.schedule_pass(ops, rpred, rsuccs, bprio)
            deadline = [lb - 1 - bc for bc in bcycles]
            fprio = [(deadline[i],) for i in range(n)]
            fcycles, lf = self.schedule_pass(ops, pred_count, succs, fprio)
            if best is None or lf < best[1]:
                best = (fcycles, lf)
            # Next backward pass: latest-finishing-first in reverse time.
            bprio = [(-fcycles[i], -i) for i in range(n)]

        return self.to_bundles(ops, best[0], best[1])

    def vop(self, op, dest, a, b, scalar=lambda: False):
        """An elementwise vector op, either as one valu slot or as VLEN
        scalar alu slots (the alu engine has 12 slots/cycle and is otherwise
        idle, so spilling cheap vector ops there relieves the valu engine).
        `scalar` is a callable so the spill policy can rotate per op.
        """
        if scalar():
            for i in range(VLEN):
                self.emit("alu", (op, dest + i, a + i, b + i))
        else:
            self.emit("valu", (op, dest, a, b))

    def emit_hash_v(self, val, t1, t2, scalar=lambda: False, final_c1=None):
        """Vectorized myhash on the vector register `val` (in place).

        Stages of the form (a + C) + (a << k) collapse to a single
        multiply_add: a * (1 + 2**k) + C. The xor-based stages need the full
        three ops; those are cheap and may be spilled to the scalar alu.

        A multiply_add stage followed by (a + C') ^ (a << k') fuses further:
        both xor operands are linear in the multiply_add's input x
        (x*m + (C+C') and x*(m<<k') + (C<<k')), so the pair costs two
        multiply_adds and a xor instead of four ops, and the dependency
        chain shrinks by a cycle.

        final_c1 substitutes the last stage's xor constant; passing
        C_last ^ x fuses an extra `^ x` into the hash for free (the last
        stage is (a ^ C) ^ (a >> k)).
        """
        si = 0
        while si < len(HASH_STAGES):
            op1, c1, op2, op3, c3 = HASH_STAGES[si]
            if op1 == "+" and op2 == "+" and op3 == "<<":
                m = 1 + (1 << c3)
                if si + 1 < len(HASH_STAGES):
                    n1, nc1, n2, n3, nc3 = HASH_STAGES[si + 1]
                    if n1 == "+" and n2 == "^" and n3 == "<<":
                        sh = 1 << nc3
                        self.emit(
                            "valu",
                            ("multiply_add", t1, val, self.vconst(m),
                             self.vconst(c1 + nc1)),
                        )
                        self.emit(
                            "valu",
                            ("multiply_add", t2, val, self.vconst(m * sh),
                             self.vconst(c1 * sh)),
                        )
                        self.vop("^", val, t1, t2, scalar)
                        si += 2
                        continue
                self.emit(
                    "valu",
                    ("multiply_add", val, val, self.vconst(m), self.vconst(c1)),
                )
            else:
                c1v = self.vconst(c1)
                if final_c1 is not None and si == len(HASH_STAGES) - 1:
                    c1v = final_c1
                self.vop(op3, t1, val, self.vconst(c3), scalar)
                self.vop(op1, t2, val, c1v, scalar)
                self.vop(op2, val, t1, t2, scalar)
            si += 1

    def emit_select_tree(
        self, d, bits, bcast, diff, free, bottom_flow=False, split=False
    ):
        """Compute the node value at depth d from the path bits, without
        touching memory: select among the 2**d possible nodes.

        Bottom-level selects between two known node values are a single
        multiply_add with precomputed broadcast diffs; the merging selects
        between per-element vectors go to the flow engine (vselect), which
        is otherwise idle. With bottom_flow the bottom level also goes to
        the flow engine, trading valu for flow slots.

        Latency: the tree is keyed so the *newest* path bit (computed last
        round) decides the root select, while everything below depends only
        on older bits and can be scheduled ahead of time. The level at
        recursion depth k keys on bits[d-1-k]; the bottom level pairs nodes
        differing in the oldest bit (the j-MSB), i.e. p vs p + 2**(d-1).

        With split=True the root select is left to the caller: returns the
        (left, right) candidate registers instead (broadcasts for d == 1).
        """
        half = 2 ** (d - 1)

        def rec(p, k):
            if k == d - 1:
                t = free.pop()
                if bottom_flow:
                    self.emit(
                        "flow",
                        ("vselect", t, bits[0], bcast[d][p + half], bcast[d][p]),
                    )
                else:
                    self.emit(
                        "valu",
                        ("multiply_add", t, bits[0], diff[d][p], bcast[d][p]),
                    )
                return t
            left = rec(p, k + 1)
            right = rec(p + 2**k, k + 1)
            self.emit("flow", ("vselect", left, bits[d - 1 - k], right, left))
            free.append(right)
            return left

        if split:
            if d == 1:
                return bcast[d][0], bcast[d][1]
            return rec(0, 1), rec(1, 1)
        return rec(0, 0)

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
        # Max depth resolved by select trees; deeper rounds gather. Lowering
        # this trades valu select ops for load slots and frees the deepest
        # broadcast tables' scratch.
        D = min(int(os.environ.get("KB_SELECT_DEPTH", "4")), forest_height)

        # Tuning knobs (env-overridable for experiments; defaults are the
        # measured best).
        # Number of rotating register pools (in-flight vectors).
        N_POOLS = int(os.environ.get("KB_POOLS", "15"))
        # Select-tree bottom levels at these depths go to the flow engine
        # (e.g. "2" or "23"). Earlier all-depth variants regressed: flow
        # executes only 1 slot/cycle and deep rounds burst vselects.
        flow_depths = os.environ.get("KB_FLOW_DEPTHS", "")
        # Only every flow_bottom_mod-th vector uses flow for bottom selects,
        # keeping the 1-slot flow engine below its capacity.
        flow_bottom_mod = int(os.environ.get("KB_FLOW_BOTTOM_MOD", "2"))
        # Number of trailing vectors that get the latency-optimized
        # (pre-xored) select transitions.
        boost = int(os.environ.get("KB_BOOST", "0"))
        # Every g4_mod-th vector gathers at depth D instead of using the
        # deepest select tree (0 disables).
        g4_mod = int(os.environ.get("KB_G4_MOD", "0"))
        # Fraction (out of 32) of cheap vector ops that run as VLEN scalar
        # slots on the alu engine instead of one valu slot.
        alu_frac = int(os.environ.get("KB_ALU_FRAC", "11"))

        # First pause: matches the first yield of reference_kernel2. Memory
        # is first modified by the final vstores, so this can sit at cycle 0
        # and let the setup work overlap the main body.
        self.emit("flow", ("pause",))

        # Rotating register pools: each in-flight vector owns its values,
        # gather address, path bits and temporaries.
        pools = []
        for p in range(N_POOLS):
            bits = [self.alloc_scratch(f"b{i}_{p}", VLEN) for i in range(D)]
            # The gather address can alias bits[0]: addr is written by the
            # Horner at depth D (whose first op consumes bits[0]) and last
            # read at the bottom of the descent, before the next descent's
            # depth-0 round rewrites bits[0].
            addr = bits[0] if D >= 1 else self.alloc_scratch(f"addr_{p}", VLEN)
            pools.append(
                {
                    "val": self.alloc_scratch(f"val_{p}", VLEN),
                    "addr": addr,
                    "bits": bits,
                    "tmp": [self.alloc_scratch(f"m{i}_{p}", VLEN) for i in range(4)],
                }
            )

        # Load the top D+1 levels of the tree and broadcast each node value.
        # The staging buffer aliases pool 0's temps (the broadcasts all read
        # it before vector 0's first rounds write those temps).
        n_top = 2 ** (D + 1) - 1
        n_top_pad = (n_top + VLEN - 1) // VLEN * VLEN
        assert n_top_pad <= 4 * VLEN
        nodes_s = pools[0]["tmp"][0]
        # The last staging lane is unused by node values; scalar constants
        # for vector broadcasts bounce through it.
        if n_top < n_top_pad:
            self.vconst_staging = nodes_s + n_top_pad - 1

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
        # addr at depth Dv+1 = horner(path bits) + 2^(Dv+1) - 1 + forest_p,
        # for both per-vector select depths.
        vc_horner = {
            dv: self.vconst(2 ** (dv + 1) - 1 + forest_p)
            for dv in {D, max(D - 1, 0)}
        }

        for k in range(0, n_top_pad, VLEN):
            self.emit(
                "load",
                ("vload", nodes_s + k, self.scratch_const(forest_p + k)),
                mem="forest",
            )
        root_b = self.alloc_scratch("root_b", VLEN)
        self.emit("valu", ("vbroadcast", root_b, nodes_s))
        # When the descent wraps at the leaves, the next round xors with the
        # root; that xor fuses into the leaf round's final hash stage
        # ((a ^ C) ^ (a >> k) becomes (a ^ (C ^ root)) ^ (a >> k)).
        op1_l, c1_l, op2_l, _, _ = HASH_STAGES[-1]
        fused_root = None
        if op1_l == "^" and op2_l == "^" and rounds > period:
            fused_root = self.alloc_scratch("fused_root", VLEN)
            self.emit("valu", ("^", fused_root, self.vconst(c1_l), root_b))
        bcast = {}
        diff = {}
        # The bottom select level pairs node p with node p + 2^(d-1) (they
        # differ in the oldest path bit).
        for d in range(1, D + 1):
            lo = 2**d - 1
            half = 2 ** (d - 1)
            lower = []
            upper = []
            diff[d] = []
            for p in range(half):
                a = self.alloc_scratch(f"nb_{d}_{p}", VLEN)
                self.emit("valu", ("vbroadcast", a, nodes_s + lo + p))
                up = self.alloc_scratch(f"nb_{d}_{p + half}", VLEN)
                self.emit("valu", ("vbroadcast", up, nodes_s + lo + half + p))
                # Upper broadcasts survive where the flow-engine bottom
                # selects or the split (pre-xor) path read them directly.
                keep = str(d) in flow_depths or (d == 1 and boost > 0)
                if keep:
                    df = self.alloc_scratch(f"nd_{d}_{p}", VLEN)
                else:
                    df = up  # diff overwrites the upper broadcast
                self.emit("valu", ("-", df, up, a))
                lower.append(a)
                upper.append(up if keep else None)
                diff[d].append(df)
            bcast[d] = lower + upper

        # Per-vector value addresses are formed on the flow engine from one
        # base constant (add_imm into a temp lane), instead of 32 constants.
        values_base_c = self.scratch_const(values_p)

        def emit_vector_round(v, r):
            P = pools[v % N_POOLS]
            val, addr, bits, tmp = P["val"], P["addr"], P["bits"], P["tmp"]
            if r == 0:
                ad = tmp[3]  # lane 0, free until this round's temps are used
                self.emit("flow", ("add_imm", ad, values_base_c, v * VLEN))
                self.emit("load", ("vload", val, ad), mem=("values", v))
            d = r % period
            free = list(tmp)
            # Spill a deterministic spread of the cheap vector ops to the
            # scalar alu (fraction alu_frac/32, rotating per op).
            op_counter = [v * 7 + r * 5]

            def scalar():
                op_counter[0] += 11
                return op_counter[0] % 32 < alu_frac

            # Trailing (drain) vectors are latency-bound, not
            # throughput-bound: spend an extra xor to pre-compute both
            # root-select candidates so the round transition is
            # bit -> vselect (2 cycles) instead of bit -> select -> xor (3).
            boosted = v >= n_vec - boost
            # Per-vector select depth: every g4_mod-th vector gathers at
            # depth D instead of running the deepest select tree, shifting
            # valu work onto the load engine's slack.
            Dv = D - 1 if (g4_mod and v % g4_mod == 0 and D >= 1) else D
            if d == 0:
                # After a wrap the root xor was already fused into the
                # previous round's hash.
                nv = root_b if (r == 0 or fused_root is None) else None
            elif d <= Dv and boosted:
                left, right = self.emit_select_tree(
                    d, bits, bcast, diff, free, split=True
                )
                xl = free.pop()
                self.vop("^", xl, val, left, scalar)
                xr = free.pop()
                self.vop("^", xr, val, right, scalar)
                self.emit("flow", ("vselect", val, bits[d - 1], xr, xl))
                nv = None
            elif d <= Dv:
                bottom_flow = str(d) in flow_depths and v % flow_bottom_mod == 0
                nv = self.emit_select_tree(
                    d, bits, bcast, diff, free, bottom_flow=bottom_flow
                )
            else:
                nv = free.pop()
                for k in range(VLEN):
                    self.emit("load", ("load_offset", nv, addr, k), mem="forest")
            # val = myhash(val ^ node_val)
            if nv is not None:
                self.vop("^", val, val, nv, scalar)
            ht = [t for t in tmp if t != nv]
            fuse = fused_root is not None and d == forest_height and r + 1 < rounds
            self.emit_hash_v(
                val, ht[0], ht[1], scalar, final_c1=fused_root if fuse else None
            )
            if debug and not fuse:
                # (On fused rounds val carries the next round's root xor, so
                # it doesn't match the reference trace until the next round.)
                keys = tuple((r, v * VLEN + k, "hashed_val") for k in range(VLEN))
                self.emit("debug", ("vcompare", val, keys))
            # Branch bit; not needed at the leaves (wrap to root) or on
            # the last round.
            if r + 1 < rounds and d != forest_height:
                b = bits[d] if d < Dv else ht[0]
                self.vop("&", b, val, vc_one, scalar)
                # The next gather address is always arranged as
                # addr = (precomputable accumulator) + b, so the critical
                # path after the new bit is a single op.
                if d == Dv:
                    # Next round gathers: Horner over the old path bits
                    # lands in addr ahead of time, then addr += b.
                    if Dv == 0:
                        self.emit("valu", ("+", addr, b, vc_horner[Dv]))
                    else:
                        src = bits[0]
                        for x in bits[1:Dv]:
                            self.emit(
                                "valu", ("multiply_add", addr, src, vc_two, x)
                            )
                            src = addr
                        self.emit(
                            "valu", ("multiply_add", addr, src, vc_two, vc_horner[Dv])
                        )
                        self.vop("+", addr, addr, b, scalar)
                elif d > Dv:
                    # addr = (2*addr + step) + b; the multiply_add only
                    # needs last round's addr.
                    self.emit("valu", ("multiply_add", addr, addr, vc_two, vc_step))
                    self.vop("+", addr, addr, b, scalar)
            if r + 1 == rounds:
                # Write final values back to memory (indices aren't checked).
                ad = tmp[3] if tmp[3] != nv else tmp[2]
                self.emit("flow", ("add_imm", ad, values_base_c, v * VLEN))
                self.emit("store", ("vstore", ad, val), mem=("values", v))

        order = os.environ.get("KB_ORDER", "pool")
        if order == "rr":
            # Round-robin emission within each wave of in-flight vectors.
            for w0 in range(0, n_vec, N_POOLS):
                wave = range(w0, min(w0 + N_POOLS, n_vec))
                for r in range(rounds):
                    for v in wave:
                        emit_vector_round(v, r)
        elif order == "pool":
            # Pool-major: a pool's whole chain of vectors is emitted before
            # the next pool, so later generations on long-chain pools get
            # early-emission tie priority in the scheduler.
            for p in range(N_POOLS):
                for v in range(p, n_vec, N_POOLS):
                    for r in range(rounds):
                        emit_vector_round(v, r)
        else:
            for v in range(n_vec):
                for r in range(rounds):
                    emit_vector_round(v, r)

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
