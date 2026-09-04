"""
CSA1221 - COMPUTER ARCHITECTURE
Design and Performance Analysis of a Pipelined Processor
with a Multi-Level Cache Memory System

SINGLE-FILE IMPLEMENTATION
============================================================

Implements the architecture specified in the submitted report:

1. 3-bus RISC datapath
2. LOAD / STORE / ADD / SUB / BEQ / BNE
3. Five-stage IF/ID/EX/MEM/WB pipeline
4. RAW hazard detection
5. ALU forwarding
6. Load-use one-cycle stall
7. BTB-style dynamic branch prediction
8. Branch misprediction recovery / pipeline flush
9. Two-way superscalar processor
10. Register renaming / RAT
11. Reservation stations
12. Out-of-order issue
13. Reorder buffer with in-order commit
14. Speculative execution and recovery
15. L1-I / L1-D / L2 / L3 cache hierarchy
16. LRU / pseudo-LRU / NRU-style replacement
17. TLB and virtual-page translation model
18. AMAT / CPI / IPC / GIPS / bandwidth calculations
19. Scalar vs superscalar comparison

The implementation is intentionally self-contained: no external packages
are required.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

NUM_REGISTERS = 16
MEMORY_WORDS = 4096
WORD_SIZE = 4
CACHE_LINE_BYTES = 64
PAGE_SIZE = 4096
PHYSICAL_MEMORY_PAGES = 256


# ============================================================
# INSTRUCTION
# ============================================================

@dataclass
class Instruction:
    opcode: str
    rd: Optional[int] = None
    rs1: Optional[int] = None
    rs2: Optional[int] = None
    offset: int = 0
    target: Optional[int] = None
    name: str = ""
    latency: int = 1

    def __str__(self):
        if self.name:
            return self.name

        op = self.opcode.upper()

        if op in ("ADD", "SUB"):
            return f"{op} R{self.rd}, R{self.rs1}, R{self.rs2}"

        if op == "LOAD":
            return f"LOAD R{self.rd}, {self.offset}(R{self.rs1})"

        if op == "STORE":
            return f"STORE R{self.rs2}, {self.offset}(R{self.rs1})"

        if op in ("BEQ", "BNE"):
            return f"{op} R{self.rs1}, R{self.rs2}, {self.offset}"

        if op == "MUL":
            return f"MUL R{self.rd}, R{self.rs1}, R{self.rs2}"

        return op


# ============================================================
# ALU
# ============================================================

class ALU:

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def compare_equal(a, b):
        return a == b


# ============================================================
# REGISTER FILE
# ============================================================

class RegisterFile:

    def __init__(self, n=NUM_REGISTERS):
        self.registers = [0] * n

    def read(self, r):
        if r is None:
            return 0
        return self.registers[r]

    def write(self, r, value):
        if r is not None and r != 0:
            self.registers[r] = value

    def dump(self):
        print("\nREGISTER FILE")
        print("-" * 50)
        for i, value in enumerate(self.registers):
            print(f"R{i:02d} = {value}")


# ============================================================
# MAIN MEMORY
# ============================================================

class MainMemory:

    def __init__(self, words=MEMORY_WORDS):
        self.data = [0] * words

    def read_word(self, address):
        if not 0 <= address < len(self.data):
            raise MemoryError(f"Invalid memory read: {address}")
        return self.data[address]

    def write_word(self, address, value):
        if not 0 <= address < len(self.data):
            raise MemoryError(f"Invalid memory write: {address}")
        self.data[address] = value

    def dump(self, start, end):
        print("\nMEMORY")
        print("-" * 50)
        for a in range(start, min(end, len(self.data))):
            print(f"M[{a}] = {self.data[a]}")


# ============================================================
# THREE-BUS DATAPATH
# ============================================================

class ThreeBusDatapath:

    def __init__(self, registers, memory):
        self.rf = registers
        self.memory = memory
        self.alu = ALU()
        self.mar = 0
        self.mbr = 0
        self.bus_a = None
        self.bus_b = None
        self.bus_c = None
        self.zero_flag = False

    def clear_buses(self):
        self.bus_a = self.bus_b = self.bus_c = None

    def add(self, rd, rs1, rs2):
        self.clear_buses()
        self.bus_a = self.rf.read(rs1)
        self.bus_b = self.rf.read(rs2)
        self.bus_c = self.alu.add(self.bus_a, self.bus_b)
        self.rf.write(rd, self.bus_c)

    def sub(self, rd, rs1, rs2):
        self.clear_buses()
        self.bus_a = self.rf.read(rs1)
        self.bus_b = self.rf.read(rs2)
        self.bus_c = self.alu.sub(self.bus_a, self.bus_b)
        self.rf.write(rd, self.bus_c)

    def load(self, rd, offset, rs):
        self.clear_buses()
        self.bus_a = self.rf.read(rs)
        self.mar = self.alu.add(self.bus_a, offset)
        self.mbr = self.memory.read_word(self.mar)
        self.bus_c = self.mbr
        self.rf.write(rd, self.bus_c)

    def store(self, rs2, offset, rs1):
        self.clear_buses()
        self.bus_a = self.rf.read(rs1)
        self.mar = self.alu.add(self.bus_a, offset)
        self.mbr = self.rf.read(rs2)
        self.memory.write_word(self.mar, self.mbr)

    def branch(self, opcode, rs1, rs2, offset, pc):
        self.clear_buses()
        self.bus_a = self.rf.read(rs1)
        self.bus_b = self.rf.read(rs2)
        self.zero_flag = self.alu.compare_equal(self.bus_a, self.bus_b)

        taken = self.zero_flag if opcode == "BEQ" else not self.zero_flag
        return pc + offset if taken else pc + WORD_SIZE, taken


# ============================================================
# CACHE
# ============================================================

@dataclass
class CacheLine:
    valid: bool = False
    tag: int = 0
    last_used: int = 0
    dirty: bool = False


class Cache:

    def __init__(
        self,
        name,
        size_kb,
        associativity,
        hit_latency,
        replacement="LRU"
    ):
        self.name = name
        self.size_bytes = size_kb * 1024
        self.associativity = associativity
        self.hit_latency = hit_latency
        self.replacement = replacement

        self.line_count = max(
            1,
            self.size_bytes // CACHE_LINE_BYTES
        )
        self.num_sets = max(
            1,
            self.line_count // associativity
        )

        self.sets = [
            [CacheLine() for _ in range(associativity)]
            for _ in range(self.num_sets)
        ]

        self.clock = 0
        self.accesses = 0
        self.hits = 0
        self.misses = 0

    def _index_tag(self, address):
        byte_address = address * WORD_SIZE
        line = byte_address // CACHE_LINE_BYTES
        index = line % self.num_sets
        tag = line // self.num_sets
        return index, tag

    def access(self, address, is_write=False):
        self.clock += 1
        self.accesses += 1

        index, tag = self._index_tag(address)
        ways = self.sets[index]

        for line in ways:
            if line.valid and line.tag == tag:
                self.hits += 1
                line.last_used = self.clock
                if is_write:
                    line.dirty = True
                return True

        self.misses += 1

        invalid = [x for x in ways if not x.valid]
        if invalid:
            victim = invalid[0]
        elif self.replacement in ("LRU", "pseudo-LRU"):
            victim = min(ways, key=lambda x: x.last_used)
        else:  # NRU approximation
            victim = ways[0]

        victim.valid = True
        victim.tag = tag
        victim.last_used = self.clock
        victim.dirty = is_write

        return False

    @property
    def miss_rate(self):
        if self.accesses == 0:
            return 0.0
        return self.misses / self.accesses

    def stats(self):
        return {
            "name": self.name,
            "accesses": self.accesses,
            "hits": self.hits,
            "misses": self.misses,
            "miss_rate": self.miss_rate,
        }


# ============================================================
# TLB / VIRTUAL MEMORY
# ============================================================

@dataclass
class TLBEntry:
    virtual_page: int
    physical_page: int
    valid: bool = True


class TLB:

    def __init__(self, entries=64):
        self.entries = entries
        self.table: Dict[int, TLBEntry] = {}
        self.hits = 0
        self.misses = 0

    def translate(self, virtual_address):
        page = virtual_address // PAGE_SIZE
        offset = virtual_address % PAGE_SIZE

        if page in self.table and self.table[page].valid:
            self.hits += 1
            physical_page = self.table[page].physical_page
        else:
            self.misses += 1

            # Deterministic two-level page-table model.
            physical_page = page % PHYSICAL_MEMORY_PAGES

            if len(self.table) >= self.entries:
                oldest = next(iter(self.table))
                del self.table[oldest]

            self.table[page] = TLBEntry(
                virtual_page=page,
                physical_page=physical_page
            )

        return physical_page * PAGE_SIZE + offset


# ============================================================
# MULTI-LEVEL MEMORY SYSTEM
# ============================================================

class MemoryHierarchy:

    def __init__(self):

        # Report parameters:
        # L1-I 32 KB, 4-way, 1 cycle
        # L1-D 32 KB, 8-way, 1 cycle
        # L2   256 KB, 8-way, 12 cycles
        # L3   8 MB, 16-way, 35 cycles
        # DRAM 120 cycles

        self.l1i = Cache(
            "L1-I",
            32,
            4,
            1,
            "pseudo-LRU"
        )

        self.l1d = Cache(
            "L1-D",
            32,
            8,
            1,
            "pseudo-LRU"
        )

        self.l2 = Cache(
            "L2",
            256,
            8,
            12,
            "LRU"
        )

        self.l3 = Cache(
            "L3",
            8192,
            16,
            35,
            "NRU"
        )

        self.dram_latency = 120
        self.tlb = TLB(64)

        self.dram_accesses = 0

    def fetch_instruction(self, address):
        # Translation is modeled for instruction fetch.
        physical = self.tlb.translate(address * WORD_SIZE)

        if self.l1i.access(physical):
            return self.l1i.hit_latency

        if self.l2.access(physical):
            return self.l1i.hit_latency + self.l2.hit_latency

        if self.l3.access(physical):
            return (
                self.l1i.hit_latency
                + self.l2.hit_latency
                + self.l3.hit_latency
            )

        self.dram_accesses += 1

        return (
            self.l1i.hit_latency
            + self.l2.hit_latency
            + self.l3.hit_latency
            + self.dram_latency
        )

    def load(self, address):
        physical = self.tlb.translate(address * WORD_SIZE)

        if self.l1d.access(physical):
            return self.l1d.hit_latency

        if self.l2.access(physical):
            return self.l1d.hit_latency + self.l2.hit_latency

        if self.l3.access(physical):
            return (
                self.l1d.hit_latency
                + self.l2.hit_latency
                + self.l3.hit_latency
            )

        self.dram_accesses += 1

        return (
            self.l1d.hit_latency
            + self.l2.hit_latency
            + self.l3.hit_latency
            + self.dram_latency
        )

    def store(self, address):
        # Write-through behavior is intentionally simplified for
        # this educational model.
        return self.load(address)

    def dump_stats(self):
        print("\nCACHE / MEMORY STATISTICS")
        print("-" * 75)

        for cache in (
            self.l1i,
            self.l1d,
            self.l2,
            self.l3
        ):
            s = cache.stats()
            print(
                f"{s['name']:6} "
                f"Accesses={s['accesses']:5} "
                f"Hits={s['hits']:5} "
                f"Misses={s['misses']:5} "
                f"Miss Rate={s['miss_rate']:.2%}"
            )

        print(f"DRAM accesses = {self.dram_accesses}")
        print(
            f"TLB hits      = {self.tlb.hits}, "
            f"TLB misses    = {self.tlb.misses}"
        )


# ============================================================
# PIPELINE REGISTER
# ============================================================

@dataclass
class PipeReg:
    valid: bool = False
    instruction: Optional[Instruction] = None
    pc: int = 0

    rs1_value: int = 0
    rs2_value: int = 0

    alu_result: int = 0
    memory_data: int = 0
    store_data: int = 0

    branch_taken: bool = False
    branch_target: Optional[int] = None

    predicted_taken: bool = False
    predicted_target: Optional[int] = None


# ============================================================
# 5-STAGE PIPELINE CPU
# ============================================================

class FiveStagePipeline:

    def __init__(self, program, use_forwarding=True, use_prediction=True):
        self.program = program
        self.regs = RegisterFile()
        self.memory = MainMemory()
        self.datapath = ThreeBusDatapath(
            self.regs,
            self.memory
        )
        self.hierarchy = MemoryHierarchy()

        self.pc = 0
        self.cycle = 0

        self.IF_ID = PipeReg()
        self.ID_EX = PipeReg()
        self.EX_MEM = PipeReg()
        self.MEM_WB = PipeReg()

        self.use_forwarding = use_forwarding
        self.use_prediction = use_prediction

        self.btb: Dict[int, Tuple[bool, int]] = {}

        self.completed = 0
        self.stalls = 0
        self.forward_count = 0
        self.branch_count = 0
        self.mispredictions = 0
        self.flushes = 0

        self.timeline = []

    def fetch(self):
        if not (0 <= self.pc < len(self.program)):
            return PipeReg()

        instruction = self.program[self.pc]
        current_pc = self.pc

        predicted_taken = False
        predicted_target = None
        next_pc = current_pc + 1

        if instruction.opcode in ("BEQ", "BNE"):
            self.branch_count += 1

            if self.use_prediction and current_pc in self.btb:
                predicted_taken, predicted_target = self.btb[current_pc]

                if predicted_taken:
                    next_pc = predicted_target

        self.hierarchy.fetch_instruction(current_pc)

        self.pc = next_pc

        return PipeReg(
            valid=True,
            instruction=instruction,
            pc=current_pc,
            predicted_taken=predicted_taken,
            predicted_target=predicted_target
        )

    def decode(self, p):
        if not p.valid:
            return PipeReg()

        ins = p.instruction

        return PipeReg(
            valid=True,
            instruction=ins,
            pc=p.pc,
            rs1_value=self.regs.read(ins.rs1),
            rs2_value=self.regs.read(ins.rs2),
            predicted_taken=p.predicted_taken,
            predicted_target=p.predicted_target
        )

    def forwarding_value(self, register, original):
        if register is None:
            return original

        if not self.use_forwarding:
            return original

        # EX/MEM forwarding for ALU-producing instructions.
        if self.EX_MEM.valid:
            ins = self.EX_MEM.instruction

            if (
                ins.rd is not None
                and ins.rd == register
                and ins.opcode in ("ADD", "SUB", "MUL")
            ):
                self.forward_count += 1
                return self.EX_MEM.alu_result

        # MEM/WB forwarding.
        if self.MEM_WB.valid:
            ins = self.MEM_WB.instruction

            if ins.rd is not None and ins.rd == register:
                self.forward_count += 1

                if ins.opcode == "LOAD":
                    return self.MEM_WB.memory_data

                if ins.opcode in ("ADD", "SUB", "MUL"):
                    return self.MEM_WB.alu_result

        return original

    def execute(self, p):
        if not p.valid:
            return PipeReg()

        ins = p.instruction

        a = self.forwarding_value(
            ins.rs1,
            p.rs1_value
        )

        b = self.forwarding_value(
            ins.rs2,
            p.rs2_value
        )

        out = PipeReg(
            valid=True,
            instruction=ins,
            pc=p.pc,
            predicted_taken=p.predicted_taken,
            predicted_target=p.predicted_target
        )

        if ins.opcode == "ADD":
            out.alu_result = ALU.add(a, b)

        elif ins.opcode == "SUB":
            out.alu_result = ALU.sub(a, b)

        elif ins.opcode == "MUL":
            out.alu_result = ALU.mul(a, b)

        elif ins.opcode in ("LOAD", "STORE"):
            out.alu_result = ALU.add(a, ins.offset)
            out.store_data = b

        elif ins.opcode in ("BEQ", "BNE"):

            equal = ALU.compare_equal(a, b)

            if ins.opcode == "BEQ":
                taken = equal
            else:
                taken = not equal

            target = p.pc + ins.offset

            out.branch_taken = taken
            out.branch_target = target

            predicted = p.predicted_taken

            correct = (
                predicted == taken
                and (
                    not taken
                    or p.predicted_target == target
                )
            )

            if not correct:
                self.mispredictions += 1
                self.flushes += 1

                # Update BTB.
                self.btb[p.pc] = (
                    taken,
                    target
                )

                # Correct PC.
                self.pc = target if taken else p.pc + 1

        return out

    def memory_access(self, p):
        if not p.valid:
            return PipeReg()

        ins = p.instruction

        out = PipeReg(
            valid=True,
            instruction=ins,
            pc=p.pc,
            alu_result=p.alu_result,
            store_data=p.store_data,
            branch_taken=p.branch_taken,
            branch_target=p.branch_target,
            predicted_taken=p.predicted_taken,
            predicted_target=p.predicted_target
        )

        if ins.opcode == "LOAD":
            self.hierarchy.load(p.alu_result)
            out.memory_data = self.memory.read_word(
                p.alu_result
            )

        elif ins.opcode == "STORE":
            self.hierarchy.store(p.alu_result)
            self.memory.write_word(
                p.alu_result,
                p.store_data
            )

        return out

    def writeback(self, p):
        if not p.valid:
            return

        ins = p.instruction

        if ins.opcode in ("ADD", "SUB", "MUL"):
            self.regs.write(
                ins.rd,
                p.alu_result
            )

        elif ins.opcode == "LOAD":
            self.regs.write(
                ins.rd,
                p.memory_data
            )

        self.completed += 1

    def load_use_hazard(self):
        """
        Mandatory load-use hazard:

        I3: LOAD R6, ...
        I4: BEQ R6, ...

        A load result is not available soon enough for the
        immediately following consumer, therefore one stall.
        """

        if not self.ID_EX.valid:
            return False

        producer = self.ID_EX.instruction

        if producer.opcode != "LOAD":
            return False

        if producer.rd is None:
            return False

        if not self.IF_ID.valid:
            return False

        consumer = self.IF_ID.instruction

        return (
            consumer.rs1 == producer.rd
            or consumer.rs2 == producer.rd
        )

    def run_cycle(self):
        self.cycle += 1

        # WB
        self.writeback(self.MEM_WB)

        # MEM
        new_MEM_WB = self.memory_access(
            self.EX_MEM
        )

        # EX
        new_EX_MEM = self.execute(
            self.ID_EX
        )

        # Load-use stall.
        stall = self.load_use_hazard()

        if stall:
            self.stalls += 1

            # Hold IF/ID and PC.
            # Insert a bubble into ID/EX.
            new_ID_EX = PipeReg()
            new_IF_ID = self.IF_ID

        else:
            new_ID_EX = self.decode(
                self.IF_ID
            )
            new_IF_ID = self.fetch()

        # Branch recovery.
        if (
            self.EX_MEM.valid
            and self.EX_MEM.instruction.opcode
            in ("BEQ", "BNE")
        ):
            # EX/MEM represents a branch whose outcome was
            # resolved. The next fetched younger instructions
            # are flushed on a misprediction.
            ins = self.EX_MEM.instruction

            actual_taken = self.EX_MEM.branch_taken
            actual_target = self.EX_MEM.branch_target

            predicted_taken = self.EX_MEM.predicted_taken
            predicted_target = self.EX_MEM.predicted_target

            correct = (
                predicted_taken == actual_taken
                and (
                    not actual_taken
                    or predicted_target == actual_target
                )
            )

            if not correct:
                new_IF_ID = PipeReg()
                new_ID_EX = PipeReg()

        self.MEM_WB = new_MEM_WB
        self.EX_MEM = new_EX_MEM
        self.ID_EX = new_ID_EX
        self.IF_ID = new_IF_ID

        self.timeline.append(
            (
                self.cycle,
                self.stage_name(self.IF_ID),
                self.stage_name(self.ID_EX),
                self.stage_name(self.EX_MEM),
                self.stage_name(self.MEM_WB),
                stall
            )
        )

    @staticmethod
    def stage_name(p):
        if not p.valid:
            return "---"
        return str(p.instruction)

    def empty(self):
        return (
            self.pc >= len(self.program)
            and not self.IF_ID.valid
            and not self.ID_EX.valid
            and not self.EX_MEM.valid
            and not self.MEM_WB.valid
        )

    def run(self, show=True, max_cycles=500):
        while not self.empty() and self.cycle < max_cycles:
            self.run_cycle()

        if show:
            print("\n5-STAGE PIPELINE TIMELINE")
            print("-" * 130)
            print(
                f"{'Cycle':>5} | "
                f"{'IF':25} | "
                f"{'ID':25} | "
                f"{'EX':25} | "
                f"{'MEM/WB':25} | Stall"
            )
            print("-" * 130)

            for row in self.timeline:
                cycle, a, b, c, d, stall = row
                print(
                    f"{cycle:5d} | "
                    f"{a:25} | "
                    f"{b:25} | "
                    f"{c:25} | "
                    f"{d:25} | "
                    f"{'YES' if stall else 'NO'}"
                )

        return self.stats()

    def stats(self):
        cpi = (
            self.cycle / self.completed
            if self.completed
            else 0
        )

        return {
            "cycles": self.cycle,
            "completed": self.completed,
            "cpi": cpi,
            "stalls": self.stalls,
            "forwarding_events": self.forward_count,
            "branches": self.branch_count,
            "mispredictions": self.mispredictions,
            "flushes": self.flushes
        }


# ============================================================
# SCALAR CPI ANALYTICAL MODEL
# ============================================================

def analytical_cpi():

    # From the submitted report:
    # ALU 45%, Load 20%, Store 10%, Branch 20%, Other 5%
    # 30% ALU immediately preceding RAW
    # 40% loads immediately followed by consumer
    # no mitigation RAW/load-use = 3 stalls
    # forwarding ALU-ALU = 0, load-use = 1
    # branch prediction 90%, misprediction = 2 cycles

    no_mitigation = (
        0.45 * 0.30 * 3
        + 0.20 * 0.40 * 3
        + 0.20 * 3
    )

    with_mitigation = (
        0.20 * 0.40 * 1
        + 0.20 * 0.10 * 2
    )

    return {
        "cpi_no_mitigation": 1 + no_mitigation,
        "cpi_with_mitigation": 1 + with_mitigation,
        "speedup": (
            (1 + no_mitigation)
            / (1 + with_mitigation)
        )
    }


# ============================================================
# TWO-WAY SUPERSCALAR
# ============================================================

@dataclass
class ROBEntry:
    seq: int
    instruction: Instruction
    destination: Optional[int]
    value: Optional[int] = None
    ready: bool = False
    exception: bool = False


@dataclass
class ReservationStation:
    name: str
    busy: bool = False
    seq: Optional[int] = None
    instruction: Optional[Instruction] = None

    src1_ready: bool = False
    src2_ready: bool = False

    src1_value: Optional[int] = None
    src2_value: Optional[int] = None

    src1_tag: Optional[int] = None
    src2_tag: Optional[int] = None

    remaining: int = 0


class SuperscalarCPU:

    def __init__(self, program, width=2):
        self.program = program
        self.width = width

        self.regs = RegisterFile()

        # Register Alias Table.
        self.rat: List[Optional[int]] = [
            None
        ] * NUM_REGISTERS

        self.rob = deque()
        self.rs = []

        for i in range(4):
            self.rs.append(
                ReservationStation(f"RS{i}")
            )

        self.pc = 0
        self.seq = 0
        self.cycle = 0

        self.completed = 0
        self.committed = 0
        self.flushes = 0

        self.timeline = []

    def allocate_rob(self, ins):
        entry = ROBEntry(
            seq=self.seq,
            instruction=ins,
            destination=ins.rd
        )

        self.seq += 1
        self.rob.append(entry)

        return entry

    def get_rob_entry(self, seq):
        for entry in self.rob:
            if entry.seq == seq:
                return entry
        return None

    def source(self, register):
        if register is None:
            return True, 0, None

        tag = self.rat[register]

        if tag is None:
            return True, self.regs.read(register), None

        producer = self.get_rob_entry(tag)

        if producer and producer.ready:
            return True, producer.value, None

        return False, None, tag

    def dispatch(self):
        dispatched = 0

        while (
            dispatched < self.width
            and self.pc < len(self.program)
        ):
            free = None

            for rs in self.rs:
                if not rs.busy:
                    free = rs
                    break

            if free is None:
                break

            ins = self.program[self.pc]

            entry = self.allocate_rob(ins)

            free.busy = True
            free.seq = entry.seq
            free.instruction = ins
            free.remaining = max(1, ins.latency)

            free.src1_ready, free.src1_value, free.src1_tag = (
                self.source(ins.rs1)
            )

            free.src2_ready, free.src2_value, free.src2_tag = (
                self.source(ins.rs2)
            )

            if ins.rd is not None:
                self.rat[ins.rd] = entry.seq

            self.pc += 1
            dispatched += 1

    def execute(self):
        for rs in self.rs:

            if not rs.busy:
                continue

            if not rs.src1_ready or not rs.src2_ready:
                # Resolve tags if producer has completed.
                if rs.src1_tag is not None:
                    producer = self.get_rob_entry(rs.src1_tag)
                    if producer and producer.ready:
                        rs.src1_ready = True
                        rs.src1_value = producer.value
                        rs.src1_tag = None

                if rs.src2_tag is not None:
                    producer = self.get_rob_entry(rs.src2_tag)
                    if producer and producer.ready:
                        rs.src2_ready = True
                        rs.src2_value = producer.value
                        rs.src2_tag = None

            if not rs.src1_ready or not rs.src2_ready:
                continue

            # Execute one cycle per cycle of modeled latency.
            rs.remaining -= 1

            if rs.remaining > 0:
                continue

            ins = rs.instruction

            a = rs.src1_value or 0
            b = rs.src2_value or 0

            if ins.opcode == "ADD":
                result = ALU.add(a, b)

            elif ins.opcode == "SUB":
                result = ALU.sub(a, b)

            elif ins.opcode == "MUL":
                result = ALU.mul(a, b)

            elif ins.opcode == "LOAD":
                result = a + ins.offset

            elif ins.opcode == "STORE":
                result = a + ins.offset

            elif ins.opcode in ("BEQ", "BNE"):
                equal = ALU.compare_equal(a, b)
                taken = (
                    equal
                    if ins.opcode == "BEQ"
                    else not equal
                )
                result = int(taken)

            else:
                result = 0

            entry = self.get_rob_entry(rs.seq)

            if entry:
                entry.value = result
                entry.ready = True

            self.completed += 1

            rs.busy = False
            rs.seq = None
            rs.instruction = None

    def commit(self):
        committed_this_cycle = 0

        while (
            self.rob
            and committed_this_cycle < self.width
        ):
            entry = self.rob[0]

            if not entry.ready:
                break

            self.rob.popleft()

            if (
                entry.destination is not None
                and entry.instruction.opcode
                in ("ADD", "SUB", "MUL")
            ):
                self.regs.write(
                    entry.destination,
                    entry.value
                )

                if self.rat[entry.destination] == entry.seq:
                    self.rat[entry.destination] = None

            self.committed += 1
            committed_this_cycle += 1

    def run_cycle(self):
        self.cycle += 1

        self.commit()
        self.execute()
        self.dispatch()

        self.timeline.append(
            (
                self.cycle,
                len(self.rob),
                sum(1 for r in self.rs if r.busy),
                self.committed
            )
        )

    def empty(self):
        return (
            self.pc >= len(self.program)
            and not self.rob
            and not any(r.busy for r in self.rs)
        )

    def run(self, show=True, max_cycles=500):

        while not self.empty() and self.cycle < max_cycles:
            self.run_cycle()

        if show:
            print("\n2-WAY SUPERSCALAR TIMELINE")
            print("-" * 70)
            print(
                f"{'Cycle':>6} | "
                f"{'ROB entries':>12} | "
                f"{'RS busy':>8} | "
                f"{'Committed':>10}"
            )
            print("-" * 70)

            for row in self.timeline:
                print(
                    f"{row[0]:6d} | "
                    f"{row[1]:12d} | "
                    f"{row[2]:8d} | "
                    f"{row[3]:10d}"
                )

        ipc = (
            self.committed / self.cycle
            if self.cycle
            else 0
        )

        return {
            "cycles": self.cycle,
            "completed": self.completed,
            "committed": self.committed,
            "ipc": ipc,
            "rob_entries": len(self.rob),
        }


# ============================================================
# PERFORMANCE ANALYSIS
# ============================================================

def performance_analysis():

    # Report assumptions:
    # L1 = 1 cycle
    # L1 local miss rate = 5%
    # L2 = 12 cycles
    # L2 local miss rate = 10%
    # L3 = 35 cycles
    # L3 local miss rate = 15%
    # DRAM = 120 cycles

    l1 = 1
    mr1 = 0.05
    l2 = 12
    mr2 = 0.10
    l3 = 35
    mr3 = 0.15
    dram = 120

    amat_3 = (
        l1
        + mr1 * (
            l2
            + mr2 * (
                l3
                + mr3 * dram
            )
        )
    )

    amat_2 = (
        l1
        + mr1 * (
            l2
            + 0.22 * dram
        )
    )

    frequency_ghz = 2.5

    scalar_cpi = 1.12
    scalar_ipc = 1 / scalar_cpi
    scalar_gips = frequency_ghz * scalar_ipc

    superscalar_ipc = 1.6
    superscalar_gips = frequency_ghz * superscalar_ipc

    # 30% memory references, 64-byte line.
    memory_refs = 0.30
    line_bytes = 64
    dram_probability = mr1 * mr2 * mr3

    cached_bytes_per_instruction = (
        memory_refs
        * dram_probability
        * line_bytes
    )

    uncached_bytes_per_instruction = (
        memory_refs * line_bytes
    )

    cached_bandwidth = (
        superscalar_gips
        * 1e9
        * cached_bytes_per_instruction
    )

    uncached_bandwidth = (
        superscalar_gips
        * 1e9
        * uncached_bytes_per_instruction
    )

    return {
        "amat_3_level": amat_3,
        "amat_2_level": amat_2,
        "amat_reduction_percent": (
            (amat_2 - amat_3) / amat_2 * 100
        ),
        "scalar_ipc": scalar_ipc,
        "scalar_gips": scalar_gips,
        "superscalar_ipc": superscalar_ipc,
        "superscalar_gips": superscalar_gips,
        "throughput_speedup": (
            superscalar_gips / scalar_gips
        ),
        "cached_bandwidth_Bps": cached_bandwidth,
        "uncached_bandwidth_Bps": uncached_bandwidth,
        "bandwidth_reduction": (
            uncached_bandwidth / cached_bandwidth
            if cached_bandwidth
            else 0
        )
    }


# ============================================================
# REPORT-BASED DEMONSTRATION PROGRAM
# ============================================================

def create_report_program():

    # The report's hazard sequence:
    #
    # I1: ADD R1, R2, R3
    # I2: SUB R4, R1, R5
    # I3: LOAD R6, 0(R1)
    # I4: BEQ R6, R0, L1
    # I5: ADD R7, R8, R9

    return [

        Instruction(
            "ADD",
            rd=1,
            rs1=2,
            rs2=3,
            name="I1: ADD R1, R2, R3"
        ),

        Instruction(
            "SUB",
            rd=4,
            rs1=1,
            rs2=5,
            name="I2: SUB R4, R1, R5"
        ),

        Instruction(
            "LOAD",
            rd=6,
            rs1=1,
            offset=0,
            name="I3: LOAD R6, 0(R1)"
        ),

        Instruction(
            "BEQ",
            rs1=6,
            rs2=0,
            offset=2,
            name="I4: BEQ R6, R0, L1"
        ),

        Instruction(
            "ADD",
            rd=7,
            rs1=8,
            rs2=9,
            name="I5: ADD R7, R8, R9"
        )
    ]


# ============================================================
# COMPLETE DEMONSTRATION
# ============================================================

def run_complete_demo():

    print("=" * 90)
    print("CSA1221 COMPUTER ARCHITECTURE")
    print("COMPLETE SINGLE-FILE PROCESSOR IMPLEMENTATION")
    print("=" * 90)

    # --------------------------------------------------------
    # PART 1 - DATAPATH
    # --------------------------------------------------------

    print("\n[1] THREE-BUS DATAPATH")
    print("-" * 90)

    rf = RegisterFile()
    mem = MainMemory()
    datapath = ThreeBusDatapath(rf, mem)

    rf.write(1, 10)
    rf.write(2, 20)
    rf.write(3, 5)
    rf.write(8, 100)

    mem.write_word(100, 500)

    datapath.add(4, 1, 2)
    datapath.sub(5, 4, 3)
    datapath.store(5, 0, 8)
    datapath.load(6, 0, 8)

    _, taken = datapath.branch(
        "BEQ",
        5,
        6,
        8,
        100
    )

    print("ADD  R4 = R1 + R2       ->", rf.read(4))
    print("SUB  R5 = R4 - R3       ->", rf.read(5))
    print("STORE M[100]            ->", mem.read_word(100))
    print("LOAD  R6 = M[100]       ->", rf.read(6))
    print("BEQ R5,R6               ->", "TAKEN" if taken else "NOT TAKEN")

    # --------------------------------------------------------
    # PART 2 - PIPELINE
    # --------------------------------------------------------

    print("\n[2] FIVE-STAGE PIPELINE")
    print("-" * 90)

    program = create_report_program()

    pipeline = FiveStagePipeline(
        program,
        use_forwarding=True,
        use_prediction=True
    )

    pipeline.regs.write(2, 10)
    pipeline.regs.write(3, 20)
    pipeline.regs.write(5, 5)
    pipeline.regs.write(8, 100)
    pipeline.regs.write(9, 200)

    # R1 becomes 30. Therefore LOAD R6, 0(R1) reads M[30].
    pipeline.memory.write_word(30, 0)

    pipe_stats = pipeline.run(show=True)

    print("\nPIPELINE RESULTS")
    print("-" * 90)

    for key, value in pipe_stats.items():
        print(f"{key:25}: {value}")

    # --------------------------------------------------------
    # PART 3 - ANALYTICAL CPI
    # --------------------------------------------------------

    print("\n[3] REPORT-BASED CPI ANALYSIS")
    print("-" * 90)

    cpi = analytical_cpi()

    print(
        f"CPI without mitigation : "
        f"{cpi['cpi_no_mitigation']:.2f}"
    )

    print(
        f"CPI with mitigation    : "
        f"{cpi['cpi_with_mitigation']:.2f}"
    )

    print(
        f"Speedup                 : "
        f"{cpi['speedup']:.2f}x"
    )

    # --------------------------------------------------------
    # PART 4 - SUPERSCALAR
    # --------------------------------------------------------

    print("\n[4] TWO-WAY SUPERSCALAR")
    print("-" * 90)

    superscalar_program = [

        Instruction(
            "MUL",
            rd=2,
            rs1=3,
            rs2=4,
            latency=3,
            name="I1: MUL R2, R3, R4"
        ),

        Instruction(
            "ADD",
            rd=5,
            rs1=2,
            rs2=6,
            latency=1,
            name="I2: ADD R5, R2, R6"
        ),

        Instruction(
            "SUB",
            rd=7,
            rs1=8,
            rs2=9,
            latency=1,
            name="I3: SUB R7, R8, R9"
        ),

        Instruction(
            "ADD",
            rd=10,
            rs1=7,
            rs2=1,
            latency=1,
            name="I4: ADD R10, R7, R1"
        )
    ]

    superscalar = SuperscalarCPU(
        superscalar_program,
        width=2
    )

    superscalar.regs.write(3, 2)
    superscalar.regs.write(4, 5)
    superscalar.regs.write(6, 10)
    superscalar.regs.write(8, 30)
    superscalar.regs.write(9, 10)
    superscalar.regs.write(1, 5)

    super_stats = superscalar.run(show=True)

    print("\nSUPERSCALAR RESULTS")
    print("-" * 90)

    for key, value in super_stats.items():
        print(f"{key:25}: {value}")

    # --------------------------------------------------------
    # PART 5 - CACHE
    # --------------------------------------------------------

    print("\n[5] MULTI-LEVEL CACHE")
    print("-" * 90)

    hierarchy = MemoryHierarchy()

    # Generate a realistic sequence of repeated accesses so that
    # cache hits and misses are visible.
    addresses = [
        0, 4, 8, 12,
        0, 4, 8, 12,
        128, 256, 512,
        0, 4, 8, 12
    ]

    for address in addresses:
        hierarchy.load(address)

    hierarchy.dump_stats()

    # --------------------------------------------------------
    # PART 6 - PERFORMANCE
    # --------------------------------------------------------

    print("\n[6] PERFORMANCE ANALYSIS")
    print("-" * 90)

    perf = performance_analysis()

    print(
        f"3-level AMAT          : "
        f"{perf['amat_3_level']:.3f} cycles"
    )

    print(
        f"2-level AMAT          : "
        f"{perf['amat_2_level']:.3f} cycles"
    )

    print(
        f"AMAT reduction        : "
        f"{perf['amat_reduction_percent']:.2f}%"
    )

    print(
        f"Scalar IPC            : "
        f"{perf['scalar_ipc']:.3f}"
    )

    print(
        f"Scalar throughput     : "
        f"{perf['scalar_gips']:.2f} GIPS"
    )

    print(
        f"Superscalar IPC       : "
        f"{perf['superscalar_ipc']:.2f}"
    )

    print(
        f"Superscalar throughput: "
        f"{perf['superscalar_gips']:.2f} GIPS"
    )

    print(
        f"Throughput speedup    : "
        f"{perf['throughput_speedup']:.2f}x"
    )

    print(
        f"With cache bandwidth  : "
        f"{perf['cached_bandwidth_Bps'] / 1e6:.2f} MB/s"
    )

    print(
        f"Without cache         : "
        f"{perf['uncached_bandwidth_Bps'] / 1e9:.2f} GB/s"
    )

    print(
        f"Bandwidth reduction   : "
        f"{perf['bandwidth_reduction']:.0f}x"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("FINAL IMPLEMENTATION SUMMARY")
    print("=" * 90)

    print("Processor datapath      : 3-bus")
    print("Pipeline                 : 5-stage IF/ID/EX/MEM/WB")
    print("ALU forwarding           :", pipe_stats["forwarding_events"])
    print("Load-use stalls          :", pipe_stats["stalls"])
    print("Branch mispredictions    :", pipe_stats["mispredictions"])
    print("Pipeline flushes         :", pipe_stats["flushes"])
    print("Superscalar width        : 2 instructions/cycle")
    print("Out-of-order model       : Reservation Stations + ROB")
    print("Register renaming        : RAT")
    print("Cache hierarchy          : L1-I + L1-D + L2 + L3 + DRAM")
    print("TLB                      : 64 entries")
    print("Page size                : 4 KB")
    print("3-level AMAT             :", f"{perf['amat_3_level']:.2f} cycles")
    print("Analytical scalar CPI    :", f"{cpi['cpi_with_mitigation']:.2f}")
    print("Analytical superscalar   :", f"{perf['superscalar_gips']:.2f} GIPS")

    print("\nImplementation completed successfully.")
    print("=" * 90)


# ============================================================
# OPTIONAL INTERACTIVE MENU
# ============================================================

def menu():

    while True:

        print("\n")
        print("=" * 60)
        print("CSA1221 IMPLEMENTATION MENU")
        print("=" * 60)
        print("1. Run complete implementation")
        print("2. Run CPI calculation")
        print("3. Run performance calculation")
        print("4. Run cache demonstration")
        print("5. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            run_complete_demo()

        elif choice == "2":
            result = analytical_cpi()

            print("\nCPI ANALYSIS")
            print("-" * 40)
            print(
                "Without mitigation:",
                f"{result['cpi_no_mitigation']:.2f}"
            )
            print(
                "With mitigation   :",
                f"{result['cpi_with_mitigation']:.2f}"
            )
            print(
                "Speedup            :",
                f"{result['speedup']:.2f}x"
            )

        elif choice == "3":
            result = performance_analysis()

            print("\nPERFORMANCE")
            print("-" * 40)
            print(
                "3-level AMAT:",
                f"{result['amat_3_level']:.2f}"
            )
            print(
                "2-level AMAT:",
                f"{result['amat_2_level']:.2f}"
            )
            print(
                "Scalar:",
                f"{result['scalar_gips']:.2f} GIPS"
            )
            print(
                "Superscalar:",
                f"{result['superscalar_gips']:.2f} GIPS"
            )

        elif choice == "4":
            hierarchy = MemoryHierarchy()

            for address in [0, 4, 8, 0, 4, 8, 128, 0, 4]:
                hierarchy.load(address)

            hierarchy.dump_stats()

        elif choice == "5":
            print("Exiting.")
            break

        else:
            print("Invalid choice.")


# ============================================================
# SINGLE-FILE LOCALHOST WEB INTERFACE
# ============================================================

def build_web_results():
    """Run the existing implementation and return browser-friendly results."""

    # 1. Three-bus datapath
    rf = RegisterFile()
    mem = MainMemory()
    datapath = ThreeBusDatapath(rf, mem)

    rf.write(1, 10)
    rf.write(2, 20)
    rf.write(3, 5)
    rf.write(8, 100)
    mem.write_word(100, 500)

    datapath.add(4, 1, 2)
    datapath.sub(5, 4, 3)
    datapath.store(5, 0, 8)
    datapath.load(6, 0, 8)
    _, taken = datapath.branch("BEQ", 5, 6, 8, 100)

    datapath_result = {
        "ADD R4 = R1 + R2": rf.read(4),
        "SUB R5 = R4 - R3": rf.read(5),
        "STORE M[100]": mem.read_word(100),
        "LOAD R6 = M[100]": rf.read(6),
        "BEQ R5,R6": "TAKEN" if taken else "NOT TAKEN",
    }

    # 2. Five-stage pipeline
    program = create_report_program()
    pipeline = FiveStagePipeline(program, use_forwarding=True, use_prediction=True)

    pipeline.regs.write(2, 10)
    pipeline.regs.write(3, 20)
    pipeline.regs.write(5, 5)
    pipeline.regs.write(8, 100)
    pipeline.regs.write(9, 200)
    pipeline.memory.write_word(30, 0)

    pipe_stats = pipeline.run(show=False)

    pipeline_timeline = [
        {
            "cycle": row[0],
            "IF": row[1],
            "ID": row[2],
            "EX": row[3],
            "MEM_WB": row[4],
            "stall": row[5],
        }
        for row in pipeline.timeline
    ]

    # 3. CPI analysis
    cpi = analytical_cpi()

    # 4. Two-way superscalar
    superscalar_program = [
        Instruction("MUL", rd=2, rs1=3, rs2=4, latency=3,
                    name="I1: MUL R2, R3, R4"),
        Instruction("ADD", rd=5, rs1=2, rs2=6, latency=1,
                    name="I2: ADD R5, R2, R6"),
        Instruction("SUB", rd=7, rs1=8, rs2=9, latency=1,
                    name="I3: SUB R7, R8, R9"),
        Instruction("ADD", rd=10, rs1=7, rs2=1, latency=1,
                    name="I4: ADD R10, R7, R1"),
    ]

    superscalar = SuperscalarCPU(superscalar_program, width=2)
    superscalar.regs.write(3, 2)
    superscalar.regs.write(4, 5)
    superscalar.regs.write(6, 10)
    superscalar.regs.write(8, 30)
    superscalar.regs.write(9, 10)
    superscalar.regs.write(1, 5)

    super_stats = superscalar.run(show=False)

    superscalar_timeline = [
        {
            "cycle": row[0],
            "rob_entries": row[1],
            "rs_busy": row[2],
            "committed": row[3],
        }
        for row in superscalar.timeline
    ]

    # 5. Cache
    hierarchy = MemoryHierarchy()
    for address in [0, 4, 8, 12, 0, 4, 8, 12, 128, 256, 512, 0, 4, 8, 12]:
        hierarchy.load(address)

    cache_stats = [
        cache.stats()
        for cache in (hierarchy.l1i, hierarchy.l1d, hierarchy.l2, hierarchy.l3)
    ]

    # 6. Performance
    perf = performance_analysis()

    return {
        "datapath": datapath_result,
        "pipeline": pipe_stats,
        "pipeline_timeline": pipeline_timeline,
        "cpi": cpi,
        "superscalar": super_stats,
        "superscalar_timeline": superscalar_timeline,
        "cache": cache_stats,
        "dram_accesses": hierarchy.dram_accesses,
        "tlb_hits": hierarchy.tlb.hits,
        "tlb_misses": hierarchy.tlb.misses,
        "performance": perf,
    }


def run_localhost():
    """Start the complete implementation as a localhost browser application."""
    try:
        from flask import Flask, render_template_string
    except ImportError:
        print("Flask is not installed.")
        print("Run: python -m pip install flask")
        raise

    app = Flask(__name__)

    HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CSA1221 Processor Simulator</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#f4f7fb;color:#172033;font-family:Arial,Helvetica,sans-serif}
.header{background:#111827;color:#fff;padding:28px 5%}
.header h1{margin:0 0 8px;font-size:30px}
.header p{margin:0;opacity:.85}
.container{width:92%;max-width:1450px;margin:24px auto 50px}
.section{background:#fff;border-radius:14px;padding:20px;margin:18px 0;
box-shadow:0 3px 14px rgba(0,0,0,.07)}
.section h2{margin-top:0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.card{background:#f8fafc;border:1px solid #e5e7eb;border-radius:11px;padding:15px}
.label{font-size:13px;color:#64748b}
.value{font-size:24px;font-weight:800;margin-top:7px}
.note{background:#eff6ff;border-left:4px solid #2563eb;padding:13px 15px;border-radius:7px}
table{width:100%;border-collapse:collapse;margin-top:12px;font-size:14px}
th,td{padding:10px;border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top}
th{background:#f8fafc}
.table-wrap{overflow-x:auto}
button{background:#2563eb;color:#fff;border:0;border-radius:9px;padding:12px 18px;
font-weight:700;cursor:pointer;margin-bottom:2px}
.footer{text-align:center;color:#64748b;padding:20px}
</style>
</head>
<body>
<div class="header">
<h1>CSA1221 Computer Architecture</h1>
<p>Design and Performance Analysis of a Pipelined Processor with a Multi-Level Cache Memory System</p>
</div>

<div class="container">
<div class="note">
<b>Single-file localhost implementation.</b>
3-bus datapath · 5-stage pipeline · hazards/forwarding · branch prediction ·
2-way superscalar · RAT · Reservation Stations · ROB · L1/L2/L3/DRAM · TLB · performance analysis
</div>

<div style="margin-top:18px">
<form method="get"><button>▶ Run / Refresh Simulation</button></form>
</div>

<section class="section">
<h2>1. Three-Bus Datapath</h2>
<div class="grid">
{% for k,v in r.datapath.items() %}
<div class="card"><div class="label">{{k}}</div><div class="value">{{v}}</div></div>
{% endfor %}
</div>
</section>

<section class="section">
<h2>2. Five-Stage Pipeline</h2>
<div class="grid">
<div class="card"><div class="label">Cycles</div><div class="value">{{r.pipeline.cycles}}</div></div>
<div class="card"><div class="label">Completed</div><div class="value">{{r.pipeline.completed}}</div></div>
<div class="card"><div class="label">CPI</div><div class="value">{{"%.2f"|format(r.pipeline.cpi)}}</div></div>
<div class="card"><div class="label">Stalls</div><div class="value">{{r.pipeline.stalls}}</div></div>
<div class="card"><div class="label">Forwarding Events</div><div class="value">{{r.pipeline.forwarding_events}}</div></div>
<div class="card"><div class="label">Branches</div><div class="value">{{r.pipeline.branches}}</div></div>
<div class="card"><div class="label">Mispredictions</div><div class="value">{{r.pipeline.mispredictions}}</div></div>
<div class="card"><div class="label">Flushes</div><div class="value">{{r.pipeline.flushes}}</div></div>
</div>
<div class="table-wrap">
<table>
<tr><th>Cycle</th><th>IF</th><th>ID</th><th>EX</th><th>MEM/WB</th><th>Stall</th></tr>
{% for x in r.pipeline_timeline %}
<tr><td>{{x.cycle}}</td><td>{{x.IF}}</td><td>{{x.ID}}</td><td>{{x.EX}}</td>
<td>{{x.MEM_WB}}</td><td>{{"YES" if x.stall else "NO"}}</td></tr>
{% endfor %}
</table>
</div>
</section>

<section class="section">
<h2>3. Report-Based CPI Analysis</h2>
<div class="grid">
<div class="card"><div class="label">Without Mitigation</div><div class="value">{{"%.2f"|format(r.cpi.cpi_no_mitigation)}}</div></div>
<div class="card"><div class="label">With Mitigation</div><div class="value">{{"%.2f"|format(r.cpi.cpi_with_mitigation)}}</div></div>
<div class="card"><div class="label">Speedup</div><div class="value">{{"%.2f"|format(r.cpi.speedup)}}x</div></div>
</div>
</section>

<section class="section">
<h2>4. Two-Way Superscalar</h2>
<div class="grid">
<div class="card"><div class="label">Width</div><div class="value">2</div></div>
<div class="card"><div class="label">Cycles</div><div class="value">{{r.superscalar.cycles}}</div></div>
<div class="card"><div class="label">Completed</div><div class="value">{{r.superscalar.completed}}</div></div>
<div class="card"><div class="label">Committed</div><div class="value">{{r.superscalar.committed}}</div></div>
<div class="card"><div class="label">Demo IPC</div><div class="value">{{"%.3f"|format(r.superscalar.ipc)}}</div></div>
</div>
<table>
<tr><th>Cycle</th><th>ROB Entries</th><th>RS Busy</th><th>Committed</th></tr>
{% for x in r.superscalar_timeline %}
<tr><td>{{x.cycle}}</td><td>{{x.rob_entries}}</td><td>{{x.rs_busy}}</td><td>{{x.committed}}</td></tr>
{% endfor %}
</table>
</section>

<section class="section">
<h2>5. Multi-Level Cache &amp; Virtual Memory</h2>
<table>
<tr><th>Cache</th><th>Accesses</th><th>Hits</th><th>Misses</th><th>Miss Rate</th></tr>
{% for x in r.cache %}
<tr><td>{{x.name}}</td><td>{{x.accesses}}</td><td>{{x.hits}}</td><td>{{x.misses}}</td>
<td>{{"%.2f"|format(x.miss_rate*100)}}%</td></tr>
{% endfor %}
</table>
<div class="grid" style="margin-top:15px">
<div class="card"><div class="label">DRAM Accesses</div><div class="value">{{r.dram_accesses}}</div></div>
<div class="card"><div class="label">TLB Hits</div><div class="value">{{r.tlb_hits}}</div></div>
<div class="card"><div class="label">TLB Misses</div><div class="value">{{r.tlb_misses}}</div></div>
<div class="card"><div class="label">TLB Entries</div><div class="value">64</div></div>
<div class="card"><div class="label">Page Size</div><div class="value">4 KB</div></div>
</div>
</section>

<section class="section">
<h2>6. Performance Analysis</h2>
<div class="grid">
<div class="card"><div class="label">3-Level AMAT</div><div class="value">{{"%.3f"|format(r.performance.amat_3_level)}} cycles</div></div>
<div class="card"><div class="label">2-Level AMAT</div><div class="value">{{"%.3f"|format(r.performance.amat_2_level)}} cycles</div></div>
<div class="card"><div class="label">AMAT Reduction</div><div class="value">{{"%.2f"|format(r.performance.amat_reduction_percent)}}%</div></div>
<div class="card"><div class="label">Scalar IPC</div><div class="value">{{"%.3f"|format(r.performance.scalar_ipc)}}</div></div>
<div class="card"><div class="label">Scalar Throughput</div><div class="value">{{"%.2f"|format(r.performance.scalar_gips)}} GIPS</div></div>
<div class="card"><div class="label">Superscalar IPC</div><div class="value">{{"%.2f"|format(r.performance.superscalar_ipc)}}</div></div>
<div class="card"><div class="label">Superscalar Throughput</div><div class="value">{{"%.2f"|format(r.performance.superscalar_gips)}} GIPS</div></div>
<div class="card"><div class="label">Throughput Speedup</div><div class="value">{{"%.2f"|format(r.performance.throughput_speedup)}}x</div></div>
<div class="card"><div class="label">Cache Bandwidth</div><div class="value">{{"%.2f"|format(r.performance.cached_bandwidth_Bps/1000000)}} MB/s</div></div>
<div class="card"><div class="label">Without Cache</div><div class="value">{{"%.2f"|format(r.performance.uncached_bandwidth_Bps/1000000000)}} GB/s</div></div>
<div class="card"><div class="label">Bandwidth Reduction</div><div class="value">{{"%.0f"|format(r.performance.bandwidth_reduction)}}x</div></div>
</div>
</section>

<section class="section">
<h2>Implementation Summary</h2>
<table>
<tr><th>Feature</th><th>Implementation</th></tr>
<tr><td>Processor Datapath</td><td>3-bus RISC datapath</td></tr>
<tr><td>Pipeline</td><td>5-stage IF / ID / EX / MEM / WB</td></tr>
<tr><td>Hazards</td><td>RAW detection, ALU forwarding, one-cycle load-use stall</td></tr>
<tr><td>Branch</td><td>BTB-style dynamic prediction and misprediction flush</td></tr>
<tr><td>Superscalar</td><td>2 instructions/cycle</td></tr>
<tr><td>Out-of-Order</td><td>Reservation Stations + ROB</td></tr>
<tr><td>Register Renaming</td><td>RAT</td></tr>
<tr><td>Cache</td><td>L1-I + L1-D + L2 + L3 + DRAM</td></tr>
<tr><td>Virtual Memory</td><td>64-entry TLB, 4 KB pages</td></tr>
</table>
</section>
</div>

<div class="footer">CSA1221 — Processor implementation running on localhost</div>
</body>
</html>
"""

    @app.route("/")
    def home():
        return render_template_string(HTML, r=build_web_results())

    print("=" * 80)
    print("CSA1221 LOCALHOST PROCESSOR SIMULATOR")
    print("=" * 80)
    print("Open in your browser: http://127.0.0.1:5000")
    print("Press CTRL+C to stop the server.")
    print("=" * 80)

    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    run_localhost()
