#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/memory/coherence.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
]#

# ============================================================================
#  Coherence State Machine
# ============================================================================
#
# Grid uses a 4-state machine (Empty, CpuDirty, AccDirty, Consistent)
# with two lock counters (cpuLock, accLock).  We adapt this to ReliQ's
# host-vs-AoSoA duality:
#
#   "CPU"  = host GA memory (AoS layout)
#   "Acc"  = AoSoA buffer (which may be on host for OpenMP, or on
#            device for OpenCL)
#
# State transitions:
#
#   ViewOpen(iokRead):
#     Empty       → error (no data to read)
#     HostDirty   → transform AoS→AoSoA, state = Consistent
#     DeviceDirty → no-op (AoSoA already has data), state stays DeviceDirty
#     Consistent  → no-op
#
#   ViewOpen(iokWrite):
#     Any state   → state = DeviceDirty (device will be overwritten)
#                   No transform needed.
#
#   ViewOpen(iokReadWrite):
#     Empty       → error
#     HostDirty   → transform AoS→AoSoA, state = DeviceDirty
#     DeviceDirty → no-op, state stays DeviceDirty
#     Consistent  → state = DeviceDirty
#
#   ViewClose(iokWrite / iokReadWrite):
#     DeviceDirty → scatter AoSoA→AoS, state = Consistent
#     OR: defer scatter (lazy write-back), state stays DeviceDirty
#
#   ViewClose(iokRead):
#     No-op.

import std/[tables]
import std/[locks]
import std/[strutils]

import bufferpool

import types/[composite]

type IOKind* = enum iokRead, iokWrite, iokReadWrite

type CoherenceStateKind = enum 
  cskEmpty, cskHostDirty, cskDeviceDirty, cskConsistent

class CoherenceEntry:
  ## Per-field coherence tracking.
  ## 
  ## Keyed by the host pointer (the GA memory address of the field).
  ## This is stable across view opens/closes because the GA allocation
  ## persists for the field's lifetime.
  var state: CoherenceStateKind
  var hostLock: int   # number of active read-mode views on host
  var deviceLock: int # number of active write/read-write views on device
  var key: BufferKey
  var entry: BufferEntry

  method init(state: CoherenceStateKind = cskEmpty) =
    this.state = state
  
  method buffer: pointer = this.entry.data
  
  method reference: RootRef = this.entry.reference

  method bytes: int = this.entry.bytes

  method `buffer=`(value: pointer) = this.entry.data = value

  method `reference=`(value: RootRef) = this.entry.reference = value

  method `bytes=`(value: int) = this.entry.bytes = value

proc `$`(p: pointer): string =
  if p.isNil: "nil" else: "0x" & cast[uint](p).toHex

class CoherenceManager:
  ## Global coherence tracking table.
  ## 
  ## Keyed by host pointer, value is CoherenceEntry.
  var entries: Table[pointer, CoherenceEntry]
  var lock: Lock

  method init = 
    this.entries = initTable[pointer, CoherenceEntry]()
    initLock this.lock

type ViewState* = object
  needsTransform*: bool
  canReuseBuffer*: bool
  buffer*: pointer
  reference*: RootRef

implement CoherenceManager with:
  method `[]`(key: pointer): CoherenceEntry = this.entries[key]

  method `[]=`(key: pointer; value: CoherenceEntry) = this.entries[key] = value

  method ensureEntry(key: pointer) = 
    acquire this.lock
    defer: release this.lock
    if key notin this.entries:
      this[key] = newCoherenceEntry(cskHostDirty)

  method markHostDirty(key: pointer) =
    acquire this.lock
    defer: release this.lock
    if key in this.entries: this[key].state = cskHostDirty

  method hasEntry(key: pointer): bool = key in this.entries
  
  method open(key: pointer; io: IOKind): ViewState =
    ## Called at the start of newTensorFieldView.
    ## Determines whether the AoS→AoSoA transform is needed and whether
    ## we can reuse the last device buffer.
    ##
    ## This is the equivalent of Grid's AcceleratorViewOpen(ptr, mode):
    ##   iokRead      ≈ AcceleratorRead:          if CpuDirty → Clone (H2D), state = Consistent
    ##   iokWrite     ≈ AcceleratorWriteDiscard:   no Clone, state = AccDirty
    ##   iokReadWrite ≈ AcceleratorWrite:          if CpuDirty → Clone (H2D), state = AccDirty
    ##
    ## In our case, "Clone" (H2D copy) means "AoS→AoSoA transform" and
    ## "Flush" (D2H copy) means "AoSoA→AoS scatter".
    acquire this.lock
    defer: release this.lock

    if key notin this.entries:
      this[key] = newCoherenceEntry(cskHostDirty)

    case io:
    of iokRead: # view is opened for reading
      case this[key].state:
      of cskEmpty: # error: nothing to read; treat as dirty
        result.needsTransform = true
        result.canReuseBuffer = false
        this[key].state = cskConsistent
      of cskHostDirty: # host has new data; must transform AoS -> AoSoA
        result.needsTransform = true
        result.canReuseBuffer = (this[key].buffer != nil)
        result.buffer = this[key].buffer
        result.reference = this[key].reference
        this[key].state = cskConsistent
      of cskDeviceDirty: # device is current - reuse directly, no transform
        result.needsTransform = false
        result.canReuseBuffer = true
        result.buffer = this[key].buffer
        result.reference = this[key].reference
        # state stays DeviceDirty (device still has the most recent data)
      of cskConsistent: # data is consistent - reuse directly, no transform
        result.needsTransform = false
        result.canReuseBuffer = true
        result.buffer = this[key].buffer
        result.reference = this[key].reference
      inc this[key].hostLock
    of iokWrite: # view write-only; device will be overwritten, so no transform
      result.needsTransform = false
      result.canReuseBuffer = (this[key].buffer != nil)
      result.buffer = this[key].buffer
      result.reference = this[key].reference
      this[key].state = cskDeviceDirty
      inc this[key].deviceLock
    of iokReadWrite: # view is open for reading & writing
      case this[key].state:
      of cskEmpty, cskHostDirty: # read host data first, then modify device
        result.needsTransform = true
        result.canReuseBuffer = (this[key].buffer != nil)
        result.buffer = this[key].buffer
        result.reference = this[key].reference
      of cskDeviceDirty, cskConsistent: 
        # device buffer is current or both in sync - reuse, no transform
        result.needsTransform = false
        result.canReuseBuffer = true
        result.buffer = this[key].buffer
        result.reference = this[key].reference
      this[key].state = cskDeviceDirty
      inc this[key].deviceLock

  method close(key: pointer; io: IOKind; entry: BufferEntry): bool = 
    ## Called at the end of =destroy.
    ## Returns true if the caller should scatter AoSoA→AoS (write-back).
    ##
    ## Grid's AcceleratorViewClose ONLY decrements accLock and inserts
    ## into the LRU queue — no state change, no flush.  Our close() is
    ## more eager: when the last writer closes, we signal the caller to
    ## scatter immediately and set state = Consistent.  For Grid's lazy
    ## behavior, use closeLazy() instead.
    acquire this.lock
    defer: release this.lock

    if key notin this.entries: return false

    case io:
    of iokRead: # cache device buffer for future reuse
      this[key].hostLock = max(0, this[key].hostLock - 1)
      this[key].buffer = entry.data
      this[key].reference = entry.reference
      this[key].bytes = entry.bytes
      return false # read-only; no write-back
    of iokWrite, iokReadWrite:
      this[key].deviceLock = max(0, this[key].deviceLock - 1)
      this[key].buffer = entry.data
      this[key].reference = entry.reference
      this[key].bytes = entry.bytes

      if this[key].deviceLock == 0: # last writer closing; scatter AoSoA→AoS
        this[key].state = cskConsistent
        return true # last writer closed; caller should scatter AoSoA→AoS
      else: return false # still active writers; defer write-back
  
  method closeLazy(key: pointer; io: IOKind; entry: BufferEntry) =
    ## Lazy variant: cache the buffer but DON'T scatter.
    ## State stays DeviceDirty until the next time host code needs the data.
    ##
    ## This matches Grid's AcceleratorViewClose exactly: only decrement
    ## accLock and insert into LRU — no state change, no flush.  The
    ## flush (scatter) is deferred until a host-side read needs the data,
    ## triggered by markHostDirty() + open(iokRead).
    acquire this.lock
    defer: release this.lock

    if key notin this.entries: return

    case io:
    of iokRead: this[key].hostLock = max(0, this[key].hostLock - 1)
    of iokWrite, iokReadWrite: this[key].deviceLock = max(0, this[key].deviceLock - 1)

    this[key].buffer = entry.data
    this[key].reference = entry.reference
    this[key].bytes = entry.bytes

when isMainModule:
  import std/[unittest]

  # ---- helpers ----

  proc makeEntry(p: pointer, sz: int = 1024): BufferEntry =
    BufferEntry(data: p, reference: nil, bytes: sz, lastUseTick: 0)

  # ---- CoherenceEntry ----

  suite "CoherenceEntry: construction and accessors":
    test "default state is cskEmpty":
      let ce = newCoherenceEntry(cskEmpty)
      check ce.state == cskEmpty
      check ce.hostLock == 0
      check ce.deviceLock == 0

    test "init with explicit state":
      let ce = newCoherenceEntry(cskHostDirty)
      check ce.state == cskHostDirty

    test "buffer/reference/bytes round-trip through setters":
      let ce = newCoherenceEntry(cskEmpty)
      var x: int = 42
      ce.buffer = cast[pointer](addr x)
      check ce.buffer == cast[pointer](addr x)
      ce.bytes = 256
      check ce.bytes == 256

  # ---- CoherenceManager: registration ----

  suite "CoherenceManager: registration":
    test "ensureEntry creates HostDirty entry":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.ensureEntry(key)
      check cm.hasEntry(key)
      check cm[key].state == cskHostDirty

    test "ensureEntry is idempotent — does not overwrite":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.ensureEntry(key)
      cm[key].state = cskConsistent
      cm.ensureEntry(key)
      check cm[key].state == cskConsistent

    test "markHostDirty transitions to HostDirty":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.ensureEntry(key)
      cm[key].state = cskConsistent
      cm.markHostDirty(key)
      check cm[key].state == cskHostDirty

    test "markHostDirty on missing key is no-op":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.markHostDirty(key) # should not crash

  # ---- open iokRead ----

  suite "CoherenceManager: open iokRead":
    test "Empty → needsTransform, state = Consistent":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.ensureEntry(key)
      cm[key].state = cskEmpty
      let vs = cm.open(key, iokRead)
      check vs.needsTransform == true
      check vs.canReuseBuffer == false
      check cm[key].state == cskConsistent
      check cm[key].hostLock == 1

    test "HostDirty, no cached buffer → needsTransform, canReuseBuffer=false":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.ensureEntry(key)
      let vs = cm.open(key, iokRead)
      check vs.needsTransform == true
      check vs.canReuseBuffer == false
      check cm[key].state == cskConsistent
      check cm[key].hostLock == 1

    test "HostDirty, cached buffer → needsTransform, canReuseBuffer=true":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      cm[key].buffer = bufPtr
      let vs = cm.open(key, iokRead)
      check vs.needsTransform == true
      check vs.canReuseBuffer == true
      check vs.buffer == bufPtr
      check cm[key].state == cskConsistent

    test "DeviceDirty → no transform, reuse, state stays DeviceDirty":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      cm[key].state = cskDeviceDirty
      cm[key].buffer = bufPtr
      let vs = cm.open(key, iokRead)
      check vs.needsTransform == false
      check vs.canReuseBuffer == true
      check vs.buffer == bufPtr
      check cm[key].state == cskDeviceDirty

    test "Consistent → no transform, reuse":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      cm[key].state = cskConsistent
      cm[key].buffer = bufPtr
      let vs = cm.open(key, iokRead)
      check vs.needsTransform == false
      check vs.canReuseBuffer == true
      check vs.buffer == bufPtr
      check cm[key].state == cskConsistent

    test "hostLock increments on each read open":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.ensureEntry(key)
      cm[key].state = cskConsistent
      discard cm.open(key, iokRead)
      discard cm.open(key, iokRead)
      check cm[key].hostLock == 2

  # ---- open iokWrite ----

  suite "CoherenceManager: open iokWrite":
    test "any state → DeviceDirty, no transform":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      for startState in [cskEmpty, cskHostDirty, cskDeviceDirty, cskConsistent]:
        cm.ensureEntry(key)
        cm[key].state = startState
        cm[key].deviceLock = 0
        let vs = cm.open(key, iokWrite)
        check vs.needsTransform == false
        check cm[key].state == cskDeviceDirty

    test "deviceLock increments on each write open":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.ensureEntry(key)
      discard cm.open(key, iokWrite)
      discard cm.open(key, iokWrite)
      check cm[key].deviceLock == 2

    test "cached buffer is returned for reuse":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      cm[key].buffer = bufPtr
      let vs = cm.open(key, iokWrite)
      check vs.canReuseBuffer == true
      check vs.buffer == bufPtr

  # ---- open iokReadWrite ----

  suite "CoherenceManager: open iokReadWrite":
    test "HostDirty → needsTransform, state = DeviceDirty":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.ensureEntry(key)
      let vs = cm.open(key, iokReadWrite)
      check vs.needsTransform == true
      check cm[key].state == cskDeviceDirty
      check cm[key].deviceLock == 1

    test "Empty → needsTransform, state = DeviceDirty":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.ensureEntry(key)
      cm[key].state = cskEmpty
      let vs = cm.open(key, iokReadWrite)
      check vs.needsTransform == true
      check cm[key].state == cskDeviceDirty

    test "DeviceDirty → no transform, reuse":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      cm[key].state = cskDeviceDirty
      cm[key].buffer = bufPtr
      let vs = cm.open(key, iokReadWrite)
      check vs.needsTransform == false
      check vs.canReuseBuffer == true
      check vs.buffer == bufPtr
      check cm[key].state == cskDeviceDirty

    test "Consistent → no transform, reuse, state = DeviceDirty":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      cm[key].state = cskConsistent
      cm[key].buffer = bufPtr
      let vs = cm.open(key, iokReadWrite)
      check vs.needsTransform == false
      check vs.canReuseBuffer == true
      check vs.buffer == bufPtr
      check cm[key].state == cskDeviceDirty

    test "deviceLock increments":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.ensureEntry(key)
      discard cm.open(key, iokReadWrite)
      discard cm.open(key, iokReadWrite)
      check cm[key].deviceLock == 2

  # ---- close ----

  suite "CoherenceManager: close":
    test "iokRead → returns false, decrements hostLock, caches buffer":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      cm[key].state = cskConsistent
      discard cm.open(key, iokRead)
      check cm[key].hostLock == 1
      let shouldScatter = cm.close(key, iokRead, makeEntry(bufPtr))
      check shouldScatter == false
      check cm[key].hostLock == 0
      check cm[key].buffer == bufPtr

    test "iokWrite, last writer → returns true, state = Consistent":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      discard cm.open(key, iokWrite)
      check cm[key].deviceLock == 1
      let shouldScatter = cm.close(key, iokWrite, makeEntry(bufPtr))
      check shouldScatter == true
      check cm[key].deviceLock == 0
      check cm[key].state == cskConsistent
      check cm[key].buffer == bufPtr

    test "iokWrite, not last writer → returns false, state stays DeviceDirty":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      discard cm.open(key, iokWrite)
      discard cm.open(key, iokWrite)
      check cm[key].deviceLock == 2
      let shouldScatter = cm.close(key, iokWrite, makeEntry(bufPtr))
      check shouldScatter == false
      check cm[key].deviceLock == 1
      check cm[key].state == cskDeviceDirty

    test "iokReadWrite, last writer → returns true":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      discard cm.open(key, iokReadWrite)
      let shouldScatter = cm.close(key, iokReadWrite, makeEntry(bufPtr))
      check shouldScatter == true
      check cm[key].deviceLock == 0
      check cm[key].state == cskConsistent

    test "iokReadWrite, not last writer → returns false":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      discard cm.open(key, iokReadWrite)
      discard cm.open(key, iokReadWrite)
      check cm[key].deviceLock == 2
      let shouldScatter = cm.close(key, iokReadWrite, makeEntry(bufPtr))
      check shouldScatter == false
      check cm[key].deviceLock == 1

    test "close on missing key → returns false":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      check cm.close(key, iokRead, makeEntry(nil)) == false

    test "hostLock clamped at zero":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.ensureEntry(key)
      discard cm.close(key, iokRead, makeEntry(nil))
      check cm[key].hostLock == 0

    test "deviceLock clamped at zero":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.ensureEntry(key)
      discard cm.close(key, iokWrite, makeEntry(nil))
      check cm[key].deviceLock == 0

  # ---- closeLazy ----

  suite "CoherenceManager: closeLazy":
    test "iokWrite keeps state DeviceDirty, caches buffer":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      discard cm.open(key, iokWrite)
      cm.closeLazy(key, iokWrite, makeEntry(bufPtr))
      check cm[key].deviceLock == 0
      check cm[key].state == cskDeviceDirty
      check cm[key].buffer == bufPtr

    test "iokReadWrite decrements deviceLock":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      discard cm.open(key, iokReadWrite)
      cm.closeLazy(key, iokReadWrite, makeEntry(bufPtr))
      check cm[key].deviceLock == 0

    test "iokRead decrements hostLock":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.ensureEntry(key)
      cm[key].state = cskConsistent
      discard cm.open(key, iokRead)
      cm.closeLazy(key, iokRead, makeEntry(nil))
      check cm[key].hostLock == 0

    test "closeLazy on missing key is no-op":
      var cm = newCoherenceManager()
      var dummy: int = 0
      let key = cast[pointer](addr dummy)
      cm.closeLazy(key, iokWrite, makeEntry(nil)) # should not crash

  # ---- full lifecycle ----

  suite "CoherenceManager: full lifecycle":
    test "write → close → read reuses buffer without transform":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      # open write (auto-creates entry as HostDirty, transitions to DeviceDirty)
      let vs1 = cm.open(key, iokWrite)
      check vs1.needsTransform == false
      # close write → scatter, state = Consistent
      let scatter1 = cm.close(key, iokWrite, makeEntry(bufPtr))
      check scatter1 == true
      check cm[key].state == cskConsistent
      # open read → device buffer is valid, no transform
      let vs2 = cm.open(key, iokRead)
      check vs2.needsTransform == false
      check vs2.canReuseBuffer == true
      check vs2.buffer == bufPtr

    test "write → closeLazy → read: device is dirty, no transform needed":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      discard cm.open(key, iokWrite)
      cm.closeLazy(key, iokWrite, makeEntry(bufPtr))
      check cm[key].state == cskDeviceDirty
      # read: device is dirty (has latest) → no transform, just reuse
      let vs = cm.open(key, iokRead)
      check vs.needsTransform == false
      check vs.canReuseBuffer == true
      check vs.buffer == bufPtr

    test "host mutation invalidates cached device buffer":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      # write + close → Consistent
      discard cm.open(key, iokWrite)
      discard cm.close(key, iokWrite, makeEntry(bufPtr))
      check cm[key].state == cskConsistent
      # host modifies data
      cm.markHostDirty(key)
      check cm[key].state == cskHostDirty
      # read → must re-transform, but can reuse the buffer
      let vs = cm.open(key, iokRead)
      check vs.needsTransform == true
      check vs.canReuseBuffer == true
      check cm[key].state == cskConsistent

    test "multiple readers then writer":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      cm[key].state = cskConsistent
      cm[key].buffer = bufPtr
      # two concurrent readers
      discard cm.open(key, iokRead)
      discard cm.open(key, iokRead)
      check cm[key].hostLock == 2
      # close both readers
      discard cm.close(key, iokRead, makeEntry(bufPtr))
      discard cm.close(key, iokRead, makeEntry(bufPtr))
      check cm[key].hostLock == 0
      # writer opens
      let vs = cm.open(key, iokWrite)
      check vs.needsTransform == false
      check cm[key].state == cskDeviceDirty
      check cm[key].deviceLock == 1

    test "readwrite → close scatters, then read is free":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      # readwrite from HostDirty → needs transform
      let vs1 = cm.open(key, iokReadWrite)
      check vs1.needsTransform == true
      check cm[key].state == cskDeviceDirty
      # close readwrite → scatter
      let scatter = cm.close(key, iokReadWrite, makeEntry(bufPtr))
      check scatter == true
      check cm[key].state == cskConsistent
      # subsequent read → no transform
      let vs2 = cm.open(key, iokRead)
      check vs2.needsTransform == false
      check vs2.canReuseBuffer == true

    test "consecutive writes reuse buffer":
      var cm = newCoherenceManager()
      var dummy: int = 0
      var buf: int = 0
      let key = cast[pointer](addr dummy)
      let bufPtr = cast[pointer](addr buf)
      cm.ensureEntry(key)
      # first write
      discard cm.open(key, iokWrite)
      cm.closeLazy(key, iokWrite, makeEntry(bufPtr))
      check cm[key].state == cskDeviceDirty
      # second write → device dirty, buffer cached, no transform
      let vs = cm.open(key, iokWrite)
      check vs.needsTransform == false
      check vs.canReuseBuffer == true
      check vs.buffer == bufPtr
      check cm[key].state == cskDeviceDirty