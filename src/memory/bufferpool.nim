#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/memory/bufferpool.nim
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

import std/[tables]
import std/[locks]
import std/[strutils]

import types/[composite]

import opencl/[oclbufferpool]

export oclbufferpool

const
  MaxPoolEntries* = 128
  MaxPoolBytes* = 2 * 1024 * 1024 * 1024 # 2 GB

type
  BufferKey* = object
    numSites*: int        # total sites (including halo)
    elementsPerSite*: int # scalar elements per site (including halo)
    elementSize*: int     # sizeof(typdesc)
  
  BufferEntry* = object
    data*: pointer      # raw pointer to buffer
    reference*: RootRef # prevent garbage collection
    bytes*: int         # allocation size in bytes
    lastUseTick*: int   # for LRU eviction
  
proc bytes*(key: BufferKey): int =
  ## Compute total byte size for a given BufferKey
  return key.numSites * key.elementsPerSite * key.elementSize

class BufferPool:
  ## Global pool of reusable buffers.
  ##
  ## Design rationale:
  ##   Pool intercepts allocations.  On view close, the buffer is returned 
  ##   to the pool (Insert).  On view open, the pool is checked first (Lookup).  
  ##   Only if no matching buffer exists do we allocate.
  ##
  ##   This mirrors Grid's AllocationCache but uses Nim seq + RootRef
  ##   instead of raw malloc/free.
  ## 
  ##   See: https://github.com/paboyle/Grid/tree/develop/Grid/allocator
  var buckets: Table[BufferKey, seq[BufferEntry]]
  var entries: int
  var bytes: int
  var tick: int64
  var lock: Lock

  method init =
    this.buckets = initTable[BufferKey, seq[BufferEntry]]()
    this.entries = 0
    this.bytes = 0
    this.tick = 0
    initLock this.lock
  
  method lookup*(key: BufferKey): (bool, BufferEntry) =
    ## Try to retrieve a buffer from the pool.
    ## Returns (true, entry) if found, (false, default) if not.
    ##
    ## This is Grid's Lookup(size) adapted to Nim:
    ##   Grid uses exact-size matching and pops from a std::multimap.
    ##   We use BufferKey (which includes numSites, elemsPerSite, elementSize)
    ##   and pop from a seq (LIFO for cache warmth).
    acquire this.lock
    defer: release this.lock

    if key in this.buckets:
      var bucket = addr this.buckets[key]
      if bucket[].len > 0:
        let lastEntry = bucket[].pop()
        this.entries -= 1
        this.bytes -= lastEntry.bytes
        return (true, lastEntry)
    
    return (false, BufferEntry())

  method evict*: bool =
    ## Evict the least-recently-used buffer entry.
    ## Returns true if an entry was evicted, false if pool is empty.
    ##
    ## Grid's Victim() does the same: walks the multimap, removes the
    ## entry with the smallest tick, and calls free().  Here we just
    ## drop the reference — Nim's GC frees the underlying seq.
    var oldestTick: int64 = high(int64)
    var oldestKey: BufferKey
    var oldestIdx: int = -1
    var found: bool = false

    for key, bucket in this.buckets:
      for i in 0..<bucket.len:
        if bucket[i].lastUseTick < oldestTick:
          oldestTick = bucket[i].lastUseTick
          oldestKey = key
          oldestIdx = i
          found = true
    
    if not found: return false

    let evicted = this.buckets[oldestKey][oldestIdx]
    this.buckets[oldestKey].delete(oldestIdx)
    if this.buckets[oldestKey].len == 0:
      this.buckets.del(oldestKey)
    this.entries -= 1
    this.bytes -= evicted.bytes
    return true

  method insert*(key: BufferKey; entry: BufferEntry) =
    ## Return a buffer to the pool for reuse.
    ##
    ## If the pool is over budget, evict the oldest entry (LRU).
    ## This is Grid's Insert(ptr, size) + Victim() combined.
    acquire this.lock
    defer: release this.lock

    # Over budget: evict LRU entry
    while (this.entries >= MaxPoolEntries or this.bytes + entry.bytes > MaxPoolBytes):
      if not this.evict(): break 
    
    var bucket = entry
    this.tick += 1
    bucket.lastUseTick = this.tick

    if key notin this.buckets:
      this.buckets[key] = newSeq[BufferEntry]()
    this.buckets[key].add(bucket)
    this.entries += 1
    this.bytes += entry.bytes
  
  method drain* =
    ## Release all pooled buffers.  Call at program shutdown.
    acquire this.lock
    defer: release this.lock

    this.buckets.clear()
    this.entries = 0
    this.bytes = 0
  
  method status*: string =
    ## Return human-readable pool statistics.
    let mbUsed = this.bytes.float / (1024 * 1024).float
    result = "BufferPool: " & $this.entries 
    result &= " entries: " & mbUsed.formatFloat(ffDecimal, 1) & " MB"

when isMainModule:
  import std/[unittest]

  # Helper to make a BufferKey
  proc makeKey(sites, elems, size: int): BufferKey =
    BufferKey(numSites: sites, elementsPerSite: elems, elementSize: size)

  # Helper to make a BufferEntry with a given byte size
  proc makeEntry(bytes: int): BufferEntry =
    BufferEntry(data: nil, reference: nil, bytes: bytes, lastUseTick: 0)

  suite "BufferPool: construction":
    test "new pool has zero entries and bytes":
      var pool = newBufferPool()
      check pool.entries == 0
      check pool.bytes == 0

    test "status string reports zero":
      var pool = newBufferPool()
      check "0 entries" in pool.status()

  suite "BufferPool: insert and lookup":
    test "lookup on empty pool returns (false, _)":
      var pool = newBufferPool()
      let key = makeKey(100, 4, 8)
      let (found, _) = pool.lookup(key)
      check not found

    test "insert then lookup returns the entry":
      var pool = newBufferPool()
      let key = makeKey(100, 4, 8)
      let entry = makeEntry(3200)
      pool.insert(key, entry)
      check pool.entries == 1
      check pool.bytes == 3200

      let (found, got) = pool.lookup(key)
      check found
      check got.bytes == 3200
      check pool.entries == 0
      check pool.bytes == 0

    test "lookup with wrong key returns (false, _)":
      var pool = newBufferPool()
      let key1 = makeKey(100, 4, 8)
      let key2 = makeKey(200, 4, 8)
      pool.insert(key1, makeEntry(1000))

      let (found, _) = pool.lookup(key2)
      check not found
      check pool.entries == 1  # original still there

    test "multiple inserts with same key — LIFO order":
      var pool = newBufferPool()
      let key = makeKey(10, 1, 8)
      pool.insert(key, makeEntry(100))
      pool.insert(key, makeEntry(200))
      pool.insert(key, makeEntry(300))
      check pool.entries == 3

      # LIFO: last inserted (300) comes out first
      let (f1, e1) = pool.lookup(key)
      check f1
      check e1.bytes == 300

      let (f2, e2) = pool.lookup(key)
      check f2
      check e2.bytes == 200

      let (f3, e3) = pool.lookup(key)
      check f3
      check e3.bytes == 100

      # Now empty
      let (f4, _) = pool.lookup(key)
      check not f4

    test "multiple keys are independent":
      var pool = newBufferPool()
      let keyA = makeKey(10, 1, 4)
      let keyB = makeKey(20, 2, 8)
      pool.insert(keyA, makeEntry(100))
      pool.insert(keyB, makeEntry(200))
      check pool.entries == 2
      check pool.bytes == 300

      let (fa, ea) = pool.lookup(keyA)
      check fa
      check ea.bytes == 100

      let (fb, eb) = pool.lookup(keyB)
      check fb
      check eb.bytes == 200

  suite "BufferPool: eviction (LRU)":
    test "evict on empty pool returns false":
      var pool = newBufferPool()
      check not pool.evict()

    test "evict removes the oldest entry":
      var pool = newBufferPool()
      let keyA = makeKey(10, 1, 4)
      let keyB = makeKey(20, 2, 8)
      # Insert A first (tick=1), then B (tick=2)
      pool.insert(keyA, makeEntry(100))
      pool.insert(keyB, makeEntry(200))
      check pool.entries == 2
      check pool.bytes == 300

      # Evict should remove A (oldest tick)
      check pool.evict()
      check pool.entries == 1
      check pool.bytes == 200

      # Only B remains
      let (fa, _) = pool.lookup(keyA)
      check not fa
      let (fb, eb) = pool.lookup(keyB)
      check fb
      check eb.bytes == 200

    test "evict all entries one by one":
      var pool = newBufferPool()
      let key = makeKey(1, 1, 1)
      for i in 1..5:
        pool.insert(key, makeEntry(i * 10))
      check pool.entries == 5

      for i in 1..5:
        check pool.evict()
      check pool.entries == 0
      check pool.bytes == 0
      check not pool.evict()

  suite "BufferPool: drain":
    test "drain clears all entries":
      var pool = newBufferPool()
      let key = makeKey(10, 1, 4)
      for i in 1..10:
        pool.insert(key, makeEntry(100))
      check pool.entries == 10

      pool.drain()
      check pool.entries == 0
      check pool.bytes == 0

      let (found, _) = pool.lookup(key)
      check not found

  suite "BufferPool: bookkeeping":
    test "entries and bytes track correctly through insert/lookup cycle":
      var pool = newBufferPool()
      let key = makeKey(10, 1, 8)
      pool.insert(key, makeEntry(500))
      pool.insert(key, makeEntry(300))
      check pool.entries == 2
      check pool.bytes == 800

      discard pool.lookup(key)
      check pool.entries == 1
      check pool.bytes == 500

      discard pool.lookup(key)
      check pool.entries == 0
      check pool.bytes == 0

    test "tick increments with each insert":
      var pool = newBufferPool()
      let key = makeKey(1, 1, 1)
      pool.insert(key, makeEntry(10))
      pool.insert(key, makeEntry(20))
      pool.insert(key, makeEntry(30))
      check pool.tick == 3
