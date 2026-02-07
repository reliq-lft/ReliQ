#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/openmp/ompsimd.nim
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

## SIMD-Aware OpenMP Dispatch
## 
## This module provides vectorized loop iteration for OpenMP backend with SIMD.
## It supports the outer-inner loop pattern where:
## - Outer loop iterates over vector groups (OpenMP parallelized)
## - Inner "loop" processes SIMD lanes together
##
## Usage:
##   for outer in eachOuter(layout):
##     # Process all SIMD lanes for this outer index
##     let simdVec = loadSimdVector(data, outer, layout)
##     let result = simdVec * 2.0
##     storeSimdVector(result, data, outer, layout)

import std/macros

import ompbase
import ompwrap
import ../simd/simdlayout
import ../simd/simdtypes

export ompbase, simdlayout, simdtypes

{.passC: "-fopenmp".}
{.passL: "-fopenmp".}

{.emit: """
#include <omp.h>
""".}

#[ ============================================================================
   Echo Statement Detection (for CPU fallback)
   ============================================================================ ]#

proc hasEchoStatement(n: NimNode): bool =
  case n.kind
  of nnkCall:
    if n[0].kind == nnkSym:
      let name = n[0].strVal
      if name in ["echo", "debugEcho"]:
        return true
    for child in n:
      if hasEchoStatement(child):
        return true
  of nnkCommand:
    if n[0].kind == nnkIdent and n[0].strVal == "echo":
      return true
    if n[0].kind == nnkSym and n[0].strVal in ["echo", "debugEcho"]:
      return true
    for child in n:
      if hasEchoStatement(child):
        return true
  else:
    for child in n:
      if hasEchoStatement(child):
        return true
  return false

#[ ============================================================================
   SIMD Outer Loop Iterator
   ============================================================================ ]#

macro eachOuter*(forLoop: ForLoopStmt): untyped =
  ## SIMD-aware outer loop iterator
  ##
  ## Iterates over vector groups (outer indices) with OpenMP parallelization.
  ## Each iteration processes nSitesInner sites via SIMD vectorization.
  ##
  ## Usage:
  ##   for outer in eachOuter(layout.nSitesOuter):
  ##     for lane in 0..<layout.nSitesInner:
  ##       let site = layout.outerInnerToLocal(outer, lane)
  ##       # Process site...
  ##
  ## Or more efficiently with SIMD loads:
  ##   for outer in eachOuter(layout.nSitesOuter):
  ##     let vec = loadSimdVector(data, outer, elemsPerSite, layout)
  ##     # Process vec...
  
  let loopVar = forLoop[0]
  let rangeNode = forLoop[1][1]  # Skip 'eachOuter' wrapper
  let body = forLoop[2]
  
  let needsSerial = hasEchoStatement(body)
  
  if rangeNode.kind == nnkInfix and rangeNode[0].strVal == "..<":
    let startExpr = rangeNode[1]
    let endExpr = rangeNode[2]
    
    if needsSerial:
      result = quote do:
        block:
          for `loopVar` in `startExpr`..<`endExpr`:
            `body`
    else:
      result = quote do:
        block:
          proc loopBody(idx: int64, ctx: pointer) {.cdecl.} =
            let `loopVar` = int(idx)
            `body`
          ompParallelFor(int64(`startExpr`), int64(`endExpr`), loopBody, nil)
  elif rangeNode.kind == nnkCall or rangeNode.kind == nnkDotExpr:
    # Handle layout.nSitesOuter or similar
    if needsSerial:
      result = quote do:
        block:
          for `loopVar` in 0..<`rangeNode`:
            `body`
    else:
      result = quote do:
        block:
          let rangeEnd = `rangeNode`
          proc loopBody(idx: int64, ctx: pointer) {.cdecl.} =
            let `loopVar` = int(idx)
            `body`
          ompParallelFor(0'i64, int64(rangeEnd), loopBody, nil)
  else:
    result = quote do:
      block:
        for `loopVar` in `rangeNode`:
          `body`

#[ ============================================================================
   SIMD Vector Load/Store for AoSoA Layout
   ============================================================================ ]#

proc loadSimdVector*[N: static[int], T](
  data: ptr UncheckedArray[T],
  outerIdx: int,
  elemIdx: int,
  elemsPerSite: int,
  nSitesInner: int
): SimdVec[N, T] {.inline.} =
  ## Load a SIMD vector from AoSoA layout
  ##
  ## Loads all nSitesInner values for a given (outerIdx, elemIdx) pair.
  ## In AoSoA layout, these values are contiguous in memory.
  ##
  ## Parameters:
  ##   data: Pointer to AoSoA data array
  ##   outerIdx: Vector group index
  ##   elemIdx: Element index within tensor (0 to elemsPerSite-1)
  ##   elemsPerSite: Number of elements per site
  ##   nSitesInner: Number of SIMD lanes (must equal N)
  assert N == nSitesInner, "SimdVec width must match nSitesInner"
  
  # In AoSoA: data[outerIdx * (elemsPerSite * nSitesInner) + elemIdx * nSitesInner + lane]
  let baseIdx = outerIdx * (elemsPerSite * nSitesInner) + elemIdx * nSitesInner
  for lane in 0..<N:
    result.data[lane] = data[baseIdx + lane]

proc storeSimdVector*[N: static[int], T](
  vec: SimdVec[N, T],
  data: ptr UncheckedArray[T],
  outerIdx: int,
  elemIdx: int,
  elemsPerSite: int,
  nSitesInner: int
) {.inline.} =
  ## Store a SIMD vector to AoSoA layout
  ##
  ## Stores all nSitesInner values for a given (outerIdx, elemIdx) pair.
  assert N == nSitesInner, "SimdVec width must match nSitesInner"
  
  let baseIdx = outerIdx * (elemsPerSite * nSitesInner) + elemIdx * nSitesInner
  for lane in 0..<N:
    data[baseIdx + lane] = vec.data[lane]

#[ ============================================================================
   Dynamic Width SIMD Load/Store
   ============================================================================ ]#

proc loadSimdVectorDyn*[T](
  data: ptr UncheckedArray[T],
  outerIdx: int,
  elemIdx: int,
  elemsPerSite: int,
  nSitesInner: int
): SimdVecDyn[T] {.inline.} =
  ## Load a dynamic-width SIMD vector from AoSoA layout
  result.width = nSitesInner
  result.data = newSeq[T](nSitesInner)
  
  let baseIdx = outerIdx * (elemsPerSite * nSitesInner) + elemIdx * nSitesInner
  for lane in 0..<nSitesInner:
    result.data[lane] = data[baseIdx + lane]

proc storeSimdVectorDyn*[T](
  vec: SimdVecDyn[T],
  data: ptr UncheckedArray[T],
  outerIdx: int,
  elemIdx: int,
  elemsPerSite: int,
  nSitesInner: int
) {.inline.} =
  ## Store a dynamic-width SIMD vector to AoSoA layout
  assert vec.width == nSitesInner
  
  let baseIdx = outerIdx * (elemsPerSite * nSitesInner) + elemIdx * nSitesInner
  for lane in 0..<nSitesInner:
    data[baseIdx + lane] = vec.data[lane]

#[ ============================================================================
   Tensor Element SIMD Operations
   ============================================================================ ]#

proc loadTensorSimd*[T](
  data: ptr UncheckedArray[T],
  outerIdx: int,
  elemsPerSite: int,
  nSitesInner: int
): seq[SimdVecDyn[T]] {.inline.} =
  ## Load all tensor elements for a vector group as SIMD vectors
  ##
  ## Returns a sequence of SIMD vectors, one per tensor element.
  ## result[elemIdx] contains values for all SIMD lanes of that element.
  result = newSeq[SimdVecDyn[T]](elemsPerSite)
  for e in 0..<elemsPerSite:
    result[e] = loadSimdVectorDyn[T](data, outerIdx, e, elemsPerSite, nSitesInner)

proc storeTensorSimd*[T](
  tensors: seq[SimdVecDyn[T]],
  data: ptr UncheckedArray[T],
  outerIdx: int,
  elemsPerSite: int,
  nSitesInner: int
) {.inline.} =
  ## Store all tensor elements from SIMD vectors
  assert tensors.len == elemsPerSite
  for e in 0..<elemsPerSite:
    storeSimdVectorDyn(tensors[e], data, outerIdx, e, elemsPerSite, nSitesInner)

#[ ============================================================================
   AoS to AoSoA Transformation with SIMD Layout
   ============================================================================ ]#

proc transformAoStoAoSoASimd*[T](
  src: pointer,
  layout: SimdLatticeLayout,
  elemsPerSite: int
): seq[T] =
  ## Transform data from AoS to AoSoA layout using SIMD layout
  ##
  ## Input AoS: site0[e0,e1,...], site1[e0,e1,...], ...
  ## Output AoSoA: outer0[e0: lane0..laneN, e1: lane0..laneN, ...], outer1[...]
  let totalElements = layout.nSites * elemsPerSite
  result = newSeq[T](totalElements)
  
  let srcData = cast[ptr UncheckedArray[T]](src)
  
  for outerIdx in 0..<layout.nSitesOuter:
    for lane in 0..<layout.nSitesInner:
      let localSite = outerInnerToLocal(outerIdx, lane, layout)
      for e in 0..<elemsPerSite:
        let aosIdx = localSite * elemsPerSite + e
        let aosoaIdx = aosoaIndex(outerIdx, lane, e, elemsPerSite, layout.nSitesInner)
        result[aosoaIdx] = srcData[aosIdx]

proc transformAoSoAtoAoSSimd*[T](
  src: pointer,
  layout: SimdLatticeLayout,
  elemsPerSite: int
): seq[T] =
  ## Transform data from AoSoA back to AoS layout using SIMD layout
  let totalElements = layout.nSites * elemsPerSite
  result = newSeq[T](totalElements)
  
  let srcData = cast[ptr UncheckedArray[T]](src)
  
  for outerIdx in 0..<layout.nSitesOuter:
    for lane in 0..<layout.nSitesInner:
      let localSite = outerInnerToLocal(outerIdx, lane, layout)
      for e in 0..<elemsPerSite:
        let aosoaIdx = aosoaIndex(outerIdx, lane, e, elemsPerSite, layout.nSitesInner)
        let aosIdx = localSite * elemsPerSite + e
        result[aosIdx] = srcData[aosoaIdx]

#[ ============================================================================
   High-Level SIMD Iteration Template
   ============================================================================ ]#

template forEachSimd*(layout: SimdLatticeLayout, body: untyped) =
  ## High-level SIMD iteration template
  ##
  ## Iterates over all vector groups with OpenMP parallelization.
  ## Within each iteration, the `outerIdx` variable is available.
  ##
  ## Example:
  ##   forEachSimd(layout):
  ##     for e in 0..<elemsPerSite:
  ##       let vec = loadSimdVectorDyn(data, outerIdx, e, elemsPerSite, layout.nSitesInner)
  ##       let result = 2.0 * vec
  ##       storeSimdVectorDyn(result, data, outerIdx, e, elemsPerSite, layout.nSitesInner)
  block:
    let layoutRef = layout
    proc loopBody(idx: int64, ctx: pointer) {.cdecl.} =
      let outerIdx {.inject.} = int(idx)
      let nSitesInner {.inject.} = layoutRef.nSitesInner
      body
    ompParallelFor(0'i64, int64(layoutRef.nSitesOuter), loopBody, nil)

#[ ============================================================================
   Tests
   ============================================================================ ]#

when isMainModule:
  import std/unittest
  
  suite "SIMD OpenMP Dispatch":
    
    test "AoS to AoSoA transformation":
      # Create a simple 2D layout: [4, 4] with simdGrid [2, 2]
      let layout = newSimdLatticeLayout([4, 4], [2, 2])
      check layout.nSitesInner == 4
      check layout.nSitesOuter == 4
      
      let elemsPerSite = 2
      
      # Create AoS data: site 0 has [0, 100], site 1 has [1, 101], etc.
      var aosData = newSeq[float64](16 * elemsPerSite)
      for site in 0..<16:
        aosData[site * elemsPerSite + 0] = float64(site)
        aosData[site * elemsPerSite + 1] = float64(site + 100)
      
      # Transform to AoSoA
      let aosoaData = transformAoStoAoSoASimd[float64](addr aosData[0], layout, elemsPerSite)
      
      # Transform back to AoS
      let recoveredAoS = transformAoSoAtoAoSSimd[float64](addr aosoaData[0], layout, elemsPerSite)
      
      # Verify round-trip
      for i in 0..<aosData.len:
        check recoveredAoS[i] == aosData[i]
    
    test "SIMD vector load/store":
      let layout = newSimdLatticeLayout([4, 4], [2, 2])
      let elemsPerSite = 3
      let nSitesInner = layout.nSitesInner  # 4
      
      # Create AoSoA data
      var data = newSeq[float64](layout.nSites * elemsPerSite)
      for i in 0..<data.len:
        data[i] = float64(i)
      
      let dataPtr = cast[ptr UncheckedArray[float64]](addr data[0])
      
      # Load a SIMD vector for outer=0, elem=0
      let vec = loadSimdVectorDyn[float64](dataPtr, 0, 0, elemsPerSite, nSitesInner)
      check vec.width == 4
      
      # The first 4 values should be lanes 0-3 of element 0 for outer group 0
      for lane in 0..<4:
        let expectedIdx = 0 * (elemsPerSite * nSitesInner) + 0 * nSitesInner + lane
        check vec.data[lane] == data[expectedIdx]
    
    test "eachOuter macro with simple range":
      var sum: int = 0
      for outer in eachOuter 0..<10:
        {.cast(gcsafe).}:
          discard atomicInc(sum)  # Use atomic since parallel
      
      check sum == 10
    
    test "SIMD arithmetic on loaded vectors":
      let layout = newSimdLatticeLayout([4, 4], [2, 2])
      let elemsPerSite = 1
      let nSitesInner = layout.nSitesInner
      
      # Create source data: 0, 1, 2, ...
      var srcData = newSeq[float64](layout.nSites * elemsPerSite)
      for i in 0..<srcData.len:
        srcData[i] = float64(i)
      
      # Transform to AoSoA
      var aosoaData = transformAoStoAoSoASimd[float64](addr srcData[0], layout, elemsPerSite)
      let dataPtr = cast[ptr UncheckedArray[float64]](addr aosoaData[0])
      
      # Double all values using SIMD
      for outer in 0..<layout.nSitesOuter:
        var vec = loadSimdVectorDyn[float64](dataPtr, outer, 0, elemsPerSite, nSitesInner)
        vec = 2.0 * vec
        storeSimdVectorDyn(vec, dataPtr, outer, 0, elemsPerSite, nSitesInner)
      
      # Transform back to AoS
      let resultAoS = transformAoSoAtoAoSSimd[float64](addr aosoaData[0], layout, elemsPerSite)
      
      # Verify all values are doubled
      for i in 0..<srcData.len:
        check resultAoS[i] == srcData[i] * 2.0
    
    test "forEachSimd template":
      let layout = newSimdLatticeLayout([8, 8], [2, 4])
      check layout.nSitesInner == 8
      check layout.nSitesOuter == 8  # (8/2) * (8/4) = 4 * 2 = 8
      
      let elemsPerSite = 2
      
      # Create and initialize data
      var data = newSeq[float64](layout.nSites * elemsPerSite)
      for i in 0..<data.len:
        data[i] = 1.0
      
      var aosoaData = transformAoStoAoSoASimd[float64](addr data[0], layout, elemsPerSite)
      let dataPtr = cast[ptr UncheckedArray[float64]](addr aosoaData[0])
      
      # Use forEachSimd to add 1.0 to all elements
      forEachSimd(layout):
        for e in 0..<elemsPerSite:
          var vec = loadSimdVectorDyn[float64](dataPtr, outerIdx, e, elemsPerSite, nSitesInner)
          vec = vec + 1.0
          storeSimdVectorDyn(vec, dataPtr, outerIdx, e, elemsPerSite, nSitesInner)
      
      # Transform back
      let resultAoS = transformAoSoAtoAoSSimd[float64](addr aosoaData[0], layout, elemsPerSite)
      
      # All values should be 2.0
      for i in 0..<resultAoS.len:
        check resultAoS[i] == 2.0
