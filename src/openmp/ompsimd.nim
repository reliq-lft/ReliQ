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

## SIMD-Aware AoSoA Layout Transforms for OpenMP Backend
## 
## This module provides:
## - AoS ↔ AoSoA data layout transformations using the SimdLatticeLayout
## - Echo statement detection helper (used by codegen for CPU fallback)
##
## All actual SIMD dispatch is handled by C codegen in ompdisp.nim and
## ompreduce.nim via the simd_intrinsics.h header.

import std/macros

import ompbase
import ompwrap
import ../simd/simdlayout

export ompbase, simdlayout

const VectorWidth* {.intdefine.} = 8

{.passC: "-fopenmp".}
{.passL: "-fopenmp".}

{.emit: """
#include <omp.h>
""".}

#[ ============================================================================
   Echo Statement Detection (for CPU fallback)
   ============================================================================ ]#

proc hasEchoStatement*(n: NimNode): bool =
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
   AoS ↔ AoSoA Transformation with SIMD Layout
   ============================================================================ ]#

proc transformAoStoAoSoASimd*[T](
  src: pointer,
  layout: SimdLatticeLayout,
  elemsPerSite: int
): seq[T] =
  ## Transform data from AoS to AoSoA layout using SIMD layout.
  ##
  ## The AoSoA buffer always uses VectorWidth-wide groups so that the
  ## C codegen kernels (which operate on VW-wide SIMD registers) can
  ## load/store contiguous lanes directly.
  ##
  ## Input AoS:   site0[e0,e1,...], site1[e0,e1,...], ...
  ## Output AoSoA: group0[e0: lane0..laneVW, e1: lane0..laneVW, ...], group1[...]
  let vw = VectorWidth
  let nGroups = (layout.nSites + vw - 1) div vw
  let totalElements = nGroups * vw * elemsPerSite
  let padElements = (64 + sizeof(T) - 1) div sizeof(T)
  result = newSeq[T](totalElements + padElements)
  
  let srcData = cast[ptr UncheckedArray[T]](src)
  
  for outerIdx in 0..<layout.nSitesOuter:
    for lane in 0..<layout.nSitesInner:
      let localSite = outerInnerToLocal(outerIdx, lane, layout)
      for e in 0..<elemsPerSite:
        let aosIdx = localSite * elemsPerSite + e
        let aosoaIdx = outerIdx * (elemsPerSite * vw) + e * vw + lane
        result[aosoaIdx] = srcData[aosIdx]

proc transformAoSoAtoAoSSimd*[T](
  src: pointer,
  layout: SimdLatticeLayout,
  elemsPerSite: int
): seq[T] =
  ## Transform data from AoSoA back to AoS layout using SIMD layout.
  ## Inverse of transformAoStoAoSoASimd.
  let totalElements = layout.nSites * elemsPerSite
  result = newSeq[T](totalElements)
  
  let srcData = cast[ptr UncheckedArray[T]](src)
  let vw = VectorWidth
  
  for outerIdx in 0..<layout.nSitesOuter:
    for lane in 0..<layout.nSitesInner:
      let localSite = outerInnerToLocal(outerIdx, lane, layout)
      for e in 0..<elemsPerSite:
        let aosoaIdx = outerIdx * (elemsPerSite * vw) + e * vw + lane
        let aosIdx = localSite * elemsPerSite + e
        result[aosIdx] = srcData[aosoaIdx]
