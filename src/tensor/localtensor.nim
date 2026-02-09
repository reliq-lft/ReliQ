#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/types/localtensor.nim
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

## LocalTensorField - Rank-Local Host-Memory Tensor Fields
## ==========================================================
##
## This module provides `LocalTensorField[D,R,L,T]`, which represents
## the portion of a distributed `TensorField` that resides in host memory
## on the current MPI rank.  It is created by down-casting a global
## tensor via `newLocalTensorField`.
##
## The ``data`` pointer points **directly** into Global Arrays memory
## obtained via ``NGA_Access``.  Any writes through ``data`` immediately
## modify the underlying GA — no explicit copy-back is needed.  Because
## the GA has ghost-padded strides, a precomputed ``siteOffsets`` lookup
## table maps each lexicographic site index to its flat offset in the
## padded memory so that ``data[siteOffsets[site] + e]`` reaches element
## ``e`` of the given site.
##
## Key operations:
##
## - **Construction**: `newLocalTensorField` obtains a direct pointer to
##   rank-local GA memory and precomputes ``siteOffsets``
## - **Element access**: `[]` for per-site proxy access, `getElement` /
##   `setElement` for per-element reads and writes
## - **Arithmetic**: site-level ``+``, ``-``, ``*``, compound ``+=``, ``-=`` on
##   vector and matrix payloads
## - **Copy / zero**: `copy`, `zero` for bulk operations
## - **Norm**: `norm2` computes the local Euclidean norm
##
## Example
## ^^^^^^^
##
## .. code-block:: nim
##   var field = lat.newTensorField([3, 3]): Complex64
##   var local = field.newLocalTensorField()
##   for n in all 0..<local.numSites():
##     var site = local[n]
##     site[0, 0] = 1.0
##   # writes go directly to GA — no flush required

import lattice
import globaltensor
import sitetensor

import globalarrays/[gatypes, gawrap]
import utils/[private, complex]

when isMainModule:
  import globalarrays/[gampi, gabase]
  import utils/[commandline]
  from lattice/simplecubiclattice import SimpleCubicLattice

type HostStorage*[T] = ptr UncheckedArray[T]

type LocalTensorField*[D: static[int], R: static[int], L: Lattice[D], T] = object
  ## Local tensor field on host memory
  ## 
  ## Represents a local tensor field on host memory defined on a lattice with 
  ## specified dimensions and data type.
  ##
  ## The ``data`` pointer points directly into GA memory via ``NGA_Access``.
  ## Writes through ``data`` immediately modify the underlying Global Array.
  ## Because GA inner dimensions have ghost padding, a ``siteOffsets`` lookup
  ## table maps lexicographic site index → flat offset in padded memory:
  ## ``data[siteOffsets[site] + e]`` reaches element ``e`` of that site.
  lattice*: L
  localGrid*: array[D, int]
  shape*: array[R, int]
  when isComplex32(T): data*: HostStorage[float32]
  elif isComplex64(T): data*: HostStorage[float64]
  else: data*: HostStorage[T]
  hasPadding*: bool
  # --- internal bookkeeping for padded GA strides ---
  siteOffsets*: seq[int]    ## precomputed flat offset for each lex site
  elemsPerSite: int          ## real elements per site (product(shape) * complexFactor)
  nLocalSites: int           ## product of localGrid

proc newLocalTensorField*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T];
  padded: bool = false
): LocalTensorField[D, R, L, T] =
  ## Create a new local tensor field from a global tensor field
  ##
  ## Obtains a direct pointer into rank-local GA memory via ``NGA_Access``
  ## and precomputes a ``siteOffsets`` lookup table that maps each
  ## lexicographic site index to its flat offset in the padded GA memory.
  ## Writes through ``data`` immediately modify the GA — no explicit
  ## copy-back is needed.
  const rank = D + R + 1
  let handle = tensor.data.getHandle()
  var paddedGrid: array[rank, cint]
  var lo, hi: array[rank, cint]
  var ld: array[rank-1, cint]
  var p: pointer
  let pid = GA_Nodeid()

  handle.NGA_Distribution(pid, addr lo[0], addr hi[0])
  if padded: handle.NGA_Access_ghosts(addr paddedGrid[0], addr p, addr ld[0])
  else: 
    handle.NGA_Access(addr lo[0], addr hi[0], addr p, addr ld[0])
    paddedGrid = tensor.data.getLocalGrid().mapTo(cint)

  # Compute local grid dimensions from lo/hi
  var localGrid: array[D, int]
  for i in 0..<D:
    localGrid[i] = int(hi[i] - lo[i] + 1)

  # Compute inner block metrics
  let innerGW = 1  # all inner dims have ghost width 1 (GA requirement)
  let innerPadded = innerPaddedBlockSize(R, innerGW)
  let elemsPerSite = innerBlockSize(R, tensor.shape, isComplex(T))
  
  # Compute number of local sites and padded geometry
  var nLocal = 1
  for d in 0..<D: nLocal *= localGrid[d]
  var paddedGeomD: array[D, int]
  let ghostWidth = tensor.ghostWidth()
  for d in 0..<D:
    paddedGeomD[d] = localGrid[d] + 2 * ghostWidth[d]

  # Precompute site offsets into padded GA memory
  var offsets = newSeq[int](nLocal)
  for site in 0..<nLocal:
    let coords = lexToCoords(site, localGrid)
    offsets[site] = localSiteOffset(coords, paddedGeomD, innerPadded)

  result = LocalTensorField[D, R, L, T](
    lattice: tensor.lattice,
    localGrid: localGrid,
    shape: tensor.shape,
    hasPadding: padded,
    siteOffsets: offsets,
    elemsPerSite: elemsPerSite,
    nLocalSites: nLocal,
  )

  when isComplex32(T): result.data = cast[HostStorage[float32]](p)
  elif isComplex64(T): result.data = cast[HostStorage[float64]](p)
  else: result.data = cast[HostStorage[T]](p)

proc newLocalScalarField*[D: static[int], L, T](
  tensor: TensorField[D, 1, L, T];
  padded: bool = false
): LocalTensorField[D, 0, L, T] =
  ## Create a new local scalar field from a global scalar field
  return tensor.newLocalTensorField(padded)

proc numGlobalSites*[D: static[int], R: static[int], L: Lattice[D], T](
  view: LocalTensorField[D, R, L, T]
): int {.inline.} =
  ## Returns the total number of lattice sites in the tensor field view
  result = 1
  for d in 0..<view.localGrid.len:
    result *= view.localGrid[d]

proc numElements*[D: static[int], R: static[int], L: Lattice[D], T](
  view: LocalTensorField[D, R, L, T]
): int {.inline.} =
  ## Returns the total number of elements (sites * elements per site)
  result = view.numGlobalSites()
  for d in 0..<view.shape.len:
    result *= view.shape[d]

proc getElement*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: LocalTensorField[D, R, L, T], idx: int
): T {.inline.} =
  ## Element access by flat index
  when isComplex64(T):
    complex64(tensor.data[idx * 2], tensor.data[idx * 2 + 1])
  elif isComplex32(T):
    complex32(tensor.data[idx * 2], tensor.data[idx * 2 + 1])
  else:
    tensor.data[idx]

proc setElement*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: var LocalTensorField[D, R, L, T], idx: int, value: T
) {.inline.} =
  ## Element assignment by flat index
  when isComplex64(T):
    tensor.data[idx * 2] = value.re
    tensor.data[idx * 2 + 1] = value.im
  elif isComplex32(T):
    tensor.data[idx * 2] = value.re
    tensor.data[idx * 2 + 1] = value.im
  else:
    tensor.data[idx] = value

#[ ============================================================================
   LocalTensorField Site Proxy Access for "for all" loops
   ============================================================================ ]#

proc numSites*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: LocalTensorField[D, R, L, T]
): int {.inline.} =
  ## Returns the number of sites (for use with "for all" loops)
  tensor.numGlobalSites()

proc tensorElementsPerSite*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: LocalTensorField[D, R, L, T]
): int {.inline.} =
  ## Returns the number of tensor elements per site
  result = 1
  for d in 0..<R:
    result *= tensor.shape[d]

template all*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: LocalTensorField[D, R, L, T]
): untyped =
  ## Returns a range over all sites: ``0 ..< numSites``.
  ## Use with ``reduce`` loops: ``for n in reduce lfield.all:``
  0 ..< tensor.numSites()

proc `[]`*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: LocalTensorField[D, R, L, T], site: int
): LocalSiteProxy[D, R, L, T] {.inline.} =
  ## Get a site proxy for the given site index (for "for all" loops)
  ## siteOffset is stored in storage-type units (e.g. float64 for Complex64)
  result.hostPtr = cast[pointer](tensor.data)
  result.siteOffset = tensor.siteOffsets[site]
  result.shape = tensor.shape
  result.elemsPerSite = tensor.tensorElementsPerSite()

proc `[]=`*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: var LocalTensorField[D, R, L, T], site: int, value: LocalSiteProxy[D, R, L, T]
) {.inline.} =
  ## Copy a site from one proxy to another
  ## siteOffset and element indexing in storage-type units
  const cf = when isComplex32(T) or isComplex64(T): 2 else: 1
  when isComplex64(T):
    let dstData = cast[ptr UncheckedArray[float64]](tensor.data)
    let srcData = cast[ptr UncheckedArray[float64]](value.hostPtr)
  elif isComplex32(T):
    let dstData = cast[ptr UncheckedArray[float32]](tensor.data)
    let srcData = cast[ptr UncheckedArray[float32]](value.hostPtr)
  else:
    let dstData = cast[ptr UncheckedArray[T]](tensor.data)
    let srcData = cast[ptr UncheckedArray[T]](value.hostPtr)
  let elemsPerSite = tensor.tensorElementsPerSite()
  let dstBase = tensor.siteOffsets[site]
  let srcBase = value.siteOffset
  for e in 0..<elemsPerSite * cf:
    dstData[dstBase + e] = srcData[srcBase + e]

proc `[]=`*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: var LocalTensorField[D, R, L, T], site: int, value: LocalAddResult[D, R, L, T]
) {.inline.} =
  ## Site-level addition/subtraction: local[n] = localA[n] + localB[n]
  const cf = when isComplex32(T) or isComplex64(T): 2 else: 1
  when isComplex64(T):
    let dstData = cast[ptr UncheckedArray[float64]](tensor.data)
    let srcAData = cast[ptr UncheckedArray[float64]](value.proxyA.hostPtr)
    let srcBData = cast[ptr UncheckedArray[float64]](value.proxyB.hostPtr)
  elif isComplex32(T):
    let dstData = cast[ptr UncheckedArray[float32]](tensor.data)
    let srcAData = cast[ptr UncheckedArray[float32]](value.proxyA.hostPtr)
    let srcBData = cast[ptr UncheckedArray[float32]](value.proxyB.hostPtr)
  else:
    let dstData = cast[ptr UncheckedArray[T]](tensor.data)
    let srcAData = cast[ptr UncheckedArray[T]](value.proxyA.hostPtr)
    let srcBData = cast[ptr UncheckedArray[T]](value.proxyB.hostPtr)
  let elemsPerSite = tensor.tensorElementsPerSite()
  let dstBase = tensor.siteOffsets[site]
  let srcABase = value.proxyA.siteOffset
  let srcBBase = value.proxyB.siteOffset
  if value.isSubtraction:
    for e in 0..<elemsPerSite * cf:
      dstData[dstBase + e] = srcAData[srcABase + e] - srcBData[srcBBase + e]
  else:
    for e in 0..<elemsPerSite * cf:
      dstData[dstBase + e] = srcAData[srcABase + e] + srcBData[srcBBase + e]

proc `[]=`*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: var LocalTensorField[D, R, L, T], site: int, value: LocalMulResult[D, R, L, T]
) {.inline.} =
  ## Site-level matrix multiplication: local[n] = localA[n] * localB[n]
  let dstProxy = tensor[site]
  
  # Get matrix dimensions from shape
  let rows = tensor.shape[0]
  let cols = if R > 1: tensor.shape[1] else: 1
  let innerDim = if value.proxyA.shape.len > 1: value.proxyA.shape[1] else: 1
  
  for i in 0..<rows:
    for j in 0..<cols:
      var sum: T = default(T)
      for k in 0..<innerDim:
        sum = sum + localProxyGet(value.proxyA, i * innerDim + k) * localProxyGet(value.proxyB, k * cols + j)
      localProxySet(dstProxy, i * cols + j, sum)

proc `[]=`*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: var LocalTensorField[D, R, L, T], site: int, value: LocalScalarMulResult[D, R, L, T]
) {.inline.} =
  ## Site-level scalar multiply: local[n] = scalar * localA[n]
  let dstProxy = tensor[site]
  let elemsPerSite = tensor.tensorElementsPerSite()
  for e in 0..<elemsPerSite:
    localProxySet(dstProxy, e, value.scalar * localProxyGet(value.proxy, e))

proc `[]=`*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: var LocalTensorField[D, R, L, T], site: int, value: LocalScalarAddResult[D, R, L, T]
) {.inline.} =
  ## Site-level scalar add: local[n] = localA[n] + scalar
  let dstProxy = tensor[site]
  let elemsPerSite = tensor.tensorElementsPerSite()
  for e in 0..<elemsPerSite:
    localProxySet(dstProxy, e, localProxyGet(value.proxy, e) + value.scalar)

when isMainModule:
  import ../openmp/omplocal
  import utils/commandline
  import unittest
  
  var argc = cargc()
  var argv = cargv(argc)
  
  discard initMPI(addr argc, addr argv)
  initGA()
  
  suite "LocalTensorField for all loops":
    setup:
      let dims: array[4, int] = [8, 8, 8, 16]
      let lattice = newSimpleCubicLattice(dims)
    
    test "Vector addition with for all loop":
      var tensorA = lattice.newTensorField([3]): float64
      var tensorB = lattice.newTensorField([3]): float64
      var tensorC = lattice.newTensorField([3]): float64
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      # Initialize: A = [1, 2, 3], B = [4, 5, 6]
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 1.0
        localA.data[base + 1] = 2.0
        localA.data[base + 2] = 3.0
        localB.data[base + 0] = 4.0
        localB.data[base + 1] = 5.0
        localB.data[base + 2] = 6.0
      
      # C = A + B using for all loop
      for n in all 0..<localC.numSites():
        localC[n] = localA[n] + localB[n]
      
      # Verify: C = [5, 7, 9]
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 5.0
        check localC.data[base + 1] == 7.0
        check localC.data[base + 2] == 9.0
    
    test "Vector subtraction with for all loop":
      var tensorA = lattice.newTensorField([4]): float64
      var tensorB = lattice.newTensorField([4]): float64
      var tensorC = lattice.newTensorField([4]): float64
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      # Initialize: A = [10, 20, 30, 40], B = [1, 2, 3, 4]
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 10.0
        localA.data[base + 1] = 20.0
        localA.data[base + 2] = 30.0
        localA.data[base + 3] = 40.0
        localB.data[base + 0] = 1.0
        localB.data[base + 1] = 2.0
        localB.data[base + 2] = 3.0
        localB.data[base + 3] = 4.0
      
      # C = A - B using for all loop
      for n in all 0..<localC.numSites():
        localC[n] = localA[n] - localB[n]
      
      # Verify: C = [9, 18, 27, 36]
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 9.0
        check localC.data[base + 1] == 18.0
        check localC.data[base + 2] == 27.0
        check localC.data[base + 3] == 36.0
    
    test "Scalar multiplication with for all loop":
      var tensorA = lattice.newTensorField([3]): float64
      var tensorC = lattice.newTensorField([3]): float64
      
      var localA = tensorA.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      # Initialize: A = [1, 2, 3]
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 1.0
        localA.data[base + 1] = 2.0
        localA.data[base + 2] = 3.0
      
      # C = 2.5 * A using for all loop
      for n in all 0..<localC.numSites():
        localC[n] = 2.5 * localA[n]
      
      # Verify: C = [2.5, 5.0, 7.5]
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 2.5
        check localC.data[base + 1] == 5.0
        check localC.data[base + 2] == 7.5
    
    test "2x2 Matrix multiplication with for all loop":
      var tensorA = lattice.newTensorField([2, 2]): float64
      var tensorB = lattice.newTensorField([2, 2]): float64
      var tensorC = lattice.newTensorField([2, 2]): float64
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      # A = [[1,2],[3,4]], B = [[1,0],[0,1]] (identity)
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 1.0; localA.data[base + 1] = 2.0
        localA.data[base + 2] = 3.0; localA.data[base + 3] = 4.0
        localB.data[base + 0] = 1.0; localB.data[base + 1] = 0.0
        localB.data[base + 2] = 0.0; localB.data[base + 3] = 1.0
      
      # C = A * B using for all loop
      for n in all 0..<localC.numSites():
        localC[n] = localA[n] * localB[n]
      
      # Verify: C = A (multiplied by identity)
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 1.0
        check localC.data[base + 1] == 2.0
        check localC.data[base + 2] == 3.0
        check localC.data[base + 3] == 4.0
    
    test "Element access through LocalSiteProxy":
      var tensorA = lattice.newTensorField([2, 2]): float64
      var localA = tensorA.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      # Set via site proxy element access
      for n in all 0..<localA.numSites():
        var proxy = localA[n]
        proxy[0, 0] = 1.0
        proxy[0, 1] = 2.0
        proxy[1, 0] = 3.0
        proxy[1, 1] = 4.0
      
      # Verify
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        check localA.data[base + 0] == 1.0
        check localA.data[base + 1] == 2.0
        check localA.data[base + 2] == 3.0
        check localA.data[base + 3] == 4.0

    # ================================================================
    # Matrix operations
    # ================================================================

    test "Matrix addition with for all loop":
      var tensorA = lattice.newTensorField([2, 2]): float64
      var tensorB = lattice.newTensorField([2, 2]): float64
      var tensorC = lattice.newTensorField([2, 2]): float64
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 1.0; localA.data[base + 1] = 2.0
        localA.data[base + 2] = 3.0; localA.data[base + 3] = 4.0
        localB.data[base + 0] = 5.0; localB.data[base + 1] = 6.0
        localB.data[base + 2] = 7.0; localB.data[base + 3] = 8.0
      
      # C = A + B
      for n in all 0..<localC.numSites():
        localC[n] = localA[n] + localB[n]
      
      # Verify: C = [[6,8],[10,12]]
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 6.0
        check localC.data[base + 1] == 8.0
        check localC.data[base + 2] == 10.0
        check localC.data[base + 3] == 12.0

    test "Matrix subtraction with for all loop":
      var tensorA = lattice.newTensorField([2, 2]): float64
      var tensorB = lattice.newTensorField([2, 2]): float64
      var tensorC = lattice.newTensorField([2, 2]): float64
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      # A = [[10,20],[30,40]], B = [[1,2],[3,4]]
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 10.0; localA.data[base + 1] = 20.0
        localA.data[base + 2] = 30.0; localA.data[base + 3] = 40.0
        localB.data[base + 0] = 1.0; localB.data[base + 1] = 2.0
        localB.data[base + 2] = 3.0; localB.data[base + 3] = 4.0
      
      # C = A - B
      for n in all 0..<localC.numSites():
        localC[n] = localA[n] - localB[n]
      
      # Verify: C = [[9,18],[27,36]]
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 9.0
        check localC.data[base + 1] == 18.0
        check localC.data[base + 2] == 27.0
        check localC.data[base + 3] == 36.0

    test "3x3 Matrix multiplication":
      # A = [[1,2,3],[4,5,6],[7,8,9]], B = identity
      var tensorA = lattice.newTensorField([3, 3]): float64
      var tensorB = lattice.newTensorField([3, 3]): float64
      var tensorC = lattice.newTensorField([3, 3]): float64
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        # A = [[1,2,3],[4,5,6],[7,8,9]]
        localA.data[base + 0] = 1.0; localA.data[base + 1] = 2.0; localA.data[base + 2] = 3.0
        localA.data[base + 3] = 4.0; localA.data[base + 4] = 5.0; localA.data[base + 5] = 6.0
        localA.data[base + 6] = 7.0; localA.data[base + 7] = 8.0; localA.data[base + 8] = 9.0
        # B = identity
        localB.data[base + 0] = 1.0; localB.data[base + 1] = 0.0; localB.data[base + 2] = 0.0
        localB.data[base + 3] = 0.0; localB.data[base + 4] = 1.0; localB.data[base + 5] = 0.0
        localB.data[base + 6] = 0.0; localB.data[base + 7] = 0.0; localB.data[base + 8] = 1.0
      
      # C = A * B
      for n in all 0..<localC.numSites():
        localC[n] = localA[n] * localB[n]
      
      # Verify: C = A
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 1.0
        check localC.data[base + 1] == 2.0
        check localC.data[base + 2] == 3.0
        check localC.data[base + 3] == 4.0
        check localC.data[base + 4] == 5.0
        check localC.data[base + 5] == 6.0
        check localC.data[base + 6] == 7.0
        check localC.data[base + 7] == 8.0
        check localC.data[base + 8] == 9.0

    test "Non-trivial 2x2 matrix multiplication":
      # A = [[1,2],[3,4]], B = [[2,0],[1,3]] => C = [[4,6],[10,12]]
      var tensorA = lattice.newTensorField([2, 2]): float64
      var tensorB = lattice.newTensorField([2, 2]): float64
      var tensorC = lattice.newTensorField([2, 2]): float64
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 1.0; localA.data[base + 1] = 2.0
        localA.data[base + 2] = 3.0; localA.data[base + 3] = 4.0
        localB.data[base + 0] = 2.0; localB.data[base + 1] = 0.0
        localB.data[base + 2] = 1.0; localB.data[base + 3] = 3.0
      
      # C = A * B
      for n in all 0..<localC.numSites():
        localC[n] = localA[n] * localB[n]
      
      # Verify: C = [[1*2+2*1, 1*0+2*3], [3*2+4*1, 3*0+4*3]] = [[4,6],[10,12]]
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 4.0
        check localC.data[base + 1] == 6.0
        check localC.data[base + 2] == 10.0
        check localC.data[base + 3] == 12.0

    # ================================================================
    # Scalar operations
    # ================================================================

    test "Scalar multiplication on matrix":
      var tensorA = lattice.newTensorField([2, 2]): float64
      var tensorC = lattice.newTensorField([2, 2]): float64
      
      var localA = tensorA.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      # A = [[1,2],[3,4]]
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 1.0; localA.data[base + 1] = 2.0
        localA.data[base + 2] = 3.0; localA.data[base + 3] = 4.0
      
      # C = 3.0 * A
      for n in all 0..<localC.numSites():
        localC[n] = 3.0 * localA[n]
      
      # Verify: C = [[3,6],[9,12]]
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 3.0
        check localC.data[base + 1] == 6.0
        check localC.data[base + 2] == 9.0
        check localC.data[base + 3] == 12.0

    test "Scalar addition to vector":
      var tensorA = lattice.newTensorField([3]): float64
      var tensorC = lattice.newTensorField([3]): float64
      
      var localA = tensorA.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      # A = [1, 2, 3]
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 1.0
        localA.data[base + 1] = 2.0
        localA.data[base + 2] = 3.0
      
      # C = A + 10.0
      for n in all 0..<localC.numSites():
        localC[n] = localA[n] + 10.0
      
      # Verify: C = [11, 12, 13]
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 11.0
        check localC.data[base + 1] == 12.0
        check localC.data[base + 2] == 13.0

    test "Scalar addition to matrix":
      var tensorA = lattice.newTensorField([2, 2]): float64
      var tensorC = lattice.newTensorField([2, 2]): float64
      
      var localA = tensorA.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      # A = [[1,2],[3,4]]
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 1.0; localA.data[base + 1] = 2.0
        localA.data[base + 2] = 3.0; localA.data[base + 3] = 4.0
      
      # C = 5.0 + A
      for n in all 0..<localC.numSites():
        localC[n] = 5.0 + localA[n]
      
      # Verify: C = [[6,7],[8,9]]
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 6.0
        check localC.data[base + 1] == 7.0
        check localC.data[base + 2] == 8.0
        check localC.data[base + 3] == 9.0

    # ================================================================
    # Element-level read/write
    # ================================================================

    test "Vector element read and write":
      var tensorA = lattice.newTensorField([4]): float64
      var localA = tensorA.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      # Write elements
      for n in all 0..<localA.numSites():
        var proxy = localA[n]
        proxy[0] = 10.0
        proxy[1] = 20.0
        proxy[2] = 30.0
        proxy[3] = 40.0
      
      # Verify via direct data access
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        check localA.data[base + 0] == 10.0
        check localA.data[base + 1] == 20.0
        check localA.data[base + 2] == 30.0
        check localA.data[base + 3] == 40.0
      
      # Verify via proxy read
      for n in all 0..<localA.numSites():
        let proxy = localA[n]
        check proxy[0] == 10.0
        check proxy[1] == 20.0
        check proxy[2] == 30.0
        check proxy[3] == 40.0

    test "Matrix element read and write":
      var tensorA = lattice.newTensorField([3, 3]): float64
      var localA = tensorA.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      # Write identity matrix via proxy
      for n in all 0..<localA.numSites():
        var proxy = localA[n]
        for i in 0..<3:
          for j in 0..<3:
            if i == j:
              proxy[i, j] = 1.0
            else:
              proxy[i, j] = 0.0
      
      # Verify via direct data access
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        check localA.data[base + 0] == 1.0  # [0,0]
        check localA.data[base + 1] == 0.0  # [0,1]
        check localA.data[base + 2] == 0.0  # [0,2]
        check localA.data[base + 3] == 0.0  # [1,0]
        check localA.data[base + 4] == 1.0  # [1,1]
        check localA.data[base + 5] == 0.0  # [1,2]
        check localA.data[base + 6] == 0.0  # [2,0]
        check localA.data[base + 7] == 0.0  # [2,1]
        check localA.data[base + 8] == 1.0  # [2,2]

    # ================================================================
    # Print support
    # ================================================================

    test "Print support for matrix with echo in all loop":
      var tensorM = lattice.newTensorField([2, 2]): float64
      var localM = tensorM.newLocalTensorField()
      
      let numSites = localM.numSites()
      
      # Initialize to identity matrix
      for site in 0..<numSites:
        let base = localM.siteOffsets[site]
        localM.data[base + 0] = 1.0
        localM.data[base + 1] = 0.0
        localM.data[base + 2] = 0.0
        localM.data[base + 3] = 1.0
      
      # Print first 2 sites
      for n in all 0..<localM.numSites():
        if n < 2:
          echo "  Site ", n, " matrix:\n", $localM[n]
      
      # Verify data is still correct
      for site in 0..<numSites:
        let base = localM.siteOffsets[site]
        check localM.data[base + 0] == 1.0
        check localM.data[base + 1] == 0.0
        check localM.data[base + 2] == 0.0
        check localM.data[base + 3] == 1.0

    test "Print support for vector":
      var tensorV = lattice.newTensorField([3]): float64
      var localV = tensorV.newLocalTensorField()
      
      let numSites = localV.numSites()
      
      # Initialize vector at each site to [1, 2, 3]
      for site in 0..<numSites:
        let base = localV.siteOffsets[site]
        localV.data[base + 0] = 1.0
        localV.data[base + 1] = 2.0
        localV.data[base + 2] = 3.0
      
      # Print first 2 sites
      for n in all 0..<localV.numSites():
        if n < 2:
          echo "  Site ", n, " vector: ", $localV[n]
      
      # Verify data
      for site in 0..<numSites:
        let base = localV.siteOffsets[site]
        check localV.data[base + 0] == 1.0
        check localV.data[base + 1] == 2.0
        check localV.data[base + 2] == 3.0

    # ================================================================
    # Multi-type tests
    # ================================================================

    test "Float32 vector addition":
      var tensorA = lattice.newTensorField([3]): float32
      var tensorB = lattice.newTensorField([3]): float32
      var tensorC = lattice.newTensorField([3]): float32
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 1.0'f32
        localA.data[base + 1] = 2.0'f32
        localA.data[base + 2] = 3.0'f32
        localB.data[base + 0] = 0.5'f32
        localB.data[base + 1] = 1.0'f32
        localB.data[base + 2] = 1.5'f32
      
      for n in all 0..<localC.numSites():
        localC[n] = localA[n] + localB[n]
      
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 1.5'f32
        check localC.data[base + 1] == 3.0'f32
        check localC.data[base + 2] == 4.5'f32

    test "Float32 scalar multiplication":
      var tensorA = lattice.newTensorField([2]): float32
      var tensorB = lattice.newTensorField([2]): float32
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 2.0'f32
        localA.data[base + 1] = 4.0'f32
      
      for n in all 0..<localB.numSites():
        localB[n] = 2.5'f32 * localA[n]
      
      for site in 0..<numSites:
        let base = localB.siteOffsets[site]
        check localB.data[base + 0] == 5.0'f32
        check localB.data[base + 1] == 10.0'f32

    test "Float32 2x2 matrix multiplication":
      # A = [[1, 2], [3, 4]], B = [[2, 0], [0, 2]] => C = [[2, 4], [6, 8]]
      var tensorA = lattice.newTensorField([2, 2]): float32
      var tensorB = lattice.newTensorField([2, 2]): float32
      var tensorC = lattice.newTensorField([2, 2]): float32
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 1.0'f32; localA.data[base + 1] = 2.0'f32
        localA.data[base + 2] = 3.0'f32; localA.data[base + 3] = 4.0'f32
        localB.data[base + 0] = 2.0'f32; localB.data[base + 1] = 0.0'f32
        localB.data[base + 2] = 0.0'f32; localB.data[base + 3] = 2.0'f32
      
      for n in all 0..<localC.numSites():
        localC[n] = localA[n] * localB[n]
      
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 2.0'f32
        check localC.data[base + 1] == 4.0'f32
        check localC.data[base + 2] == 6.0'f32
        check localC.data[base + 3] == 8.0'f32

    test "Int32 vector addition":
      var tensorA = lattice.newTensorField([3]): int32
      var tensorB = lattice.newTensorField([3]): int32
      var tensorC = lattice.newTensorField([3]): int32
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 10'i32
        localA.data[base + 1] = 20'i32
        localA.data[base + 2] = 30'i32
        localB.data[base + 0] = 5'i32
        localB.data[base + 1] = 15'i32
        localB.data[base + 2] = 25'i32
      
      for n in all 0..<localC.numSites():
        localC[n] = localA[n] + localB[n]
      
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 15'i32
        check localC.data[base + 1] == 35'i32
        check localC.data[base + 2] == 55'i32

    test "Int32 vector subtraction":
      var tensorA = lattice.newTensorField([2]): int32
      var tensorB = lattice.newTensorField([2]): int32
      var tensorC = lattice.newTensorField([2]): int32
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 100'i32
        localA.data[base + 1] = 50'i32
        localB.data[base + 0] = 30'i32
        localB.data[base + 1] = 20'i32
      
      for n in all 0..<localC.numSites():
        localC[n] = localA[n] - localB[n]
      
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == 70'i32
        check localC.data[base + 1] == 30'i32

    test "Int64 vector addition":
      var tensorA = lattice.newTensorField([2]): int64
      var tensorB = lattice.newTensorField([2]): int64
      var tensorC = lattice.newTensorField([2]): int64
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      # Use values larger than int32 can represent
      let bigVal1: int64 = 3_000_000_000'i64
      let bigVal2: int64 = 4_000_000_000'i64
      
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = bigVal1
        localA.data[base + 1] = 100'i64
        localB.data[base + 0] = bigVal2
        localB.data[base + 1] = 200'i64
      
      for n in all 0..<localC.numSites():
        localC[n] = localA[n] + localB[n]
      
      for site in 0..<numSites:
        let base = localC.siteOffsets[site]
        check localC.data[base + 0] == bigVal1 + bigVal2
        check localC.data[base + 1] == 300'i64

    test "Int64 scalar multiplication":
      var tensorA = lattice.newTensorField([2]): int64
      var tensorB = lattice.newTensorField([2]): int64
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 1_000_000'i64
        localA.data[base + 1] = 2_000_000'i64
      
      for n in all 0..<localB.numSites():
        localB[n] = 3'i64 * localA[n]
      
      for site in 0..<numSites:
        let base = localB.siteOffsets[site]
        check localB.data[base + 0] == 3_000_000'i64
        check localB.data[base + 1] == 6_000_000'i64

    # ================================================================
    # Complex operations
    # ================================================================

    test "Chain operation: A*B + C":
      # A = [[1,0],[0,1]], B = [[2,3],[4,5]], C = [[1,1],[1,1]]
      # A*B = [[2,3],[4,5]], A*B + C = [[3,4],[5,6]]
      var tensorA = lattice.newTensorField([2, 2]): float64
      var tensorB = lattice.newTensorField([2, 2]): float64
      var tensorC = lattice.newTensorField([2, 2]): float64
      var tensorR = lattice.newTensorField([2, 2]): float64
      
      var localA = tensorA.newLocalTensorField()
      var localB = tensorB.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      var localR = tensorR.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        # A = identity
        localA.data[base + 0] = 1.0; localA.data[base + 1] = 0.0
        localA.data[base + 2] = 0.0; localA.data[base + 3] = 1.0
        # B = [[2,3],[4,5]]
        localB.data[base + 0] = 2.0; localB.data[base + 1] = 3.0
        localB.data[base + 2] = 4.0; localB.data[base + 3] = 5.0
        # C = ones
        localC.data[base + 0] = 1.0; localC.data[base + 1] = 1.0
        localC.data[base + 2] = 1.0; localC.data[base + 3] = 1.0
      
      # First: temp = A * B
      var tensorTemp = lattice.newTensorField([2, 2]): float64
      var localTemp = tensorTemp.newLocalTensorField()
      
      for n in all 0..<localTemp.numSites():
        localTemp[n] = localA[n] * localB[n]
      
      # Then: R = temp + C
      for n in all 0..<localR.numSites():
        localR[n] = localTemp[n] + localC[n]
      
      # Verify: R = [[3,4],[5,6]]
      for site in 0..<numSites:
        let base = localR.siteOffsets[site]
        check localR.data[base + 0] == 3.0
        check localR.data[base + 1] == 4.0
        check localR.data[base + 2] == 5.0
        check localR.data[base + 3] == 6.0

    test "In-place-style update pattern":
      # Simulate C = C + A pattern using separate output
      var tensorA = lattice.newTensorField([2]): float64
      var tensorC = lattice.newTensorField([2]): float64
      var tensorOut = lattice.newTensorField([2]): float64
      
      var localA = tensorA.newLocalTensorField()
      var localC = tensorC.newLocalTensorField()
      var localOut = tensorOut.newLocalTensorField()
      
      let numSites = localA.numSites()
      
      # A = [10, 20], C = [1, 2]
      for site in 0..<numSites:
        let base = localA.siteOffsets[site]
        localA.data[base + 0] = 10.0
        localA.data[base + 1] = 20.0
        localC.data[base + 0] = 1.0
        localC.data[base + 1] = 2.0
      
      # Out = C + A
      for n in all 0..<localOut.numSites():
        localOut[n] = localC[n] + localA[n]
      
      # Verify: Out = [11, 22]
      for site in 0..<numSites:
        let base = localOut.siteOffsets[site]
        check localOut.data[base + 0] == 11.0
        check localOut.data[base + 1] == 22.0
  
  # All tensor fields destroyed here, safe to finalize
  finalizeGA()
  finalizeMPI()
