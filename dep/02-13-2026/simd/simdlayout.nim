#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/simd/simdlayout.nim
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

## SIMD Lattice Layout
## 
## This module provides infrastructure for SIMD-vectorized AoSoA (Array of Structures
## of Arrays) memory layouts for lattice field theory computations.
##
## The key concept is splitting the lattice into:
## - **innerGeom**: SIMD lane grid - sites processed together in one SIMD vector
## - **outerGeom**: Remaining sites - outer loop iterates over these
##
## For example, on a 4D lattice [8,8,8,16] with simdGrid [1,2,2,2]:
## - innerGeom = [1,2,2,2] → nSitesInner = 8 SIMD lanes
## - outerGeom = [8,4,4,8] → nSitesOuter = 1024 outer iterations
## - Total sites = 8 * 1024 = 8192
##
## This layout enables efficient SIMD processing where:
## - Outer loop iterates over vector groups (OpenMP parallelized)
## - Inner "loop" processes VectorWidth sites simultaneously via SIMD
##
## MPI Compatibility:
## - The simdGrid operates on the LOCAL lattice (after MPI partitioning)
## - localGrid = globalGrid / mpiGrid for each dimension
## - simdGrid must evenly divide localGrid in each dimension
## - Example: globalGrid=[16,16,16,32], mpiGrid=[2,2,2,2] → localGrid=[8,8,8,16]
##   Valid simdGrid: [1,2,2,2], [2,2,2,1], [1,1,1,8], etc.
##   Invalid: [4,4,1,1] since 4 doesn't divide 8 and 4 doesn't divide 8 and... wait it does!
##   Actually invalid example: [3,1,1,1] since 3 doesn't divide 8
##
## Reference: QEX (https://github.com/jcosborn/qex) layout implementation

type
  SimdLatticeLayout* = object
    ## SIMD-aware lattice layout for vectorized operations
    ##
    ## Stores the decomposition of a lattice into inner (SIMD) and outer
    ## (iteration) components for efficient vectorized memory access.
    nDim*: int                    ## Number of lattice dimensions
    localGeom*: seq[int]          ## Local lattice dimensions
    simdGrid*: seq[int]           ## SIMD lane grid per dimension
    innerGeom*: seq[int]          ## Inner (SIMD) geometry = simdGrid
    outerGeom*: seq[int]          ## Outer geometry = localGeom / simdGrid
    nSitesInner*: int             ## Total SIMD lanes = product(innerGeom)
    nSitesOuter*: int             ## Outer iterations = product(outerGeom)
    nSites*: int                  ## Total sites = nSitesInner * nSitesOuter
    innerStrides*: seq[int]       ## Strides for inner (lane) indexing
    outerStrides*: seq[int]       ## Strides for outer (group) indexing
    localStrides*: seq[int]       ## Strides for full local indexing

proc computeStrides*(geom: seq[int]): seq[int] =
  ## Compute lexicographic strides for a geometry
  ## stride[d] = product of dimensions 0..<d
  result = newSeq[int](geom.len)
  if geom.len == 0: return
  result[0] = 1
  for d in 1..<geom.len:
    result[d] = result[d-1] * geom[d-1]

proc computeProduct*(geom: seq[int]): int =
  ## Compute product of all dimensions
  result = 1
  for g in geom: result *= g

proc validateSimdGrid*(globalGeom, mpiGrid, simdGrid: openArray[int]): tuple[valid: bool, message: string] =
  ## Validate that simdGrid is compatible with the MPI partitioned lattice
  ##
  ## Parameters:
  ##   globalGeom: Global lattice dimensions (e.g., [16, 16, 16, 32])
  ##   mpiGrid: MPI rank grid (e.g., [2, 2, 2, 2])
  ##   simdGrid: Proposed SIMD lane grid (e.g., [1, 2, 2, 2])
  ##
  ## Returns:
  ##   (valid, message) where valid=true if compatible, false otherwise with error message
  ##
  ## Example:
  ##   let (ok, msg) = validateSimdGrid([16,16,16,32], [2,2,2,2], [1,2,2,2])
  ##   if not ok: echo "Error: ", msg
  if globalGeom.len != mpiGrid.len or globalGeom.len != simdGrid.len:
    return (false, "All grid dimensions must match: globalGeom.len=" & $globalGeom.len & 
            ", mpiGrid.len=" & $mpiGrid.len & ", simdGrid.len=" & $simdGrid.len)
  
  let nDim = globalGeom.len
  for d in 0..<nDim:
    # Check MPI grid divides global grid
    if globalGeom[d] mod mpiGrid[d] != 0:
      return (false, "mpiGrid[" & $d & "]=" & $mpiGrid[d] & 
              " does not evenly divide globalGeom[" & $d & "]=" & $globalGeom[d])
    
    let localSize = globalGeom[d] div mpiGrid[d]
    
    # Check SIMD grid divides local grid
    if localSize mod simdGrid[d] != 0:
      return (false, "simdGrid[" & $d & "]=" & $simdGrid[d] & 
              " does not evenly divide localGeom[" & $d & "]=" & $localSize &
              " (computed from globalGeom[" & $d & "]=" & $globalGeom[d] & 
              " / mpiGrid[" & $d & "]=" & $mpiGrid[d] & ")")
  
  return (true, "SIMD grid is compatible with MPI layout")

proc computeLocalGeom*(globalGeom, mpiGrid: openArray[int]): seq[int] =
  ## Compute local geometry from global geometry and MPI grid
  ##
  ## Parameters:
  ##   globalGeom: Global lattice dimensions
  ##   mpiGrid: MPI rank grid
  ##
  ## Returns:
  ##   Local geometry for each MPI rank: localGeom[d] = globalGeom[d] / mpiGrid[d]
  result = newSeq[int](globalGeom.len)
  for d in 0..<globalGeom.len:
    assert globalGeom[d] mod mpiGrid[d] == 0,
      "mpiGrid[" & $d & "]=" & $mpiGrid[d] & " must evenly divide globalGeom[" & $d & "]=" & $globalGeom[d]
    result[d] = globalGeom[d] div mpiGrid[d]

proc newSimdLatticeLayout*(localGeom: openArray[int], simdGrid: openArray[int]): SimdLatticeLayout =
  ## Create a new SIMD lattice layout
  ##
  ## Parameters:
  ##   localGeom: Local lattice dimensions (e.g., [8, 8, 8, 16])
  ##   simdGrid: SIMD lane grid per dimension (e.g., [1, 2, 2, 2] for 8 lanes)
  ##
  ## The simdGrid must evenly divide localGeom in each dimension.
  ## Total SIMD lanes = product of simdGrid elements.
  ##
  ## Example:
  ##   let layout = newSimdLatticeLayout([8,8,8,16], [1,2,2,2])
  ##   # nSitesInner = 8 (SIMD width)
  ##   # nSitesOuter = 1024 (outer loop iterations)
  assert localGeom.len == simdGrid.len, "localGeom and simdGrid must have same dimensions"
  
  let nDim = localGeom.len
  
  result.nDim = nDim
  result.localGeom = @localGeom
  result.simdGrid = @simdGrid
  result.innerGeom = @simdGrid  # Inner geometry is the SIMD grid
  result.outerGeom = newSeq[int](nDim)
  
  # Compute outer geometry (localGeom / simdGrid)
  for d in 0..<nDim:
    assert localGeom[d] mod simdGrid[d] == 0,
      "simdGrid[" & $d & "]=" & $simdGrid[d] & " must evenly divide localGeom[" & $d & "]=" & $localGeom[d]
    result.outerGeom[d] = localGeom[d] div simdGrid[d]
  
  # Compute site counts
  result.nSitesInner = computeProduct(result.innerGeom)
  result.nSitesOuter = computeProduct(result.outerGeom)
  result.nSites = result.nSitesInner * result.nSitesOuter
  
  # Compute strides
  result.innerStrides = computeStrides(result.innerGeom)
  result.outerStrides = computeStrides(result.outerGeom)
  result.localStrides = computeStrides(result.localGeom)

proc newSimdLatticeLayout*(localGeom: openArray[int], simdWidth: int): SimdLatticeLayout =
  ## Create SIMD layout with automatic lane distribution
  ##
  ## Automatically distributes simdWidth lanes across dimensions,
  ## prioritizing faster-varying (lower) dimensions.
  ##
  ## Example:
  ##   let layout = newSimdLatticeLayout([8,8,8,16], 8)
  ##   # Might produce simdGrid = [2,2,2,1] or [1,1,2,4] depending on divisibility
  let nDim = localGeom.len
  var simdGrid = newSeq[int](nDim)
  for d in 0..<nDim: simdGrid[d] = 1
  
  var remainingLanes = simdWidth
  
  # Distribute lanes across dimensions, starting from fastest-varying
  for d in 0..<nDim:
    if remainingLanes <= 1: break
    
    # Find largest power of 2 that divides both remainingLanes and localGeom[d]
    var lanesFordim = 1
    var candidate = 2
    while candidate <= remainingLanes and candidate <= localGeom[d]:
      if localGeom[d] mod candidate == 0 and remainingLanes mod candidate == 0:
        lanesFordim = candidate
      candidate *= 2
    
    simdGrid[d] = lanesFordim
    remainingLanes = remainingLanes div lanesFordim
  
  assert remainingLanes == 1, "Could not distribute " & $simdWidth & " lanes across lattice"
  
  result = newSimdLatticeLayout(localGeom, simdGrid)

proc lexicographicToCoords*(idx: int, strides: seq[int], geom: seq[int]): seq[int] =
  ## Convert lexicographic index to coordinates
  result = newSeq[int](geom.len)
  var remaining = idx
  for d in countdown(geom.len - 1, 0):
    result[d] = remaining div strides[d]
    remaining = remaining mod strides[d]

proc coordsToLexicographic*(coords: seq[int], strides: seq[int]): int =
  ## Convert coordinates to lexicographic index
  result = 0
  for d in 0..<coords.len:
    result += coords[d] * strides[d]

proc outerInnerToLocal*(outerIdx, innerIdx: int, layout: SimdLatticeLayout): int {.inline.} =
  ## Convert (outerIdx, innerIdx) pair to local site index
  ##
  ## Given an outer index (vector group) and inner index (SIMD lane),
  ## returns the corresponding local site index.
  ##
  ## Local coordinates: localCoord[d] = outerCoord[d] * innerGeom[d] + innerCoord[d]
  var localIdx = 0
  var outerRemaining = outerIdx
  var innerRemaining = innerIdx
  
  for d in countdown(layout.nDim - 1, 0):
    let outerCoord = outerRemaining div layout.outerStrides[d]
    outerRemaining = outerRemaining mod layout.outerStrides[d]
    
    let innerCoord = innerRemaining div layout.innerStrides[d]
    innerRemaining = innerRemaining mod layout.innerStrides[d]
    
    let localCoord = outerCoord * layout.innerGeom[d] + innerCoord
    localIdx += localCoord * layout.localStrides[d]
  
  result = localIdx

proc localToOuterInner*(localIdx: int, layout: SimdLatticeLayout): tuple[outer, inner: int] {.inline.} =
  ## Convert local site index to (outerIdx, innerIdx) pair
  ##
  ## Inverse of outerInnerToLocal.
  var outerIdx = 0
  var innerIdx = 0
  var remaining = localIdx
  
  for d in countdown(layout.nDim - 1, 0):
    let localCoord = remaining div layout.localStrides[d]
    remaining = remaining mod layout.localStrides[d]
    
    let outerCoord = localCoord div layout.innerGeom[d]
    let innerCoord = localCoord mod layout.innerGeom[d]
    
    outerIdx += outerCoord * layout.outerStrides[d]
    innerIdx += innerCoord * layout.innerStrides[d]
  
  result = (outerIdx, innerIdx)

proc simdLanes*(layout: SimdLatticeLayout): int {.inline.} =
  ## Return the number of SIMD lanes (sites per vector group)
  layout.nSitesInner

proc vectorGroups*(layout: SimdLatticeLayout): int {.inline.} =
  ## Return the number of vector groups (outer loop iterations)
  layout.nSitesOuter

proc `$`*(layout: SimdLatticeLayout): string =
  ## String representation of SIMD layout
  result = "SimdLatticeLayout:\n"
  result &= "  localGeom: " & $layout.localGeom & "\n"
  result &= "  simdGrid: " & $layout.simdGrid & "\n"
  result &= "  innerGeom: " & $layout.innerGeom & " (nSitesInner=" & $layout.nSitesInner & ")\n"
  result &= "  outerGeom: " & $layout.outerGeom & " (nSitesOuter=" & $layout.nSitesOuter & ")\n"
  result &= "  totalSites: " & $layout.nSites

#[ ============================================================================
   AoSoA Index Computation for Vectorized Layout
   ============================================================================ ]#

proc aosoaIndex*(outerIdx, innerIdx, elemIdx, elemsPerSite, nSitesInner: int): int {.inline.} =
  ## Compute AoSoA memory index for vectorized layout
  ##
  ## AoSoA layout: [outerIdx][elemIdx][innerIdx]
  ## - outerIdx: vector group index (0 to nSitesOuter-1)
  ## - innerIdx: SIMD lane (0 to nSitesInner-1)
  ## - elemIdx: element within tensor (0 to elemsPerSite-1)
  ##
  ## Memory: group0[e0: lane0,lane1,..., e1: lane0,lane1,...], group1[...]
  outerIdx * (elemsPerSite * nSitesInner) + elemIdx * nSitesInner + innerIdx

proc aosoaIndexFromLocal*(localIdx, elemIdx, elemsPerSite: int, layout: SimdLatticeLayout): int {.inline.} =
  ## Compute AoSoA memory index from local site index
  let (outerIdx, innerIdx) = localToOuterInner(localIdx, layout)
  aosoaIndex(outerIdx, innerIdx, elemIdx, elemsPerSite, layout.nSitesInner)

#[ ============================================================================
   Coordinate Table Generation
   ============================================================================ ]#

proc generateCoordTable*(layout: SimdLatticeLayout): seq[seq[int]] =
  ## Generate coordinate lookup table: coordTable[outerIdx][lane] = localSiteIdx
  ##
  ## Pre-computes the mapping from (outerIdx, lane) to local site index
  ## for efficient vectorized iteration.
  result = newSeq[seq[int]](layout.nSitesOuter)
  for outer in 0..<layout.nSitesOuter:
    result[outer] = newSeq[int](layout.nSitesInner)
    for lane in 0..<layout.nSitesInner:
      result[outer][lane] = outerInnerToLocal(outer, lane, layout)

#[ ============================================================================
   Tests
   ============================================================================ ]#

when isMainModule:
  import std/unittest
  import std/strutils
  
  suite "SimdLatticeLayout":
    
    test "Basic 4D layout with explicit simdGrid":
      let layout = newSimdLatticeLayout([8, 8, 8, 16], [1, 2, 2, 2])
      
      check layout.nDim == 4
      check layout.nSitesInner == 8  # 1*2*2*2
      check layout.nSitesOuter == 1024  # 8*4*4*8
      check layout.nSites == 8192
      check layout.innerGeom == @[1, 2, 2, 2]
      check layout.outerGeom == @[8, 4, 4, 8]
    
    test "Auto-distributed simdWidth":
      let layout = newSimdLatticeLayout([8, 8, 8, 16], 8)
      
      check layout.nSitesInner == 8
      check layout.nSites == 8 * 8 * 8 * 16
      echo "Auto simdGrid: ", layout.simdGrid
    
    test "Outer-inner to local conversion":
      let layout = newSimdLatticeLayout([4, 4], [2, 2])
      
      # Total 16 sites, 4 inner, 4 outer
      check layout.nSitesInner == 4
      check layout.nSitesOuter == 4
      
      # Test round-trip for all sites
      for localIdx in 0..<16:
        let (outer, inner) = localToOuterInner(localIdx, layout)
        let recovered = outerInnerToLocal(outer, inner, layout)
        check recovered == localIdx
    
    test "AoSoA index computation":
      let layout = newSimdLatticeLayout([4, 4], [2, 2])
      let elemsPerSite = 3  # e.g., 3-element vector at each site
      
      # For outer=0, inner=0, elem=0 → index should be 0
      check aosoaIndex(0, 0, 0, elemsPerSite, layout.nSitesInner) == 0
      
      # For outer=0, inner=1, elem=0 → index should be 1
      check aosoaIndex(0, 1, 0, elemsPerSite, layout.nSitesInner) == 1
      
      # For outer=0, inner=0, elem=1 → index should be 4 (after all lanes of elem 0)
      check aosoaIndex(0, 0, 1, elemsPerSite, layout.nSitesInner) == 4
      
      # For outer=1, inner=0, elem=0 → index should be 12 (after all elems of outer 0)
      check aosoaIndex(1, 0, 0, elemsPerSite, layout.nSitesInner) == 12
    
    test "Coordinate table generation":
      let layout = newSimdLatticeLayout([4, 4], [2, 2])
      let coordTable = generateCoordTable(layout)
      
      check coordTable.len == layout.nSitesOuter
      check coordTable[0].len == layout.nSitesInner
      
      # Verify all sites are covered exactly once
      var covered = newSeq[bool](layout.nSites)
      for outer in 0..<layout.nSitesOuter:
        for lane in 0..<layout.nSitesInner:
          let localIdx = coordTable[outer][lane]
          check not covered[localIdx]  # Site should not be covered twice
          covered[localIdx] = true
      
      for localIdx in 0..<layout.nSites:
        check covered[localIdx]  # Every site should be covered
    
    test "String representation":
      let layout = newSimdLatticeLayout([8, 8, 8, 16], [1, 2, 2, 2])
      let s = $layout
      check "nSitesInner=8" in s
      check "nSitesOuter=1024" in s
    
    test "1D layout":
      let layout = newSimdLatticeLayout([16], [4])
      check layout.nSitesInner == 4
      check layout.nSitesOuter == 4
      check layout.nSites == 16
    
    test "Non-power-of-2 dimensions":
      let layout = newSimdLatticeLayout([12, 12], [3, 2])
      check layout.nSitesInner == 6
      check layout.nSitesOuter == 24  # 4*6
      check layout.nSites == 144
    
    test "validateSimdGrid - valid configuration":
      # globalGrid=[16,16,16,32], mpiGrid=[2,2,2,2] → localGrid=[8,8,8,16]
      # simdGrid=[1,2,2,2] divides [8,8,8,16] evenly
      let (valid, msg) = validateSimdGrid([16,16,16,32], [2,2,2,2], [1,2,2,2])
      check valid == true
      echo "Valid: ", msg
    
    test "validateSimdGrid - invalid simdGrid doesn't divide localGrid":
      # globalGrid=[16,16,16,32], mpiGrid=[2,2,2,2] → localGrid=[8,8,8,16]
      # simdGrid=[3,1,1,1] - 3 doesn't divide 8
      let (valid, msg) = validateSimdGrid([16,16,16,32], [2,2,2,2], [3,1,1,1])
      check valid == false
      check "simdGrid[0]" in msg
      check "does not evenly divide" in msg
      echo "Invalid: ", msg
    
    test "validateSimdGrid - mismatched dimensions":
      let (valid, msg) = validateSimdGrid([16,16,16], [2,2,2,2], [1,2,2,2])
      check valid == false
      check "dimensions must match" in msg
    
    test "validateSimdGrid - mpiGrid doesn't divide globalGrid":
      # 3 doesn't divide 16
      let (valid, msg) = validateSimdGrid([16,16,16,32], [3,2,2,2], [1,2,2,2])
      check valid == false
      check "mpiGrid[0]" in msg
    
    test "computeLocalGeom - basic":
      let local = computeLocalGeom([16,16,16,32], [2,2,2,2])
      check local == @[8, 8, 8, 16]
    
    test "computeLocalGeom - asymmetric":
      let local = computeLocalGeom([32,16,8,64], [4,2,1,8])
      check local == @[8, 8, 8, 8]
