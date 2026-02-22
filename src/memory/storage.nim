#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/memory/storage.nim
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

import simdlayout
import utils/[private]
import utils/[complex]
import openmp/[ompbase]

type LocalStorage*[T] = distinct ptr UncheckedArray[T]

#[ host storage facilities ]#

proc `[]`*[T](s: LocalStorage[T], i: int): var T =
  (ptr UncheckedArray[T])(s)[i]

proc `[]=`*[T](s: LocalStorage[T], i: int, val: T) =
  (ptr UncheckedArray[T])(s)[i] = val

#[ layout transformation facilities ]#

#[
  layoutTransformation: AoS (GlobalArrays) → AoSoA (SIMD-vectorized)
  ====================================================================

  Converts a flat, padded, Array-of-Structures buffer produced by
  GlobalArrays into an Array-of-Structures-of-Arrays buffer suitable
  for SIMD execution.  The target groups VectorWidth (e.g. 8) host
  sites into a single "device site" so that one SIMD register load
  fetches the same tensor element across all lanes.

  ─────────────────────────────────────────────────────────────────────
  SOURCE LAYOUT  (GlobalArrays — padded AoS, row-major / C-order)
  ─────────────────────────────────────────────────────────────────────

  GA stores every local lattice site with ghost padding in every
  dimension — both in the lattice (outer) dimensions and in the tensor
  (inner) dimensions.

    paddedGrid  = hostGrid + 2 * latticePadding     (outer)
    paddedShape = shape    + 2 * tensorPadding       (inner)

  Flat index into the source buffer:

    srcIdx = paddedHostLex * paddedElementsPerSite + paddedElemIdx
             ──────┬─────   ──────────┬──────────   ──────┬──────
             site offset     stride per site       element offset

  Example — 2D, hostGrid = [4,4], latticePadding = [1,1],
            scalar real (shape = [1,1]):

    paddedGrid = [6,6]          paddedShape = [3,3]

    Source buffer (36 sites × 9 elems = 324 entries):

       site (0,0)          site (0,1)         ...  site (5,5)
    ┌─────────────┐    ┌─────────────┐           ┌─────────────┐
    │ e0 e1 ... e8│    │ e0 e1 ... e8│    ...    │ e0 e1 ... e8│
    └─────────────┘    └─────────────┘           └─────────────┘
    │<── paddedElementsPerSite = 9 ──>│

    The local (non-ghost) lattice sites live at padded coords
    [1..4, 1..4]; the ghost shell occupies the border.

  The accessPadded() GA call returns a pointer already shifted past the
  inner ghost cells, so element index 0 maps to paddedShape coords
  equal to tensorPadding.  layoutTransformation accounts for this by
  computing:

    paddedElemIdx = elemIdx.lexToCoords(shape).coordsToLex(paddedShape)

  which embeds the unpadded element coordinate into the padded tensor
  grid (skipping the ghost padding implicitly via the coord mapping).

  ─────────────────────────────────────────────────────────────────────
  TARGET LAYOUT  (AoSoA — SIMD-vectorized)
  ─────────────────────────────────────────────────────────────────────

  The target is a 3D array with shape:

    [totalDeviceSlots, elementsPerSite, numSIMDSites]

  where numSIMDSites = VectorWidth (e.g. 8).

  Innermost dimension = SIMD lanes (contiguous in memory), so a single
  aligned load of element `e` from device slot `s` fills one SIMD
  register with that element's value across all VectorWidth host sites:

    target[s, e, 0..7]  →  one SIMD register
                            lane 0   lane 1   ...   lane 7
                            host_a   host_b   ...   host_h

  Example — hostGrid = [4,4], simdGrid = [4,2], deviceGrid = [1,2]:

    Device site 0 (deviceCoords [0,0]):
      lane 0 → host (0,0)    lane 1 → host (0,1)
      lane 2 → host (1,0)    lane 3 → host (1,1)
      lane 4 → host (2,0)    lane 5 → host (2,1)
      lane 6 → host (3,0)    lane 7 → host (3,1)

    Device site 1 (deviceCoords [0,1]):
      lane 0 → host (0,2)    lane 1 → host (0,3)
      lane 2 → host (1,2)    lane 3 → host (1,3)
      lane 4 → host (2,2)    lane 5 → host (2,3)
      lane 6 → host (3,2)    lane 7 → host (3,3)

    hostGrid [4,4]:
    ┌────────┬────────┐
    │(0,0)   │(0,2)   │
    │  (0,1) │  (0,3) │
    │(1,0)   │(1,2)   │
    │  (1,1) │  (1,3) │    device 0 = left half
    │(2,0)   │(2,2)   │    device 1 = right half
    │  (2,1) │  (2,3) │
    │(3,0)   │(3,2)   │
    │  (3,1) │  (3,3) │
    └────────┴────────┘
     device 0  device 1

  Target memory (scalar real, elementsPerSite = 1):

    target[0, 0, 0..7] = values at hosts (0,0)..(3,1)   ← device 0
    target[1, 0, 0..7] = values at hosts (0,2)..(3,3)   ← device 1
    target[2, 0, 0..7] = ghost slot 0 (dim 0 low face)  ← ghost
    ...

  ─────────────────────────────────────────────────────────────────────
  GHOST BOUNDARY SLOTS
  ─────────────────────────────────────────────────────────────────────

  Ghost slots are appended sequentially after the local device sites.
  They inherit the EXACT same SIMD lane decomposition as the boundary
  device site they originate from — each lane's host coordinate is
  simply shifted by ±g in the ghost dimension:

    totalDeviceSlots = numDeviceSites + numGhostSlots

    slots:  [ 0 .. numDeviceSites-1 | numDeviceSites .. totalDeviceSlots-1 ]
             ────── local ──────────  ──────────── ghost ─────────────────

  For each dimension d, for each device site on a boundary face
  (deviceCoords[d] == 0  or  deviceCoords[d] == deviceGrid[d]-1),
  we create latticePadding[d] ghost copies per face:

    low face  (deviceCoords[d] == 0):
      ghost depth g ∈ {1, ..., latticePadding[d]}
      each lane's hostCoords[d] is shifted by -g

    high face (deviceCoords[d] == deviceGrid[d]-1):
      ghost depth g ∈ {1, ..., latticePadding[d]}
      each lane's hostCoords[d] is shifted by +g

  Example — continuing from above (deviceGrid = [1,2], padding = [1,1]):

    dim 0:  deviceGrid[0] = 1, so device sites 0 and 1 are BOTH on the
            low face (coord 0 == 0) AND high face (coord 0 == 0 == 1-1).
            Each spawns 1 ghost copy per face → 2 ghost slots per site,
            4 ghost slots total for dim 0.

    dim 1:  deviceGrid[1] = 2.
            Device 0 (coord 1 == 0) is on the low face  → 1 ghost slot.
            Device 1 (coord 1 == 1) is on the high face → 1 ghost slot.
            2 ghost slots total for dim 1.

    numGhostSlots = 4 + 2 = 6
    totalDeviceSlots = 2 + 6 = 8

    Slot assignment order (d iterates outer, deviceIdx inner):

      slot 2: dim 0, device 0, low  face, g=1 (lanes shifted by -1 in dim 0)
      slot 3: dim 0, device 1, low  face, g=1
      slot 4: dim 0, device 0, high face, g=1 (lanes shifted by +1 in dim 0)
      slot 5: dim 0, device 1, high face, g=1
      slot 6: dim 1, device 0, low  face, g=1 (lanes shifted by -1 in dim 1)
      slot 7: dim 1, device 1, high face, g=1 (lanes shifted by +1 in dim 1)

  The ghost slots land in the GA ghost region of the source buffer
  (the border of paddedGrid), which GA has already filled with data
  from neighboring ranks via GA_Update_ghosts.

  ─────────────────────────────────────────────────────────────────────
  WRITE TEMPLATE
  ─────────────────────────────────────────────────────────────────────

  Both local and ghost slots are written by the same `write` template.
  It takes a target slot index, the originating device site index
  (which determines the lane→host mapping), and a host-space shift
  vector (all zeros for local, ±g·ê_d for ghosts):

    write(slot, deviceIdx, hostShift)

      for each lane:
        hostIdx    = deviceIdxAndLaneIdxToHostIdx(deviceIdx, laneIdx)
        hostCoords = lexToCoords(hostIdx, grid) + hostShift + latticePadding
        paddedHostBase = coordsToLex(hostCoords, paddedGrid) * paddedElementsPerSite
        for each elem:
          src  index = paddedHostBase + embed(elemIdx, shape → paddedShape)
          tgt  index = coordsToLex([slot, elemIdx, laneIdx], reshape)
          target[tgtIdx] = src[srcIdx]

  ─────────────────────────────────────────────────────────────────────
]#

proc layoutTransformation*[D: static[int], R: static[int], S](
  src: LocalStorage[S];
  tensorShape: array[R, int];
  latticePadding: array[D, int];
  simdLayout: SIMDLayout[D];
  T: typedesc;
): LocalStorage[S] =
  let grid = simdLayout.hostGrid
  let innerGhostWidth = 1
  var shape: array[R+1, int]
  var tensorPadding: array[R+1, int]

  for r in 0..R: 
    if r != R: shape[r] = tensorShape[r]
    else: shape[r] = (if isComplex(T): 2 else: 1)
    tensorPadding[r] = innerGhostWidth
  
  let paddedShape = shape + 2 * tensorPadding
  let paddedGrid = grid + 2 * latticePadding

  let numSIMDSites = simdLayout.numSIMDSites
  let numDeviceSites = simdLayout.numDeviceSites
  let elementsPerSite = product(shape)
  let paddedElementsPerSite = product(paddedShape)

  # Ghost slots: for each face boundary device site, duplicate with shifted
  # host coords. Ghost slots inherit the same SIMD lane structure — each lane
  # is shifted by the same offset in the ghost dimension.
  var numGhostSlots = 0
  for d in 0..<D:
    let faceDeviceSites = numDeviceSites div simdLayout.deviceGrid[d]
    numGhostSlots += 2 * faceDeviceSites * latticePadding[d]

  let totalDeviceSlots = numDeviceSites + numGhostSlots
  let reshape = [totalDeviceSlots, elementsPerSite, numSIMDSites]
  let numElements = product(reshape) * sizeof(S)

  var target = cast[LocalStorage[S]](alloc(numElements))

  # Write one slot: map (deviceIdx, laneIdx, elemIdx) → target and src locations.
  template write(slot, deviceIdx: int; hostShift: array[D, int]) =
    for laneIdx in 0..<numSIMDSites:
      let hostIdx = simdLayout.deviceIdxAndLaneIdxToHostIdx(deviceIdx, laneIdx)
      let hostCoords = hostIdx.lexToCoords(grid) + hostShift + latticePadding
      let paddedHostBase = hostCoords.coordsToLex(paddedGrid) * paddedElementsPerSite
      for elemIdx in 0..<elementsPerSite:
        let paddedElemIdx = elemIdx.lexToCoords(shape).coordsToLex(paddedShape)
        let targetIdx = [slot, elemIdx, laneIdx].coordsToLex(reshape)
        target[targetIdx] = src[paddedHostBase + paddedElemIdx]

  threads:
    # local region
    var noShift: array[D, int]
    for deviceIdx in 0..<numDeviceSites: write(deviceIdx, deviceIdx, noShift)

    # ghost boundaries — slots appended after local device sites.
    # Each boundary device site spawns latticePadding[d] ghost copies per face,
    # inheriting the same lane decomposition with host coords shifted by ±g.
    var ghostSlot = numDeviceSites
    for d in 0..<D:
      let signAndBoundary = [(-1, 0), (1, simdLayout.deviceGrid[d] - 1)]
      for deviceIdx in 0..<numDeviceSites:
        let deviceCoords = deviceIdx.lexToCoords(simdLayout.deviceGrid)
        for (sign, boundary) in signAndBoundary:
          if deviceCoords[d] != boundary: continue
          for g in 1..latticePadding[d]:
            var shift: array[D, int]
            shift[d] = sign * g
            write(ghostSlot, deviceIdx, shift)
            ghostSlot += 1

  return target

#[
  inverseLayoutTransformation: AoSoA (SIMD-vectorized) → AoS (GlobalArrays)
  ==========================================================================

  The exact reverse of layoutTransformation.  Reads from the AoSoA target
  buffer and writes back into a padded AoS buffer that GA can consume.

  Only the LOCAL device sites (slots 0 .. numDeviceSites-1) are written
  back — ghost slots are transient copies of neighboring-rank data and
  are never pushed back into the GA buffer.

  Index mapping (inverse of the forward direction):

    for each local deviceIdx:
      for each laneIdx:
        hostIdx       = deviceIdxAndLaneIdxToHostIdx(deviceIdx, laneIdx)
        hostCoords    = lexToCoords(hostIdx, grid) + latticePadding
        paddedHostBase = coordsToLex(hostCoords, paddedGrid) * paddedElementsPerSite
        for each elemIdx:
          srcIdx = coordsToLex([deviceIdx, elemIdx, laneIdx], reshape)
          dstIdx = paddedHostBase + embed(elemIdx, shape → paddedShape)
          dst[dstIdx] = src[srcIdx]

  The destination buffer must be pre-allocated with size
  product(paddedGrid) * product(paddedShape).
]#

proc inverseLayoutTransformation*[D: static[int], R: static[int], S](
  src: LocalStorage[S];
  dst: LocalStorage[S];
  tensorShape: array[R, int];
  latticePadding: array[D, int];
  simdLayout: SIMDLayout[D];
  T: typedesc;
) =
  let grid = simdLayout.hostGrid
  let innerGhostWidth = 1
  var shape: array[R+1, int]
  var tensorPadding: array[R+1, int]

  for r in 0..R:
    if r != R: shape[r] = tensorShape[r]
    else: shape[r] = (if isComplex(T): 2 else: 1)
    tensorPadding[r] = innerGhostWidth

  let paddedShape = shape + 2 * tensorPadding
  let paddedGrid = grid + 2 * latticePadding

  let numSIMDSites = simdLayout.numSIMDSites
  let numDeviceSites = simdLayout.numDeviceSites
  let elementsPerSite = product(shape)
  let paddedElementsPerSite = product(paddedShape)

  var numGhostSlots = 0
  for d in 0..<D:
    let faceDeviceSites = numDeviceSites div simdLayout.deviceGrid[d]
    numGhostSlots += 2 * faceDeviceSites * latticePadding[d]

  let totalDeviceSlots = numDeviceSites + numGhostSlots
  let reshape = [totalDeviceSlots, elementsPerSite, numSIMDSites]

  # Write back local region only — ghost slots are not written back.
  threads:
    for deviceIdx in 0..<numDeviceSites:
      for laneIdx in 0..<numSIMDSites:
        let hostIdx = simdLayout.deviceIdxAndLaneIdxToHostIdx(deviceIdx, laneIdx)
        let hostCoords = hostIdx.lexToCoords(grid) + latticePadding
        let paddedHostBase = hostCoords.coordsToLex(paddedGrid) * paddedElementsPerSite
        for elemIdx in 0..<elementsPerSite:
          let paddedElemIdx = elemIdx.lexToCoords(shape).coordsToLex(paddedShape)
          let srcIdx = [deviceIdx, elemIdx, laneIdx].coordsToLex(reshape)
          dst[paddedHostBase + paddedElemIdx] = src[srcIdx]

when isMainModule:
  import std/[unittest]
  import lattice/[indexing]

  suite "layoutTransformation":

    # Helper: build a padded AoS source buffer.
    # Fill each slot with a unique tag = float64(flatIndex).
    proc makeSource[D, R1: static[int]](
      paddedGrid: array[D, int];
      shape: array[R1, int];
      paddedShape: array[R1, int];
    ): (LocalStorage[float64], int) =
      let paddedElementsPerSite = product(paddedShape)
      let numPaddedHostSites = product(paddedGrid)
      let total = numPaddedHostSites * paddedElementsPerSite
      var buf = cast[LocalStorage[float64]](alloc(total * sizeof(float64)))
      for i in 0..<total:
        buf[i] = float64(i)
      return (buf, total)

    # Helper: compute expected ghost slot count
    proc ghostSlotCount[D: static[int]](
      sl: SIMDLayout[D]; latticePadding: array[D, int]
    ): int =
      for d in 0..<D:
        let faceSites = sl.numDeviceSites div sl.deviceGrid[d]
        result += 2 * faceSites * latticePadding[d]

    # Helper: verify local region of the target
    proc verifyLocal[D, R1: static[int]](
      src, target: LocalStorage[float64];
      sl: SIMDLayout[D];
      hostGrid, latticePadding, paddedGrid: array[D, int];
      shape, paddedShape: array[R1, int];
      reshape: array[3, int];
    ): int =
      let paddedElementsPerSite = product(paddedShape)
      let elementsPerSite = product(shape)
      var checked = 0
      for deviceIdx in 0..<sl.numDeviceSites:
        for laneIdx in 0..<sl.numSIMDSites:
          let hostIdx = sl.deviceIdxAndLaneIdxToHostIdx(deviceIdx, laneIdx)
          let hostCoords = hostIdx.lexToCoords(hostGrid) + latticePadding
          let paddedHostBase = hostCoords.coordsToLex(paddedGrid) * paddedElementsPerSite
          for elemIdx in 0..<elementsPerSite:
            let paddedElemIdx = elemIdx.lexToCoords(shape).coordsToLex(paddedShape)
            let targetIdx = [deviceIdx, elemIdx, laneIdx].coordsToLex(reshape)
            doAssert target[targetIdx] == src[paddedHostBase + paddedElemIdx],
              "local mismatch at device=" & $deviceIdx &
              " lane=" & $laneIdx & " elem=" & $elemIdx
            checked += 1
      return checked

    # Helper: verify ghost region of the target
    proc verifyGhost[D, R1: static[int]](
      src, target: LocalStorage[float64];
      sl: SIMDLayout[D];
      hostGrid, latticePadding, paddedGrid: array[D, int];
      shape, paddedShape: array[R1, int];
      reshape: array[3, int];
    ): int =
      let paddedElementsPerSite = product(paddedShape)
      let elementsPerSite = product(shape)
      var checked = 0
      var ghostSlot = sl.numDeviceSites
      for d in 0..<D:
        for deviceIdx in 0..<sl.numDeviceSites:
          let deviceCoords = deviceIdx.lexToCoords(sl.deviceGrid)
          for (sign, boundary) in [(-1, 0), (1, sl.deviceGrid[d] - 1)]:
            if deviceCoords[d] != boundary: continue
            for g in 1..latticePadding[d]:
              for laneIdx in 0..<sl.numSIMDSites:
                let hostIdx = sl.deviceIdxAndLaneIdxToHostIdx(deviceIdx, laneIdx)
                var hostCoords = hostIdx.lexToCoords(hostGrid)
                hostCoords[d] += sign * g
                let paddedHostCoords = hostCoords + latticePadding
                let paddedHostBase = paddedHostCoords.coordsToLex(paddedGrid) * paddedElementsPerSite
                for elemIdx in 0..<elementsPerSite:
                  let paddedElemIdx = elemIdx.lexToCoords(shape).coordsToLex(paddedShape)
                  let targetIdx = [ghostSlot, elemIdx, laneIdx].coordsToLex(reshape)
                  doAssert target[targetIdx] == src[paddedHostBase + paddedElemIdx],
                    "ghost mismatch at dim=" & $d & " sign=" & $sign & " g=" & $g &
                    " device=" & $deviceIdx & " lane=" & $laneIdx & " elem=" & $elemIdx
                  checked += 1
              ghostSlot += 1
      doAssert ghostSlot == reshape[0],
        "ghost slot count mismatch: " & $ghostSlot & " vs " & $reshape[0]
      return checked

    test "local region: scalar real 2D":
      const D = 2; const R = 1
      let hostGrid: array[D, int] = [4, 4]
      let inputSIMDGrid: array[D, int] = [4, 2]
      let latticePadding: array[D, int] = [1, 1]
      var sl = newSIMDLayout(hostGrid, inputSIMDGrid)
      check sl.simdGrid == [4, 2]
      check sl.deviceGrid == [1, 2]

      let tensorShape: array[R, int] = [1]
      var shape: array[R+1, int] = [1, 1]
      let paddedShape: array[R+1, int] = [3, 3]
      let paddedGrid = hostGrid + 2 * latticePadding
      let (src, _) = makeSource(paddedGrid, shape, paddedShape)
      let target = src.layoutTransformation(tensorShape, latticePadding, sl): float64

      let totalSlots = sl.numDeviceSites + ghostSlotCount(sl, latticePadding)
      let reshape = [totalSlots, product(shape), sl.numSIMDSites]
      let lc = verifyLocal(src, target, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape, reshape)
      check lc == sl.numDeviceSites * sl.numSIMDSites * product(shape)

    test "local + ghost: scalar real 2D":
      const D = 2; const R = 1
      let hostGrid: array[D, int] = [4, 4]
      let inputSIMDGrid: array[D, int] = [4, 2]
      let latticePadding: array[D, int] = [1, 1]
      var sl = newSIMDLayout(hostGrid, inputSIMDGrid)

      let tensorShape: array[R, int] = [1]
      var shape: array[R+1, int] = [1, 1]
      let paddedShape: array[R+1, int] = [3, 3]
      let paddedGrid = hostGrid + 2 * latticePadding
      let (src, _) = makeSource(paddedGrid, shape, paddedShape)
      let target = src.layoutTransformation(tensorShape, latticePadding, sl): float64

      let totalSlots = sl.numDeviceSites + ghostSlotCount(sl, latticePadding)
      let reshape = [totalSlots, product(shape), sl.numSIMDSites]
      let gc = verifyGhost(src, target, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape, reshape)
      check gc > 0

    test "local + ghost: 2x2 real tensor 2D":
      const D = 2; const R = 2
      let hostGrid: array[D, int] = [4, 4]
      let inputSIMDGrid: array[D, int] = [4, 2]
      let latticePadding: array[D, int] = [1, 1]
      var sl = newSIMDLayout(hostGrid, inputSIMDGrid)

      let tensorShape: array[R, int] = [2, 2]
      var shape: array[R+1, int] = [2, 2, 1]
      let paddedShape: array[R+1, int] = [4, 4, 3]
      let paddedGrid = hostGrid + 2 * latticePadding
      let (src, _) = makeSource(paddedGrid, shape, paddedShape)
      let target = src.layoutTransformation(tensorShape, latticePadding, sl): float64

      let totalSlots = sl.numDeviceSites + ghostSlotCount(sl, latticePadding)
      let reshape = [totalSlots, product(shape), sl.numSIMDSites]
      let lc = verifyLocal(src, target, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape, reshape)
      let gc = verifyGhost(src, target, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape, reshape)
      check lc == sl.numDeviceSites * sl.numSIMDSites * product(shape)
      check gc > 0

    test "roundtrip: host coords bijective for local region 2D":
      const D = 2
      let hostGrid: array[D, int] = [4, 4]
      let inputSIMDGrid: array[D, int] = [4, 2]
      var sl = newSIMDLayout(hostGrid, inputSIMDGrid)
      var seen = newSeq[int](sl.numHostSites)
      for deviceIdx in 0..<sl.numDeviceSites:
        for laneIdx in 0..<sl.numSIMDSites:
          let hostIdx = sl.deviceIdxAndLaneIdxToHostIdx(deviceIdx, laneIdx)
          check hostIdx >= 0
          check hostIdx < sl.numHostSites
          seen[hostIdx] += 1
      for i in 0..<sl.numHostSites:
        check seen[i] == 1

    # ========================================================================
    # 4D tests
    # ========================================================================

    test "4D scalar real, auto SIMD grid":
      const D = 4; const R = 1
      let hostGrid: array[D, int] = [4, 4, 4, 8]
      let inputSIMDGrid: array[D, int] = [-1, -1, -1, -1]
      let latticePadding: array[D, int] = [1, 1, 1, 1]
      var sl = newSIMDLayout(hostGrid, inputSIMDGrid)
      check product(sl.simdGrid) == 8

      let tensorShape: array[R, int] = [1]
      var shape: array[R+1, int] = [1, 1]
      let paddedShape: array[R+1, int] = [3, 3]
      let paddedGrid = hostGrid + 2 * latticePadding
      let (src, _) = makeSource(paddedGrid, shape, paddedShape)
      let target = src.layoutTransformation(tensorShape, latticePadding, sl): float64

      let totalSlots = sl.numDeviceSites + ghostSlotCount(sl, latticePadding)
      let reshape = [totalSlots, product(shape), sl.numSIMDSites]
      let lc = verifyLocal(src, target, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape, reshape)
      let gc = verifyGhost(src, target, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape, reshape)
      check lc == sl.numDeviceSites * sl.numSIMDSites * product(shape)
      check gc > 0

    test "4D 3x3 real tensor, simdGrid [2,2,2,1]":
      const D = 4; const R = 2
      let hostGrid: array[D, int] = [4, 4, 4, 4]
      let inputSIMDGrid: array[D, int] = [2, 2, 2, 1]
      let latticePadding: array[D, int] = [1, 1, 1, 1]
      var sl = newSIMDLayout(hostGrid, inputSIMDGrid)
      check sl.simdGrid == [2, 2, 2, 1]
      check sl.deviceGrid == [2, 2, 2, 4]

      let tensorShape: array[R, int] = [3, 3]
      var shape: array[R+1, int] = [3, 3, 1]
      let paddedShape: array[R+1, int] = [5, 5, 3]
      let paddedGrid = hostGrid + 2 * latticePadding
      let (src, _) = makeSource(paddedGrid, shape, paddedShape)
      let target = src.layoutTransformation(tensorShape, latticePadding, sl): float64

      let totalSlots = sl.numDeviceSites + ghostSlotCount(sl, latticePadding)
      let reshape = [totalSlots, product(shape), sl.numSIMDSites]
      let lc = verifyLocal(src, target, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape, reshape)
      let gc = verifyGhost(src, target, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape, reshape)
      check lc == sl.numDeviceSites * sl.numSIMDSites * product(shape)
      check gc > 0

    test "4D 3x3 complex, simdGrid [1,1,2,4]":
      const D = 4; const R = 2
      let hostGrid: array[D, int] = [8, 8, 4, 4]
      let inputSIMDGrid: array[D, int] = [1, 1, 2, 4]
      let latticePadding: array[D, int] = [1, 1, 1, 1]
      var sl = newSIMDLayout(hostGrid, inputSIMDGrid)
      check sl.simdGrid == [1, 1, 2, 4]
      check sl.deviceGrid == [8, 8, 2, 1]

      let tensorShape: array[R, int] = [3, 3]
      var shape: array[R+1, int] = [3, 3, 2]
      let paddedShape: array[R+1, int] = [5, 5, 4]
      let paddedGrid = hostGrid + 2 * latticePadding
      let (src, _) = makeSource(paddedGrid, shape, paddedShape)
      let target = src.layoutTransformation(tensorShape, latticePadding, sl): Complex64

      let totalSlots = sl.numDeviceSites + ghostSlotCount(sl, latticePadding)
      let reshape = [totalSlots, product(shape), sl.numSIMDSites]
      let lc = verifyLocal(src, target, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape, reshape)
      let gc = verifyGhost(src, target, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape, reshape)
      check lc == sl.numDeviceSites * sl.numSIMDSites * product(shape)
      check gc > 0

    test "4D asymmetric ghost widths":
      const D = 4; const R = 1
      let hostGrid: array[D, int] = [4, 4, 4, 8]
      let inputSIMDGrid: array[D, int] = [2, 2, 1, 2]
      let latticePadding: array[D, int] = [1, 2, 1, 3]
      var sl = newSIMDLayout(hostGrid, inputSIMDGrid)
      check sl.simdGrid == [2, 2, 1, 2]
      check sl.deviceGrid == [2, 2, 4, 4]

      let tensorShape: array[R, int] = [1]
      var shape: array[R+1, int] = [1, 1]
      let paddedShape: array[R+1, int] = [3, 3]
      let paddedGrid = hostGrid + 2 * latticePadding
      let (src, _) = makeSource(paddedGrid, shape, paddedShape)
      let target = src.layoutTransformation(tensorShape, latticePadding, sl): float64

      let totalSlots = sl.numDeviceSites + ghostSlotCount(sl, latticePadding)
      let reshape = [totalSlots, product(shape), sl.numSIMDSites]
      let lc = verifyLocal(src, target, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape, reshape)
      let gc = verifyGhost(src, target, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape, reshape)
      check lc == sl.numDeviceSites * sl.numSIMDSites * product(shape)
      check gc > 0

    test "4D bijective roundtrip, multiple configs":
      const D = 4
      let configs = [
        ([4, 4, 4, 8], [2, 2, 1, 2]),
        ([4, 4, 4, 4], [2, 2, 2, 1]),
        ([8, 8, 4, 4], [1, 1, 2, 4]),
        ([8, 4, 4, 4], [4, 1, 1, 2]),
      ]
      for (hg, sg) in configs:
        let hostGrid: array[D, int] = hg
        let inputSIMDGrid: array[D, int] = sg
        var sl = newSIMDLayout(hostGrid, inputSIMDGrid)
        var seen = newSeq[int](sl.numHostSites)
        for deviceIdx in 0..<sl.numDeviceSites:
          for laneIdx in 0..<sl.numSIMDSites:
            let hostIdx = sl.deviceIdxAndLaneIdxToHostIdx(deviceIdx, laneIdx)
            doAssert hostIdx >= 0 and hostIdx < sl.numHostSites,
              "hostIdx out of range for config " & $hostGrid & " / " & $sl.simdGrid
            seen[hostIdx] += 1
        for i in 0..<sl.numHostSites:
          doAssert seen[i] == 1,
            "host site " & $i & " seen " & $seen[i] & " times for config " & $hostGrid

  suite "inverseLayoutTransformation":

    proc makeSource[D, R1: static[int]](
      paddedGrid: array[D, int];
      shape: array[R1, int];
      paddedShape: array[R1, int];
    ): (LocalStorage[float64], int) =
      let paddedElementsPerSite = product(paddedShape)
      let numPaddedHostSites = product(paddedGrid)
      let total = numPaddedHostSites * paddedElementsPerSite
      var buf = cast[LocalStorage[float64]](alloc(total * sizeof(float64)))
      for i in 0..<total:
        buf[i] = float64(i)
      return (buf, total)

    # Helper: verify the local region of dst matches src at every
    # local host site and element, returning the count of checked entries.
    proc verifyInverse[D, R1: static[int]](
      src, dst: LocalStorage[float64];
      sl: SIMDLayout[D];
      hostGrid, latticePadding, paddedGrid: array[D, int];
      shape, paddedShape: array[R1, int];
    ): int =
      let paddedElementsPerSite = product(paddedShape)
      let elementsPerSite = product(shape)
      var checked = 0
      for deviceIdx in 0..<sl.numDeviceSites:
        for laneIdx in 0..<sl.numSIMDSites:
          let hostIdx = sl.deviceIdxAndLaneIdxToHostIdx(deviceIdx, laneIdx)
          let hostCoords = hostIdx.lexToCoords(hostGrid) + latticePadding
          let paddedHostBase = hostCoords.coordsToLex(paddedGrid) * paddedElementsPerSite
          for elemIdx in 0..<elementsPerSite:
            let paddedElemIdx = elemIdx.lexToCoords(shape).coordsToLex(paddedShape)
            let idx = paddedHostBase + paddedElemIdx
            doAssert dst[idx] == src[idx],
              "inverse mismatch at device=" & $deviceIdx &
              " lane=" & $laneIdx & " elem=" & $elemIdx &
              " got=" & $dst[idx] & " expected=" & $src[idx]
            checked += 1
      return checked

    test "roundtrip scalar real 2D":
      const D = 2; const R = 1
      let hostGrid: array[D, int] = [4, 4]
      let inputSIMDGrid: array[D, int] = [4, 2]
      let latticePadding: array[D, int] = [1, 1]
      var sl = newSIMDLayout(hostGrid, inputSIMDGrid)

      let tensorShape: array[R, int] = [1]
      var shape: array[R+1, int] = [1, 1]
      let paddedShape: array[R+1, int] = [3, 3]
      let paddedGrid = hostGrid + 2 * latticePadding
      let (src, total) = makeSource(paddedGrid, shape, paddedShape)

      let aosoa = src.layoutTransformation(tensorShape, latticePadding, sl): float64
      var dst = cast[LocalStorage[float64]](alloc(total * sizeof(float64)))
      aosoa.inverseLayoutTransformation(dst, tensorShape, latticePadding, sl): float64

      let checked = verifyInverse(src, dst, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape)
      check checked == sl.numDeviceSites * sl.numSIMDSites * product(shape)

    test "roundtrip 2x2 real tensor 2D":
      const D = 2; const R = 2
      let hostGrid: array[D, int] = [4, 4]
      let inputSIMDGrid: array[D, int] = [4, 2]
      let latticePadding: array[D, int] = [1, 1]
      var sl = newSIMDLayout(hostGrid, inputSIMDGrid)

      let tensorShape: array[R, int] = [2, 2]
      var shape: array[R+1, int] = [2, 2, 1]
      let paddedShape: array[R+1, int] = [4, 4, 3]
      let paddedGrid = hostGrid + 2 * latticePadding
      let (src, total) = makeSource(paddedGrid, shape, paddedShape)

      let aosoa = src.layoutTransformation(tensorShape, latticePadding, sl): float64
      var dst = cast[LocalStorage[float64]](alloc(total * sizeof(float64)))
      aosoa.inverseLayoutTransformation(dst, tensorShape, latticePadding, sl): float64

      let checked = verifyInverse(src, dst, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape)
      check checked == sl.numDeviceSites * sl.numSIMDSites * product(shape)

    test "roundtrip 4D scalar real, auto SIMD grid":
      const D = 4; const R = 1
      let hostGrid: array[D, int] = [4, 4, 4, 8]
      let inputSIMDGrid: array[D, int] = [-1, -1, -1, -1]
      let latticePadding: array[D, int] = [1, 1, 1, 1]
      var sl = newSIMDLayout(hostGrid, inputSIMDGrid)

      let tensorShape: array[R, int] = [1]
      var shape: array[R+1, int] = [1, 1]
      let paddedShape: array[R+1, int] = [3, 3]
      let paddedGrid = hostGrid + 2 * latticePadding
      let (src, total) = makeSource(paddedGrid, shape, paddedShape)

      let aosoa = src.layoutTransformation(tensorShape, latticePadding, sl): float64
      var dst = cast[LocalStorage[float64]](alloc(total * sizeof(float64)))
      aosoa.inverseLayoutTransformation(dst, tensorShape, latticePadding, sl): float64

      let checked = verifyInverse(src, dst, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape)
      check checked == sl.numDeviceSites * sl.numSIMDSites * product(shape)

    test "roundtrip 4D 3x3 real tensor, simdGrid [2,2,2,1]":
      const D = 4; const R = 2
      let hostGrid: array[D, int] = [4, 4, 4, 4]
      let inputSIMDGrid: array[D, int] = [2, 2, 2, 1]
      let latticePadding: array[D, int] = [1, 1, 1, 1]
      var sl = newSIMDLayout(hostGrid, inputSIMDGrid)

      let tensorShape: array[R, int] = [3, 3]
      var shape: array[R+1, int] = [3, 3, 1]
      let paddedShape: array[R+1, int] = [5, 5, 3]
      let paddedGrid = hostGrid + 2 * latticePadding
      let (src, total) = makeSource(paddedGrid, shape, paddedShape)

      let aosoa = src.layoutTransformation(tensorShape, latticePadding, sl): float64
      var dst = cast[LocalStorage[float64]](alloc(total * sizeof(float64)))
      aosoa.inverseLayoutTransformation(dst, tensorShape, latticePadding, sl): float64

      let checked = verifyInverse(src, dst, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape)
      check checked == sl.numDeviceSites * sl.numSIMDSites * product(shape)

    test "roundtrip 4D 3x3 complex, simdGrid [1,1,2,4]":
      const D = 4; const R = 2
      let hostGrid: array[D, int] = [8, 8, 4, 4]
      let inputSIMDGrid: array[D, int] = [1, 1, 2, 4]
      let latticePadding: array[D, int] = [1, 1, 1, 1]
      var sl = newSIMDLayout(hostGrid, inputSIMDGrid)

      let tensorShape: array[R, int] = [3, 3]
      var shape: array[R+1, int] = [3, 3, 2]
      let paddedShape: array[R+1, int] = [5, 5, 4]
      let paddedGrid = hostGrid + 2 * latticePadding
      let (src, total) = makeSource(paddedGrid, shape, paddedShape)

      let aosoa = src.layoutTransformation(tensorShape, latticePadding, sl): Complex64
      var dst = cast[LocalStorage[float64]](alloc(total * sizeof(float64)))
      aosoa.inverseLayoutTransformation(dst, tensorShape, latticePadding, sl): Complex64

      let checked = verifyInverse(src, dst, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape)
      check checked == sl.numDeviceSites * sl.numSIMDSites * product(shape)

    test "roundtrip 4D asymmetric ghost widths":
      const D = 4; const R = 1
      let hostGrid: array[D, int] = [4, 4, 4, 8]
      let inputSIMDGrid: array[D, int] = [2, 2, 1, 2]
      let latticePadding: array[D, int] = [1, 2, 1, 3]
      var sl = newSIMDLayout(hostGrid, inputSIMDGrid)

      let tensorShape: array[R, int] = [1]
      var shape: array[R+1, int] = [1, 1]
      let paddedShape: array[R+1, int] = [3, 3]
      let paddedGrid = hostGrid + 2 * latticePadding
      let (src, total) = makeSource(paddedGrid, shape, paddedShape)

      let aosoa = src.layoutTransformation(tensorShape, latticePadding, sl): float64
      var dst = cast[LocalStorage[float64]](alloc(total * sizeof(float64)))
      aosoa.inverseLayoutTransformation(dst, tensorShape, latticePadding, sl): float64

      let checked = verifyInverse(src, dst, sl, hostGrid, latticePadding, paddedGrid, shape, paddedShape)
      check checked == sl.numDeviceSites * sl.numSIMDSites * product(shape)