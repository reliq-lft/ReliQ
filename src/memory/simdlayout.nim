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

when defined(CPUBuild):
  const VectorWidth* {.intdefine.} = 8
elif defined(GPUBuild):
  when defined(UseOpenCL):
    const VectorWidth* {.intdefine.} = 16
  else:
    const VectorWidth* {.intdefine.} = 32
else: # assume CPU build
  const VectorWidth* {.intdefine.} = 8

import hostlayout

import types/[composite]
import utils/[private]

record SIMDLayout*[D: static[int]]:
  var hostGrid: array[D, int]
  var simdGrid: array[D, int]
  var deviceGrid: array[D, int]

  var hostStrides: array[D, int]
  var simdStrides: array[D, int]
  var deviceStrides: array[D, int]

  var numHostSites: int
  var numDeviceSites: int
  var numSIMDSites: int = VectorWidth

proc defaultSIMDGrid*[D: static[int]]: array[D, int] =
  for i in 0..<D: result[i] = -1

proc newSIMDGrid[D: static[int]](
  hostGrid, inputSIMDGrid: array[D, int]
): array[D, int] =
  ## Compute a SIMD grid that divides the host grid and vector width
  ## 
  ## If any entry < 0, tries to distribute SIMD lanes evenly, so as to 
  ## reduce "surface area" of SIMD blocks. 
  var availableLanes = VectorWidth
  var deviceGrid = hostGrid
  
  # take care of what has already been specified
  for d in 0..<D: 
    result[d] = 1
    if inputSIMDGrid[d] > 0: 
      assert deviceGrid[d] mod inputSIMDGrid[d] == 0, "SIMD grid must divide host grid"
      assert availableLanes mod inputSIMDGrid[d] == 0, "SIMD grid must divide vector width"
      result[d] = inputSIMDGrid[d]
      deviceGrid[d] = deviceGrid[d] div inputSIMDGrid[d]
      availableLanes = availableLanes div inputSIMDGrid[d]

  # take care of anything that has not been specified
  while availableLanes > 1:
    # divide deviceGrid in increments of 2
    for d in countdown(D-1, 0):
      if availableLanes <= 1: break
      let next = (d - 1 + D) mod D
      if inputSIMDGrid[next] < 0 and deviceGrid[next] > deviceGrid[d]: continue
      if deviceGrid[d] mod 2 == 0 and inputSIMDGrid[d] < 0:
        result[d] *= 2
        deviceGrid[d] = deviceGrid[d] div 2
        availableLanes = availableLanes div 2
    
    # check to make sure that we can continue iterating
    var dimensionsAvailable = false
    for d in 0..<D:
      if deviceGrid[d] mod 2 == 0 and inputSIMDGrid[d] < 0: 
        dimensionsAvailable = true
        break
    if (not dimensionsAvailable) and (availableLanes > 1):
      var err = "SIMD grid cannot be determined from input"
      err &= "\n  hostGrid: " & $hostGrid
      err &= "\n  inputSIMDGrid: " & $inputSIMDGrid
      err &= "\n  deviceGrid: " & $deviceGrid
      err &= "\n  result: " & $result
      raise newException(ValueError, err)

implement SIMDLayout with:
  method init(hostGrid, inputSimdGrid: array[D, int]) =
    this.hostGrid = hostGrid
    this.simdGrid = newSIMDGrid(this.hostGrid, inputSimdGrid)
    this.deviceGrid = this.hostGrid div this.simdGrid

    this.hostStrides = strides(this.hostGrid)
    this.simdStrides = strides(this.simdGrid)
    this.deviceStrides = strides(this.deviceGrid)

    this.numHostSites = product(this.hostGrid)
    this.numDeviceSites = product(this.deviceGrid)

  method deviceIdxAndLaneIdxToHostIdx(deviceIdx, laneIdx: int): int {.immutable, inline.} =
    var deviceRemainder = deviceIdx
    var laneRemainder = laneIdx
    result = 0
    for d in 0..<D:
      let laneCoord = laneRemainder div this.simdStrides[d]
      let deviceCoord = deviceRemainder div this.deviceStrides[d]
      laneRemainder = laneRemainder mod this.simdStrides[d]
      deviceRemainder = deviceRemainder mod this.deviceStrides[d]
      result += (deviceCoord * this.simdGrid[d] + laneCoord) * this.hostStrides[d]

  method hostIdxToDeviceIdxAndLaneIdx(hostIdx: int): (int, int) {.immutable, inline.} =
    var deviceIdx = 0
    var laneIdx = 0
    var hostRemainder = hostIdx
    for d in 0..<D:
      let hostCoord = hostRemainder div this.hostStrides[d]
      hostRemainder = hostRemainder mod this.hostStrides[d]
      deviceIdx += (hostCoord div this.simdGrid[d]) * this.deviceStrides[d]
      laneIdx += (hostCoord mod this.simdGrid[d]) * this.simdStrides[d]
    return (deviceIdx, laneIdx)
