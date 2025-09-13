#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/lattice/simplecubiclattice.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  Notes:
  * What kind of gain (if any) could be achieved by considering inter-node
  connectivity in partitioning of lattice?

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of chadge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, medge, publish, distribute, sublicense, and/or sell
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

import backend
import runtime
import utils
import latticeconcept

# import backend header files
backend: discard

#[ frontend: simple cubic lattice type definition ]#

type
  # simple cubic block dimensions
  SimpleCubicDims = tuple[start, stop, size: GeometryType]

  # root object for simple cubic lattice types
  SimpleCubicLatticeRoot {.inheritable.} = object of RootObj
    geometry, partition, strides: seq[GeometryType]
    coordinate: seq[GeometryType]
    dimensions: seq[SimpleCubicDims]
    numSites: GeometryType

  # shared memory simple cubic Bravais (sub)lattice
  SharedSimpleCubicLattice = object of SimpleCubicLatticeRoot
    vecLaneLattices: seq[SimpleCubicLatticeRoot]

  # distributed memory simple cubic Bravais (sub)lattice
  DistributedSimpleCubicLattice* = object of SimpleCubicLatticeRoot
    globalGeometry: seq[GeometryType]
    numGlobalSites: GeometryType
    shrdMemLattices: seq[SharedSimpleCubicLattice]
  
  # distributed memory simple cubic Bravais (sub)lattice with padding
  PaddedDistributedSimpleCubicLattice* = object of SimpleCubicLatticeRoot
    distLattice: DistributedSimpleCubicLattice
    padding: seq[GeometryType]
  
  # distributed simple cubic lattices
  SimpleCubicLattice* = GlobalPointer[DistributedSimpleCubicLattice]
  PaddedSimpleCubicLattice* = GlobalPointer[PaddedDistributedSimpleCubicLattice]

# idea:
# * you can have padding be allocated dynamically, so that padded lattices
#   wrap unpadded lattices; in this way, the padding is treated as a computational
#   tool and not a fundamental part of the lattice structure

#[ backend: types for exception handling ]#

# enumerate possible errors that a user may run into
type LatticeInitializationErrors = enum 
  LatticeSubdivisionError,
  IncompatibleDistMemoryGeometryError,
  BadDistMemoryGeometrySpecificationError,
  IncompatibleShrdMemoryGeometryError,
  BadShrdMemoryGeometrySpecificationError,
  IncompatibleVectorLaneGeometryError,
  BadVectorLaneGeometrySpecificationError

# special error type for handling exception during lattice initialization
type LatticeInitializationError = object of CatchableError

#[ backend: implementation of exception handling types ]#

template newLatticeInitializationError(
  err: LatticeInitializationErrors,
  appendToMessage: untyped
): untyped =
  # constructs error to be raised according to LatticeInitializationErrors spec
  if myRank() == 0:
    var errorMessage {.inject.} = case err:
      of LatticeSubdivisionError:
        "Not enough factors of 2 for subdivision of lattice."
      of IncompatibleDistMemoryGeometryError:
        "Dimension of lattice and distributed memory geometry are incompatible."
      of BadDistMemoryGeometrySpecificationError:
        "Product of entries in distributed memory geometry must equal rank number."
      of IncompatibleShrdMemoryGeometryError:
        "Dimension of lattice and shared memory geometry are incompatible."
      of BadShrdMemoryGeometrySpecificationError:
        "Product of entries in shared memory geometry must equal # threads."
      of IncompatibleVectorLaneGeometryError:
        "Dimension of lattice and vector lane geometry are incompatible."
      of BadVectorLaneGeometrySpecificationError:
        "Product of entries in vector lane geometry must equal # SIMD lanes."
    errorMessage = printBreak & errorMessage
    appendToMessage
    print errorMessage & printBreak
    raise newException(LatticeInitializationError, "")

proc check[L,D,S,V: SomeInteger](
  latticeGeometry: openArray[L],
  distributedMemoryPartition: openArray[D],
  sharedMemoryPartition: openArray[S],
  vectorLanePartition: openArray[V]
) =
  if latticeGeometry.len != distributedMemoryPartition.len:
     newLatticeInitializationError(IncompatibleDistMemoryGeometryError):
      errorMessage &= "\nlattice geometry:" + $latticeGeometry
      errorMessage &= "\nrank geometry:" + $distributedMemoryPartition
  if distributedMemoryPartition.product != numRanks():
     newLatticeInitializationError(BadDistMemoryGeometrySpecificationError):
      errorMessage &= "\nrank geometry:" + $distributedMemoryPartition
      errorMessage &= "\n# ranks:" + $numRanks()
  if latticeGeometry.len != sharedMemoryPartition.len:
     newLatticeInitializationError(IncompatibleShrdMemoryGeometryError):
      errorMessage &= "\nshared memory geometry:" + $sharedMemoryPartition
      errorMessage &= "\nlattice geometry:" + $latticeGeometry
  if sharedMemoryPartition.product != numThreads():
     newLatticeInitializationError(BadShrdMemoryGeometrySpecificationError):
      errorMessage &= "\nshared memory geometry:" + $sharedMemoryPartition
      errorMessage &= "\n# threads:" + $numThreads()
  if latticeGeometry.len != vectorLanePartition.len:
     newLatticeInitializationError(IncompatibleVectorLaneGeometryError):
      errorMessage &= "\nvector lane geometry:" + $vectorLanePartition
      errorMessage &= "\nlattice geometry:" + $latticeGeometry
  if vectorLanePartition.product != numLanes():
     newLatticeInitializationError(BadVectorLaneGeometrySpecificationError):
      errorMessage &= "\nvector lane geometry:" + $vectorLanePartition
      errorMessage &= "\n# SIMD lanes:" + $numLanes()

#[ backend: helper procedures: for constructors ]#

proc partition[L: SomeInteger, B: SomeInteger](
  lg: openArray[L]; 
  bins: B,
  startWithLast: bool = true
): seq[L] =
  # partitions grid of size "nd" into integral number of bins
  type T = type(lg[0])
  let nd = lg.len
  var rlg = lg.toSeq()
  var (r, mu) = (bins, if startWithLast: (bins div nd + bins mod nd)*nd - 1 else: 0)
  result = ones(nd, T)
  while r > 1:
    if not startWithLast: # try to reduce nodal surface area w/ vector lanes
      let 
        lastIsLarger = rlg[mu] < rlg[mu + 1]
        lastIsNotOdd = rlg[mu + 1] mod 2 == 0
      if all(lastIsLarger, lastIsNotOdd): inc mu
    let dim = abs(mu) mod nd
    if divides(2, rlg[dim]):
      result[dim] *= 2
      rlg[dim] = rlg[dim] div 2
      r = r div 2
    elif not dividesSomeElement(2, rlg) and (r > 1): 
       newLatticeInitializationError(LatticeSubdivisionError):
        errorMessage &= "\nlattice geometry:" + $lg
        errorMessage &= "\nleftover lattice geometry:" + $rlg
        errorMessage &= "\nunfilled bins (ranks, SIMD lanes, ...):" + $r
    if startWithLast: dec mu
    else: inc mu

proc newBlockCoord(dg: seq[GeometryType]; lexIdx: int): seq[GeometryType] =
  # gets block coordinate of block-distributed lattice (not to be confused
  # with lattice coordinate)
  var rank = GeometryType(lexIdx)
  result = newSeq[GeometryType](dg.len)
  for mu in dg.reversedDimensions:
    result[mu] = rank mod dg[mu]
    rank = rank div dg[mu]

proc newBlockDims(lg, dg, rc: seq[GeometryType]): seq[SimpleCubicDims] =
  # gets range of each dimension for sublattice that this rank is responsible for
  result = newSeq[SimpleCubicDims](dg.len)
  for mu in dg.dimensions:
    let (b, r) = (lg[mu] div dg[mu], lg[mu] mod dg[mu])
    let start = rc[mu] * b + min(rc[mu], r)
    let size = b + GeometryType(if rc[mu] < r: 1 else: 0)
    result[mu] = (start: start, stop: start + size - 1, size: size)

proc newLocalStrides(rb: seq[SimpleCubicDims]): seq[GeometryType] =
  # gets local strides for this block: needed for index flattening
  result = newSeq[GeometryType](rb.len)
  result[^1] = 1
  for mu in rb.reversedDimensions(start = 1):
    result[mu] = result[mu + 1] * rb[mu + 1].size

#[ frontend: methods and templated "virtual" attribute accessors ]#

# sites accessors
template numGlobalSites*(l: SimpleCubicLattice): untyped =
  l.local()[].numGlobalSites
template numDistributedSites*(l: SimpleCubicLattice): untyped =
  l.local()[].numSites
template numSharedSites*(l: SimpleCubicLattice): untyped =
  l.local()[].shrdMemLattices[0].numSites
template numVectorLaneSites*(l: SimpleCubicLattice): untyped =
  l.local()[].shrdMemLattices[0].vecLaneLattices[0].numSites

# partition accessors
template distributedMemoryPartition*(l: SimpleCubicLattice): untyped =
  l.local()[].partition
template sharedMemoryPartition*(l: SimpleCubicLattice): untyped =
  l.local()[].shrdMemLattices[0].partition
template vectorLanePartition*(l: SimpleCubicLattice): untyped =
  l.local()[].shrdMemLattices[0].vecLaneLattices[0].partition

# local geometry accessors
template globalGeometry*(l: SimpleCubicLattice): untyped =
  l.local()[].globalGeometry
template distributedMemoryGeometry*(l: SimpleCubicLattice): untyped =
  l.local()[].geometry
template sharedMemoryGeometry*(l: SimpleCubicLattice): untyped =
  l.local()[].shrdMemLattices[0].geometry
template vectorLaneGeometry*(l: SimpleCubicLattice): untyped =
  l.local()[].shrdMemLattices[0].vecLaneLattices[0].geometry

proc `$`*(l: SimpleCubicLattice): string =
  ## String representation of SimpleCubicLattice
  const sp1 = "            "
  const sp2 = "     "
  const sp3 = "            "
  result = "SimpleCubicLattice:\n"
  result &= "  lattice geometry:" & sp1 + $(l.globalGeometry) + "\n"
  result &= "  distributed memory partition:" + $(l.distributedMemoryPartition) + "\n"
  result &= "  distributed memory geometry: " + $(l.distributedMemoryGeometry) + "\n"
  result &= "  shared memory partition:" & sp2 + $(l.sharedMemoryPartition) + "\n"
  result &= "  shared memory geometry: " & sp2 + $(l.sharedMemoryGeometry) + "\n"
  result &= "  vector partition:" & sp3 + $(l.vectorLanePartition) + "\n"
  result &= "  vector geometry: " & sp3 + $(l.vectorLaneGeometry)

#[ frontend: SimpleCubicLattice constructors ]#

#proc initSimpleCubicLatticeRoot(geometry, partition: seq[GeometryType])

proc initSimpleCubicLatticeRoot(
  l: var SimpleCubicLatticeRoot,
  lg, p: seq[GeometryType];
  lexIdx: int,
) =
  l.geometry = lg / p
  l.partition = p
  l.coordinate = newBlockCoord(p, lexIdx)
  l.dimensions = lg.newBlockDims(p, l.coordinate)
  l.strides = newLocalStrides(l.dimensions)
  l.numSites = 1
  for mu in l.geometry.dimensions: l.numSites *= l.geometry[mu]

proc newSimpleCubicLatticeRoot(
  lg, p: seq[GeometryType];
  lexIdx: int
): SimpleCubicLatticeRoot = result.initSimpleCubicLatticeRoot(lg, p, lexIdx)

proc newSharedLattice(
  lg, sp, vp: seq[GeometryType];
  thread: int
): SharedSimpleCubicLattice =
  result.initSimpleCubicLatticeRoot(lg, sp, thread)
  var vecLaneLattices = newSeq[SimpleCubicLatticeRoot](numLanes())
  for lane in 0..<numLanes():
    vecLaneLattices[lane] = (lg / sp).newSimpleCubicLatticeRoot(vp, lane)
  result.vecLaneLattices = vecLaneLattices

proc newSimpleCubicLattice*[L,D,S,V: SomeInteger](
  latticeGeometry: openArray[L],
  distributedMemoryPartition: openArray[D],
  sharedMemoryPartition: openArray[S],
  vectorLanePartition: openArray[V],
  quiet: bool = false
): SimpleCubicLattice =
  ## `SimpleCubicLattice` constructor
  ## 
  ## TL;DR: base `SimpleCubicLattice` constructor; called by all constructor variants
  ## 
  ## Default constructor for `SimpleCubicLattice`. Lattice indices are assigned 
  ## hierarhically to ranks, threads, and vector lanes.
  ## 
  ## NOTE: SHOULD INCLUDE DIAGRAM OF HIERARCHICAL PARTITIONING AND AFFINITIES
  let gg = latticeGeometry.toSeq(GeometryType)
  let
    dp = distributedMemoryPartition.toSeq(GeometryType)
    sp = sharedMemoryPartition.toSeq(GeometryType)
    vp = vectorLanePartition.toSeq(GeometryType)
  var sharedMemoryLattices = newSeq[SharedSimpleCubicLattice](numThreads())
  var localLattice: ptr DistributedSimpleCubicLattice
  
  # perform error checking before allocating memory to simple cubic lattice
  check(gg, dp, sp, vp)
  result = DistributedSimpleCubicLattice(globalGeometry: gg).newGlobalPointer()

  # fill in attributes of this chunk of distributed memory lattice
  localLattice = result.local()
  localLattice[].initSimpleCubicLatticeRoot(gg, dp, myRank())
  localLattice[].numGlobalSites = gg.product
  
  # set up shared memory sublattices
  for thread in 0..<numThreads():
    sharedMemoryLattices[thread] = (gg / dp).newSharedLattice(sp, vp, thread)
  localLattice[].shrdMemLattices = sharedMemoryLattices
  
  # print information about full global lattice partitioning
  if not quiet: reliqLog $result

proc newSimpleCubicLattice*(
  latticeGeometry: openArray[SomeInteger],
  quiet: bool = false
): SimpleCubicLattice = 
  ## `SimpleCubicLattice` constructor
  ## 
  ## TL;DR: simplest `SimpleCubicLattice` constructor
  ## 
  ## The following attributes of `SimpleCubicLattice` are inferred.
  ## * Distributed memory geometry inferred from rank number. Splitting of lattice
  ##   dimensions into ranks starts with last dimension (conventional
  ##   Euclidean time direction).
  ## * Shared memory geometry inferred from shared memory rank number
  ## * Vector lane geometry inferred from SIMD lane number
  ## 
  ## Please refer to primary constructor method for further details.
  let 
    g = latticeGeometry.toSeq(GeometryType)
    dp = g.partition(numRanks())
    lg = g / dp
    sp = lg.partition(numThreads(), startWithLast = lg[0] < lg[^1])
    vlg = lg / sp
    vlp = vlg.partition(numLanes(), startWithLast = vlg[0] < vlg[^1])
  return g.newSimpleCubicLattice(dp, sp, vlp, quiet = quiet)

proc newSimpleCubicLattice*(
  latticeGeometry: openArray[SomeInteger],
  distributedMemoryPartition: openArray[SomeInteger],
  quiet: bool = false
): SimpleCubicLattice = 
  ## `SimpleCubicLattice` constructor
  ## 
  ## TL;DR: next-to-simplest `SimpleCubicLattice` constructor
  ## 
  ## The following attributes of `SimpleCubicLattice` are inferred.
  ## * Shared memory geometry inferred from shared memory rank number
  ## * Vector lane geometry inferred from SIMD lane number
  ## 
  ## Please refer to primary constructor method for further details.
  let 
    g = latticeGeometry.toSeq(GeometryType)
    dp = distributedMemoryPartition.toSeq(GeometryType)
    lg = g / dp
    sp = lg.partition(numThreads(), startWithLast = lg[0] < lg[^1])
    vlg = lg / sp
    vlp = vlg.partition(numLanes(), startWithLast = vlg[0] < vlg[^1])
  return g.newSimpleCubicLattice(dp, sp, vlp, quiet = quiet)

proc newSimpleCubicLattice*(
  latticeGeometry: openArray[SomeInteger],
  distributedMemoryPartition: openArray[SomeInteger],
  sharedMemoryPartition: openArray[SomeInteger],
  quiet: bool = false
): SimpleCubicLattice = 
  ## `SimpleCubicLattice` constructor
  ## 
  ## TL;DR: next-to-next-to-simplest `SimpleCubicLattice` constructor
  ## 
  ## The following attributes of `SimpleCubicLattice` are inferred.
  ## * Vector lane geometry inferred from SIMD lane number
  ## 
  ## Please refer to primary constructor method for further details.
  let 
    g = latticeGeometry.toSeq(GeometryType)
    dp = distributedMemoryPartition.toSeq(GeometryType)
    lg = g / dp
    sp = sharedMemoryPartition.toSeq(GeometryType)
    vlg = lg / sp
    vlp = vlg.partition(numLanes(), startWithLast = vlg[0] < vlg[^1])
  return g.newSimpleCubicLattice(dp, sp, vlp, quiet = quiet)

when isMainModule:
  reliq:
    let lattice = [8, 8, 8, 16].newSimpleCubicLattice() 

