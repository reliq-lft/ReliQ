#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/lattice/simplecubiclattice.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  Notes:
  * What kind of gain (if any) could be achieved by considering inter-node
  connectivity in partitioning of lattice?

  MIT License
  
  Copyright (c) 2025 Curtis Taylor Peterson
  
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
import utils
import latticeconcept

#[ frontend: simple cubic lattice type definition ]#

type
  SimpleCubicRankDim = tuple[start, stop, size: GeometryType]
  SimpleCubicLattice* = object
    ## Simple cubic Bravais lattice
    ## 
    ## <in need of documentation>
    globalGeometry: seq[GeometryType]
    localPartition: seq[GeometryType]
    localBlockCoordinate: seq[GeometryType]
    localGeometry: seq[SimpleCubicRankDim]
    localStrides: seq[GeometryType]
    numLocalSites*: GeometryType
    
#[ backend: types for exception handling ]#

# enumerate possible errors that a user may run into
type LatticeInitializationErrors = enum 
  LatticeSubdivisionError,
  IncompatibledistMemoryGeometryError,
  BadDistMemoryGeometrySpecificationError

# special error type for handling exception during lattice initialization
type LatticeInitializationError = object of CatchableError

#[ backend: implementation of exception handling types ]#

template newLatticeInitializationError*(
  err: LatticeInitializationErrors,
  appendToMessage: untyped
): untyped =
  # constructs error to be raised according to LatticeInitializationErrors spec
  if myRank() == 0:
    var errorMessage {.inject.} = case err:
      of LatticeSubdivisionError:
        "Not enough factors of 2 for subdivision of lattice."
      of IncompatibledistMemoryGeometryError:
        "Dimension of lattice and rank geometry are incompatible."
      of BadDistMemoryGeometrySpecificationError:
        "Product of entries in rank geometry must equal rank number."
    errorMessage = printBreak & errorMessage
    appendToMessage
    print errorMessage & printBreak
  newException(LatticeInitializationError, "")

#[ backend: helper procedures: for constructors ]#

proc partition(lg: openArray[SomeInteger]; bins: SomeInteger): seq[SomeInteger] =
  # partitions grid of size "nd" into integral number of bins
  let nd = lg.len
  var rlg = lg.toSeq()
  var (r, mu) = (bins, (bins div nd + bins mod nd)*nd - 1)
  result = ones(nd, type(bins))
  while r > 1:
    let dim = abs(mu) mod nd
    if divides(2, rlg[dim]):
      result[dim] *= 2
      rlg[dim] = rlg[dim] div 2
      r = r div 2
    elif not dividesSomeElement(2, rlg) and (r > 1): 
      raise newLatticeInitializationError(LatticeSubdivisionError):
        errorMessage &= "\nlattice geometry:" + $lg
        errorMessage &= "\nleftover lattice geometry:" + $rlg
        errorMessage &= "\nunfilled bins (ranks, SIMD lanes, ...):" + $r
    dec mu

proc newRankCoord(dg: seq[GeometryType]): seq[GeometryType] =
  # gets rank coordinate of block-distributed lattice (not to be confused
  # with lattice coordinate)
  var rank = GeometryType(myRank())
  result = newSeq[GeometryType](dg.len)
  for mu in dg.reversedDimensions:
    result[mu] = rank mod dg[mu]
    rank = rank div dg[mu]

proc newRankBlock(lg, dg, rc: seq[GeometryType]): seq[SimpleCubicRankDim] =
  # gets range of each dimension for sublattice that this rank is responsible for
  result = newSeq[SimpleCubicRankDim](dg.len)
  for mu in dg.dimensions:
    let (b, r) = (lg[mu] div dg[mu], lg[mu] mod dg[mu])
    let start = rc[mu] * b + min(rc[mu], r)
    let size = b + GeometryType(if rc[mu] < r: 1 else: 0)
    result[mu] = (start: start, stop: start + size - 1, size: size)

proc newLocalStrides(rb: seq[SimpleCubicRankDim]): seq[GeometryType] =
  # gets local strides for this block: needed for index flattening
  result = newSeq[GeometryType](rb.len)
  result[^1] = 1
  for mu in rb.reversedDimensions(start = 1):
    result[mu] = result[mu + 1] * rb[mu + 1].size

#[ frontend: SimpleCubicLattice constructors ]#

proc newSimpleCubicLattice*(other: SimpleCubicLattice): SimpleCubicLattice = 
  ## SimpleCubicLattice copy constructor
  ## 
  ## Copy constructor for SimpleCubicLattice.
  ## <in need of more documentation>
  result = SimpleCubicLattice(
    globalGeometry: other.globalGeometry,
    localPartition: other.localPartition,
    localBlockCoordinate: other.localBlockCoordinate,
    localGeometry: other.localGeometry,
    localStrides: other.localStrides,
    numLocalSites: other.numLocalSites
  )

proc newSimpleCubicLattice*(
  latticeGeometry: openArray[SomeInteger],
  distMemoryGeometry: openArray[SomeInteger],
  sharedMemoryGeometry: openArray[SomeInteger]
): SimpleCubicLattice =
  ## SimpleCubicLattice constructor
  ## 
  ## TL;DR: Base SimpleCubicLattice constructor
  ## 
  ## Primary constructor for SimpleCubicLattice. All constructor variants
  ## attempt to infer any information that is not provided explicitly to
  ## this constructor. Plase see variants of SimpleCubicLattice constructor
  ## for details of what is inferred and how it is inferred.
  ## 
  ## <in need of more documentation>
  
  #[ distributed memory specifications ]#

  # catch most common errors
  if latticeGeometry.len != distMemoryGeometry.len:
    raise newLatticeInitializationError(IncompatibledistMemoryGeometryError):
      errorMessage &= "\nlattice geometry:" + $latticeGeometry
      errorMessage &= "\nrank geometry:" + $distMemoryGeometry
  if distMemoryGeometry.product != numRanks():
    raise newLatticeInitializationError(BadDistMemoryGeometrySpecificationError):
      errorMessage &= "\nrank geometry:" + $distMemoryGeometry
      errorMessage &= "\n# ranks:" + $numRanks()

  # set up lattice geometry
  let lg = latticeGeometry.toSeq(GeometryType)

  # set up distributed memory geometry
  let
    dg = distMemoryGeometry.toSeq(GeometryType)
    rc = newRankCoord(dg)
    rb = newRankBlock(lg, dg, rc)
    ls = newLocalStrides(rb)
  var numLocalSites: GeometryType = 1
  for mu in rb.dimensions: numLocalSites *= rb[mu].size

  # return instantiated SimpleCubicLattice
  return SimpleCubicLattice(
    globalGeometry: lg,
    localPartition: dg,
    localBlockCoordinate: rc,
    localGeometry: rb,
    localStrides: ls,
    numLocalSites: numLocalSites
  )

proc newSimpleCubicLattice*(
  latticeGeometry: openArray[SomeInteger],
  distMemoryGeometry: openArray[SomeInteger]
): SimpleCubicLattice = 
  ## SimpleCubicLattice constructor
  ## 
  ## TL;DR: Next-to-simplest SimpleCubicLattice constructor
  ## 
  ## The following attributes of SimpleCubicLattice are inferred.
  ## * Shared memory geometry inferred from shared memory rank number
  ## 
  ## Please refer to primary constructor method for further details.
  let sg = latticeGeometry.partition(numThreads())
  return newSimpleCubicLattice(latticeGeometry, distMemoryGeometry, sg)

proc newSimpleCubicLattice*(
  latticeGeometry: openArray[SomeInteger]
): SimpleCubicLattice = 
  ## SimpleCubicLattice constructor
  ## 
  ## TL;DR: Simplest SimpleCubicLattice constructor
  ## 
  ## The following attributes of SimpleCubicLattice are inferred.
  ## * Distributed memory geometry inferred from rank number. Splitting of lattice
  ##   dimensions into ranks starts with last dimension (conventional
  ##   Euclidean time direction).
  ## * Shared memory geometry inferred from shared memory rank number
  ## 
  ## Please refer to primary constructor method for further details.
  let 
    dg = latticeGeometry.partition(numRanks())
    sg = latticeGeometry.partition(numLanes())
  return newSimpleCubicLattice(latticeGeometry, dg, sg)

#[ frontend: Lattice concept conformance ]#

# implement sites method for Lattice concept
iterator sites*(l: SimpleCubicLattice): int =
  for n in 0..<l.numLocalSites: yield n

# implement latticeCoordinate method for Lattice concept
proc latticeCoordinate*(l: SimpleCubicLattice; n: int): seq[int] =
  ## Gets lattice coordinate from flattened index
  ## 
  ## <in need of documentation>
  result = newSeq[int](l.localGeometry.len)
  var nIdx = n
  for mu in l.localGeometry.dimensions:
    result[mu] = l.localGeometry[mu].start + nIdx div l.localStrides[mu]
    nIdx = nIdx mod l.localStrides[mu]

when isMainModule:
  import runtime
  const verbosity = 1
  reliq:
    let latGeom = [8, 8, 8, 16]
    let latA = newSimpleCubicLattice(latGeom)
    let
      rankGeomB = latGeom.partition(numRanks()) # used here for test: not exposed
      latB = newSimpleCubicLattice(latGeom, rankGeomB)
    
    for site in latA.sites:
      let coord = latA.latticeCoordinate(site)
      for mu in coord.dimensions:
        assert(coord[mu] >= latA.localGeometry[mu].start)
        assert(coord[mu] <= latA.localGeometry[mu].stop)
      if verbosity > 1: 
        echo "[" & $myRank() & "] " & "latA site: ", site, " coord: ", coord