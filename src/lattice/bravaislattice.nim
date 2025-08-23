#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/lattice/bravais.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  Notes:
  * What kind of gain (if any) could be achieved by considering inter-node
  connectivity in partitioning of lattice?

  MIT License
  
  Copyright (c) 2025 Curtis Taylor Peterson
  
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

import latticeconcept
import utils
import upcxx

#[ frontend: Bravais lattice types ]#

type
  SimpleCubicLattice* = object
    ## Simple cubic Bravais lattice
    ## Author: Curtis Taylor Peterson
    ## 
    ## <in need of documentation>
    latticeGeometry: seq[GeometryType]
    rankGeometry: seq[GeometryType]
    #globalPtr: upcxx_global_ptr[CoordinateType]
    #sites: seq[CoordinateType]

#[ backend: types for exception handling ]#

# enumerate possible errors that a user may run into
type LatticeInitializationErrors = enum 
  LatticeSubdivisionError,
  IncompatibleRankGeometryError,
  BadRankGeometrySpecificationError

# special error type for handling exception during lattice initialization
type LatticeInitializationError = object of CatchableError

#[ backend: implementation of exception handling types ]#

template newLatticeInitializationError*(
  err: LatticeInitializationErrors,
  appendToMessage: untyped
): untyped =
  # constructs error to be raised according to LatticeInitializationErrors spec
  if upcxx_rank_me() == 0:
    var errorMessage {.inject.} = case err:
      of LatticeSubdivisionError:
        "Not enough factors of 2 for subdivision of lattice."
      of IncompatibleRankGeometryError:
        "Dimension of lattice and rank geometry are incompatible."
      of BadRankGeometrySpecificationError:
        "Product of entries in rank geometry must equal rank number."
    errorMessage = printBreak & errorMessage
    appendToMessage
    print errorMessage & printBreak
  newException(LatticeInitializationError, "")

#[ backend: helper procedures: for constructors ]#

proc partition(lg: openArray[SomeInteger]; bins: SomeInteger): seq[SomeInteger] =
  # partitions grid of size "nd" into integral number of bins
  let nd = lg.len
  var rlg = lg.copyToSeq
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
        errorMessage &= "\nlattice geometry: " & $lg
        errorMessage &= "\nleftover lattice geometry: " & $rlg
        errorMessage &= "\nunfilled bins (ranks, SIMD lanes, ...): " & $r
    dec mu

proc toGeomSeq[T](g: openArray[T]): seq[GeometryType] =
  # converts openArray of generic type to sequence of GeometryType
  result = newSeq[GeometryType](g.len)
  for mu in g.dimensions: result[mu] = GeometryType(g[mu])

proc newRankCoord(rg: seq[GeometryType]): seq[GeometryType] =
  # gets rank coordinate of block-distributed lattice (not to be confused
  # with lattice coordinate)
  var rank = GeometryType(upcxx_rank_me())
  result = newSeq[GeometryType](rg.len)
  for mu in rg.reversedDimensions:
    result[mu] = rank mod rg[mu]
    rank = rank div rg[mu]

proc newRankBlock(lg, rg, rc: seq[GeometryType]): seq[seq[GeometryType]] =
  # gets range of each dimension for sublattice that this rank is responsible for
  result = newSeq[seq[GeometryType]](rg.len)
  for mu in rg.dimensions:
    let (b, r) = (lg[mu] div rg[mu], lg[mu] mod rg[mu])
    let start = rc[mu] * b + min(rc[mu], r)
    let size = b + GeometryType(if rc[mu] < r: 1 else: 0)
    result[mu] = @[start, size]

proc newLocalStrides(rb: seq[seq[GeometryType]]): seq[GeometryType] =
  # gets local strides for this block: needed for index flattening
  result = newSeq[GeometryType](rb.len)
  result[^1] = 1
  for mu in rb.reversedDimensions(start = 1):
    result[mu] = result[mu + 1] * rb[mu + 1][^1]

proc newLocalIndices(rb: seq[seq[GeometryType]]): upcxx_global_ptr[CoordinateType] =
  # constructs a upcxx::global_ptr to the flattened lattices indices owned 
  # by current rank
  var numLocalIndices: csize_t = 1
  for mu in rb.dimensions: numLocalIndices *= csize_t(rb[mu][^1])
  result = upcxx_new_array[CoordinateType](numLocalIndices)

#[ frontend: SimpleCubicLattice constructors ]#

proc newSimpleCubicLattice*(
  latticeGeometry: openArray[SomeInteger],
  rankGeometry: openArray[SomeInteger]
): SimpleCubicLattice =
  ## SimpleCubicLattice constructor
  ## Author: Curtis Taylor Peterson
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
  if latticeGeometry.len != rankGeometry.len:
    raise newLatticeInitializationError(IncompatibleRankGeometryError):
      errorMessage &= "\nlattice geometry: " & $latticeGeometry
      errorMessage &= "\nrank geometry: " & $rankGeometry
  if rankGeometry.product != upcxx_rank_n():
    raise newLatticeInitializationError(BadRankGeometrySpecificationError):
      errorMessage &= "\nrank geometry: " & $rankGeometry
      errorMessage &= "\n# ranks: " & $upcxx_rank_n()

  # set up lattice geometry
  let
    nd = latticeGeometry.len
    lg = toGeomSeq(latticeGeometry)
  
  # set up rank geometry
  let 
    rg = toGeomSeq(rankGeometry)
    rc = newRankCoord(rg)
    rb = newRankBlock(lg, rg, rc)
  let (ls, li) = (newLocalStrides(rb), newLocalIndices(rb))

  # return instantiated SimpleCubicLattice
  return SimpleCubicLattice(
    latticeGeometry: lg, 
    rankGeometry: rg
  )

proc newSimpleCubicLattice*(
  latticeGeometry: openArray[SomeInteger]
): SimpleCubicLattice = 
  ## SimpleCubicLattice constructor
  ## 
  ## TL;DR: Simplest SimpleCubicLattice constructor
  ## 
  ## The following attributes of SimpleCubicLattice are inferred.
  ## * Rank geometry inferred from rank number. Splitting of lattice 
  ##   dimensions into ranks starts with last dimension (conventional 
  ##   Euclidean time direction).
  ## 
  ## Please refer to primary constructor method for further details.
  let rg = latticeGeometry.partition(upcxx_rank_n())
  return newSimpleCubicLattice(latticeGeometry, rg)

#[ tests ]#

when isMainModule: 
  # nim cpp --path:/home/curtyp/Software/ReliQ/src bravaislattice
  # local test: upcxx-run -n 4 -localhost bravaislattice
  upcxx_init()

  let latGeom = [8, 8, 8, 16]
  let latA = newSimpleCubicLattice(latGeom)
  let
    rankGeomB = latGeom.partition(upcxx_rank_n()) # used here for test: not exposed
    latB = newSimpleCubicLattice(latGeom, rankGeomB)

  upcxx_finalize()