#[ 
  QXX lattice field theory framework: github.com/ctpeterson/QXX
  Source file: test/tlattice/tbravais.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

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

import utils
import latticeconcept
import upcxx

#[ frontend: Bravais lattice types ]#

type
  SimpleCubicLattice* = object
    ## Simple cubic Bravais lattice
    ## Author: Curtis Taylor Peterson
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

template newLatticeInitializationError*(err: LatticeInitializationErrors): untyped =
  # constructs error to be raised according to LatticeInitializationErrors spec
  var msg = case err:
    of LatticeSubdivisionError:
      "Not enough factors of 2 for subdivision of lattice."
    of IncompatibleRankGeometryError:
      "Dimension of lattice and rank geometry are incompatible."
    of BadRankGeometrySpecificationError:
      "Product of entries in rank geometry must accumulate to total number of ranks."
  newException(LatticeInitializationError, msg)

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
      raise newLatticeInitializationError(LatticeSubdivisionError)
    dec mu

proc toGeomSeq[T](ilg, irg: openArray[T]): (seq[GeometryType], seq[GeometryType]) =
  # converts open arrays specifying lattice and rank geometry into sequences
  var (olg, org) = (newSeq[GeometryType](ilg.len), newSeq[GeometryType](irg.len))
  for mu in olg.dimensions: 
    (olg[mu], org[mu]) = (GeometryType(ilg[mu]), GeometryType(irg[mu]))
  return (olg, org)

proc newRankCoord(rg: seq[GeometryType]): seq[GeometryType] =
  # gets rank coordinate of block-distributed lattice (not to be confused
  # with lattice coordinate)
  var rank = GeometryType(upcxx_rank_me())
  result = newSeq[GeometryType](rg.len)
  for mu in countdown(rg.len - 1, 0):
    result[mu] = rank mod rg[mu]
    rank = rank div rg[mu]

proc newRankBlock(
  lg, rg, rc: seq[GeometryType]
): seq[(GeometryType, GeometryType)] =
  # gets range of each dimension for sublattice that this rank is
  # responsible for
  result = newSeq[(GeometryType, GeometryType)](rg.len)
  for mu in rg.dimensions:
    let (b, r) = (lg[mu] div rg[mu], lg[mu] mod rg[mu])
    let start = rc[mu] * b + min(rc[mu], r)
    let size = b + GeometryType(if rc[mu] < r: 1 else: 0)
    result[mu] = (start, size)

#[ frontend: SimpleCubicLattice constructors ]#

proc newSimpleCubicLattice*(
  latticeGeometry, rankGeometry: openArray[SomeInteger]
): SimpleCubicLattice =
  ## SimpleCubicLattice constructor
  ## Author: Curtis Taylor Peterson
  ## <in need of documentation>
  if latticeGeometry.len != rankGeometry.len:
    raise newLatticeInitializationError(IncompatibleRankGeometryError)
  if rankGeometry.product != upcxx_rank_n():
    raise newLatticeInitializationError(BadRankGeometrySpecificationError)

  let nd = latticeGeometry.len
  let (lg, rg) = toGeomSeq(latticeGeometry, rankGeometry)
  let 
    rc = newRankCoord(rg)
    rb = newRankBlock(lg, rg, rc)

  return SimpleCubicLattice(latticeGeometry: lg, rankGeometry: rg)

proc newSimpleCubicLattice*(
  latticeGeometry: openArray[SomeInteger]
): SimpleCubicLattice = 
  ## SimpleCubicLattice constructor
  ## Author: Curtis Taylor Peterson
  ## <in need of documentation>
  let rg = latticeGeometry.partition(upcxx_rank_n())
  return newSimpleCubicLattice(latticeGeometry, rg)

when isMainModule: 
  # nim cpp --path:/home/curtyp/Software/QXX/src bravais
  # upcxx-run -n 4 -localhost bravais
  upcxx_init()

  let lat = newSimpleCubicLattice([8, 8, 8, 16])

  upcxx_finalize()