#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/lattice/simplecubiclattice.nim
  Contact: reliq-lft@proton.me

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

# import backend header files
backend: discard

#[ frontend: simple cubic lattice type definition ]#

type
  # halo information for simple cubic lattice
  SimpleCubicLatticeHalo = object
    depth: int
    neighbors: seq[tuple[left, right: int]]

type
  # root object for simple cubic lattice types
  SimpleCubicLatticeRoot {.inheritable.} = object of RootObj
    geometry, partition, strides: seq[int]
    coordinate: seq[int]
    dimensions: seq[tuple[start, stop, size: int]]
    numSites: int

  # shared memory simple cubic Bravais (sub)lattice
  SharedSimpleCubicLattice = object of SimpleCubicLatticeRoot
    vecLaneLattices: seq[SimpleCubicLatticeRoot]

  SimpleCubicLattice* = object of SimpleCubicLatticeRoot
    ## Simple cubic (Bravais) lattice
    ## ------------------------------
    ## 
    ## TL;DR: Specifies geometry simple cubic Bravais lattices
    ## 
    ## Specifies lattice geometry and parallel hierarchy of simple cubic Bravais
    ## lattices. This is `ReliQ`'s analogue of `Grid`'s `GridBase` type and 
    ## `Quantum EXpressions` (`QEX`'s) `layout` type. Lattice indices are assigned 
    ## hierarchically to distributed memory ranks, shared memory threads, and
    ## local shared memory vector lanes. The distributed memory ranks are ReliQ's
    ## realization of Message Passing Interface (MPI) ranks; note that such ranks
    ## run asynchronously when using the `UPC++` backend. How the shared memory 
    ## threads and vector lanes map to hardware depends on the hardware being 
    ## targeted. Each level of parallelism/locality has its own geometry and 
    ## partitioning. As we descend down the hierarchy (ranks --> lanes), the target 
    ## lattice geometry becomes increasingly local. 
    ## 
    ## Attributes:
    ## -----------
    ## * geometry: Geometry of local distributed memory lattice
    ## * strides: Strides of local distributed memory lattice
    ## * partition: Distributed memory partitioning of global lattice geometry
    ## * coordinate: Distributed memory rank coordinate in distributed memory geometry
    ## * dimensions: Range of each dimension for local distributed memory lattice
    ## * numSites: Number of sites in local distributed memory lattice
    ## * globalGeometry: Global lattice geometry
    ## * shrdMemLattices: Sequence of shared memory sublattices
    ## * halo: Halo information for distributed memory lattice
    ## 
    ## Below are a few examples of how to construct a SimpleCubicLattice. In all 
    ## cases, we consider a 4D lattice of geometry [16, 16, 16, 32]; however, the 
    ## lattice does not need to be 4-dimensional. In most cases, one will only work 
    ## with a single lattice at a time; however, it is possible to have multiple 
    ## lattices co-existing in the same progarm (e.g., for multigrid applications).
    ## 
    ## Example:
    ## --------
    ## .. code-block:: nim
    ##   # all hierarchical paritions of lattice are inferred
    ##   let latticeA = [16, 16, 16, 32].newSimpleCubicLattice()
    ## 
    ##   # shared memory and vector lane partitions are inferred
    ##   # product of distributed memory partition must equal rank number; in 
    ##   # this case, numRanks() == 4
    ##   let latticeB = [16, 16, 16, 32].newSimpleCubicLattice([1, 1, 2, 2])
    ##     
    ##   let # vector lane partition is inferred
    ##    latticeC = lat.newSimpleCubicLattice(
    ##       [1, 1, 2, 2], # distributed memory partition (here, numRanks() == 4)
    ##       [1, 2, 2, 2], # shared memory partition (here, numThreads() == 8)
    ##     )
    ## 
    ##   let # all hierarchical partitions of lattice are specified
    ##     latticeD = lat.newSimpleCubicLattice(
    ##       [1, 1, 2, 2], # distributed memory partition (here, numRanks() == 4)
    ##       [1, 2, 2, 2], # shared memory partition (here, numThreads() == 8)
    ##       [2, 2, 2, 1], # vector lane partition (here, numLanes() == 8)
    ##     )
    ## 
    ##   let # ... each constructor also has two optional arguments... -------------+
    ##     latticeE = [16, 16, 16, 32].newSimpleCubicLattice(                       |
    ##       [1, 1, 2, 2],  # distributed memory partition (here, numRanks() == 4)  |
    ##       [1, 2, 2, 2],  # shared memory partition (here, numThreads() == 8)     |
    ##       [2, 2, 2, 1],  # vector lane partition (here, numLanes() == 8)         |
    ##       haloDepth = 1, # optional: default is already 1 <----------------------+
    ##       quiet = false  # optional: default is already false <------------------+
    ##     )
    ## 
    ## Notes:
    ## ------
    ## * Any distributed/shared/vector partition that is not specified will be inferred. 
    ##   Inferred partitions are guided by minimizing the surface area of the local 
    ##   lattice geometry at each level of the hierarchy. In some cases, this may lead 
    ##   to suboptimal partitioning due the topology of the hardware being use; 
    ##   however, it is recommended to first try the inferred partitions before
    ##   specifying custom partitions, if needed. 
    ## * Strides for each local lattice start with time direction (last dimension) 
    ##   to ensure that time-local operations benefit most from spatial locality in 
    ##   memory. We may support custom striding if it is requested; e.g., for 
    ##   applications that would benefit from space-local operations or that continue 
    ##   calculations from another framework with a different time direction. 
    ## * Halo depth is isotropic and defaults to 1; anisotropic halo depths are not
    ##   supported at this time. Halo depths are also fixed at this time; it may be
    ##   nice to support dynamic halo depths in the future.
    ## 
    ## Comments, questions, or suggestions? Please either open an issue on GitHub
    ## or contact us at reliq-lft@proton.me
    globalGeometry: seq[int]
    shrdMemLattices: seq[SharedSimpleCubicLattice]
    halo: SimpleCubicLatticeHalo

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

#[ frontend/backend: some generic helpsers ]#

proc fromLex*(lexIdx: int; geom: seq[int]): seq[int] =
  # converts lexicographic index to coordinate
  var rank = int(lexIdx)
  result = newSeq[int](geom.len)
  for mu in geom.reversedDimensions:
    result[mu] = rank mod geom[mu]
    rank = rank div geom[mu]

proc toLex*(coord, geom: seq[int]): int =
  ## Converts n-dimensional coordinate to lexicographic index using geometry
  var idx = 0
  var stride = 1
  for mu in countdown(coord.len - 1, 0):
    idx += coord[mu] * stride
    stride *= geom[mu]
  return idx

proc toStrides*(geom: seq[int]): seq[int] =
  ## Returns the strides for a given geometry (row-major order)
  result = newSeq[int](geom.len)
  if geom.len == 0: return
  result[^1] = 1
  for i in countdown(geom.len - 2, 0):
    result[i] = result[i + 1] * geom[i + 1]

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
        lastIsLarger = rlg[abs(mu) mod nd] < rlg[(abs(mu) + 1) mod nd]
        lastIsNotOdd = rlg[(abs(mu) + 1) mod nd] mod 2 == 0
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

proc newBlockDims[I:SomeInteger](
  lg, dg, rc: seq[I]
): seq[tuple[start, stop, size: I]] =
  # gets range of each dimension for sublattice that this rank is responsible for
  result = newSeq[tuple[start, stop, size: I]](dg.len)
  for mu in dg.dimensions:
    let (b, r) = (lg[mu] div dg[mu], lg[mu] mod dg[mu])
    let start = rc[mu] * b + min(rc[mu], r)
    let size = b + int(if rc[mu] < r: 1 else: 0)
    result[mu] = (start: start, stop: start + size - 1, size: size)

proc newLocalStrides[I:SomeInteger](rb: seq[tuple[start, stop, size: I]]): seq[I] =
  # gets local strides for this block: needed for index flattening
  result = newSeq[int](rb.len)
  result[^1] = 1
  for mu in rb.reversedDimensions(start = 1):
    result[mu] = result[mu + 1] * rb[mu + 1].size

#[ frontend: methods and templated "virtual" attribute accessors ]#

# sites accessors
template numGlobalSites*(l: SimpleCubicLattice): untyped =
  l.globalGeometry.product * numRanks()
template numDistributedSites*(l: SimpleCubicLattice): untyped =
  l.numSites
template numSharedSites*(l: SimpleCubicLattice): untyped =
  l.shrdMemLattices[0].numSites
template numVectorLaneSites*(l: SimpleCubicLattice): untyped =
  l.shrdMemLattices[0].vecLaneLattices[0].numSites

# partition accessors
template distributedMemoryPartition*(l: SimpleCubicLattice): untyped =
  l.partition
template sharedMemoryPartition*(l: SimpleCubicLattice): untyped =
  l.shrdMemLattices[0].partition
template vectorLanePartition*(l: SimpleCubicLattice): untyped =
  l.shrdMemLattices[0].vecLaneLattices[0].partition

# local geometry accessors
template globalGeometry*(l: SimpleCubicLattice): untyped =
  l.globalGeometry
template distributedMemoryGeometry*(l: SimpleCubicLattice): untyped =
  l.geometry
template sharedMemoryGeometry*(l: SimpleCubicLattice): untyped =
  l.shrdMemLattices[0].geometry
template vectorLaneGeometry*(l: SimpleCubicLattice): untyped =
  l.shrdMemLattices[0].vecLaneLattices[0].geometry

# rank coordinate accessor
template rankCoordinate*(l: SimpleCubicLattice): untyped =
  ## Rank block coordinate in distributed memory geometry
  l.coordinate

# halo depth accessor
template haloDepth*(l: SimpleCubicLattice): untyped =
  ## Isotropic halo depth
  ## 
  ## We do not support anisotropic halo depth at this time. If this
  ## is a feature that you'd like, please contact us: 
  l.halo.depth

# neighbor accessors
template neighbors*(l: SimpleCubicLattice): untyped =
  ## Returns the ranks of neighboring distributed memory blocks
  ## 
  ## The result is a sequence of tuples, where each tuple contains
  ## the lexicographic indices of the left and right neighbors in each
  ## dimension. Example: `leftNeighbor = l.neighbors[mu].left`
  l.halo.neighbors
template leftNeighbor*(l: SimpleCubicLattice; mu: int): untyped =
  ## Returns the rank of the left neighbor in direction `mu`
  l.halo.neighbors[mu].left
template rightNeighbor*(l: SimpleCubicLattice; mu: int): untyped =
  ## Returns the rank of the right neighbor in direction `mu`
  l.halo.neighbors[mu].right
proc leftNeighbors*(l: SimpleCubicLattice): seq[int] =
  ## Returns the rank of left neighbors
  result = newSeq[int](l.halo.neighbors.len)
  for mu in l.halo.neighbors.dimensions: result[mu] = l.leftNeighbor(mu)
proc rightNeighbors*(l: SimpleCubicLattice): seq[int] =
  ## Returns the rank of right neighbors
  result = newSeq[int](l.halo.neighbors.len)
  for mu in l.halo.neighbors.dimensions: result[mu] = l.rightNeighbor(mu)

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

proc initSimpleCubicLatticeRoot(
  l: var SimpleCubicLatticeRoot,
  lg, p: seq[int];
  lexIdx: int,
) =
  l.geometry = lg / p
  l.partition = p
  l.coordinate = lexIdx.fromLex(l.geometry)
  l.dimensions = lg.newBlockDims(p, l.coordinate)
  l.strides = newLocalStrides(l.dimensions)
  l.numSites = 1
  for mu in l.geometry.dimensions: l.numSites *= l.geometry[mu]

proc newSimpleCubicLatticeRoot(
  lg, p: seq[int];
  lexIdx: int
): SimpleCubicLatticeRoot = result.initSimpleCubicLatticeRoot(lg, p, lexIdx)

proc newSharedLattice(
  lg, sp, vp: seq[int];
  thread: int
): SharedSimpleCubicLattice =
  result.initSimpleCubicLatticeRoot(lg, sp, thread)
  var vecLaneLattices = newSeq[SimpleCubicLatticeRoot](numLanes())
  for lane in 0..<numLanes():
    vecLaneLattices[lane] = (lg / sp).newSimpleCubicLatticeRoot(vp, lane)
  result.vecLaneLattices = vecLaneLattices

proc newSimpleCubicLattice*[L,D,S,V,H: SomeInteger](
  latticeGeometry: openArray[L],
  distributedMemoryPartition: openArray[D],
  sharedMemoryPartition: openArray[S],
  vectorLanePartition: openArray[V],
  haloDepth: H = 1,
  quiet: bool = false
): SimpleCubicLattice =
  ## `SimpleCubicLattice` constructor
  ## --------------------------------
  ## 
  ## Base `SimpleCubicLattice` constructor; called by all constructor variants
  ## 
  ## Arguments:
  ## ----------
  ## * latticeGeometry: Open array specifying the global lattice geometry
  ## * distributedMemoryPartition: Open array specifying the distributed memory
  ##     partitioning of the global lattice geometry. The product of the entries
  ##     must equal the number of distributed memory ranks (i.e. `numRanks()`).
  ## * sharedMemoryPartition: Open array specifying the shared memory partitioning
  ##     of the local distributed memory lattice geometry. The product of the entries
  ##     must equal the number of shared memory threads (i.e. `numThreads()`).
  ## * vectorLanePartition: Open array specifying the vector lane partitioning
  ##     of the local shared memory lattice geometry. The product of the entries
  ##     must equal the number of SIMD lanes (i.e. `numLanes()`).
  ## * haloDepth: (optional) Isotropic halo depth; default is 1.
  ## * quiet: (optional) Suppresses output of lattice information if `true`;
  ##     default is `false`.
  ## 
  ## Returns: 
  ## --------
  ##   `SimpleCubicLattice` object
  ## 
  ## Example:
  ## --------
  ## .. code-block:: nim
  ##   let lattice = [16, 16, 16, 32].newSimpleCubicLattice(
  ##     [1, 1, 2, 2],  # product must equal numRanks() = 4 in this case
  ##     [1, 2, 2, 2],  # product must equal numThreads() = 8 in this case
  ##     [2, 2, 2, 1],  # product must equal numLanes() = 8 in this case
  ##     haloDepth = 1, # optional: default is already 1
  ##     quiet = false  # optional: default is already false
  ##   )
  let gg = latticeGeometry.toSeq(int)
  let
    dp = distributedMemoryPartition.toSeq(int)
    sp = sharedMemoryPartition.toSeq(int)
    vp = vectorLanePartition.toSeq(int)
  var sharedMemoryLattices = newSeq[SharedSimpleCubicLattice](numThreads())
  
  # perform error checking before allocating memory to simple cubic lattice
  check(gg, dp, sp, vp)
  result = SimpleCubicLattice(globalGeometry: gg)

  # fill in attributes of this chunk of distributed memory lattice
  result.initSimpleCubicLatticeRoot(gg, dp, myRank())

  # set up shared memory sublattices
  result.shrdMemLattices = newSeq[SharedSimpleCubicLattice](numThreads())
  for thread in result.shrdMemLattices.dimensions:
    result.shrdMemLattices[thread] = (gg / dp).newSharedLattice(sp, vp, thread)

  # specify halo information
  result.halo = SimpleCubicLatticeHalo(
    depth: haloDepth,
    neighbors: newSeq[tuple[left, right: H]](dp.len)
  )
  for mu in dp.dimensions:
    var lCrd, rCrd: seq[H] = result.coordinate
    lCrd[mu] = (if lCrd[mu] - 1 >= 0: lCrd[mu] - 1 else: dp[mu] - 1) mod dp[mu]
    rCrd[mu] = (rCrd[mu] + 1) mod dp[mu]
    let (lLex, rLex) = (lCrd.toLex(dp), rCrd.toLex(dp))
    result.halo.neighbors[mu] = (left: lLex, right: rLex)

  # print information about full global lattice partitioning
  if not quiet: reliqLog $result

proc newSimpleCubicLattice*[L,H: SomeInteger](
  latticeGeometry: openArray[L],
  haloDepth: H = 1,
  quiet: bool = false
): SimpleCubicLattice = 
  ## `SimpleCubicLattice` constructor
  ## --------------------------------
  ## 
  ## TL;DR: Simplest `SimpleCubicLattice` constructor
  ## 
  ## The following attributes of `SimpleCubicLattice` are inferred.
  ## * Distributed memory geometry inferred from rank number. Splitting of lattice
  ##   dimensions into ranks starts with last dimension (conventional
  ##   Euclidean time direction).
  ## * Shared memory geometry inferred from shared memory rank number
  ## * Vector lane geometry inferred from SIMD lane number
  ## 
  ## Arguments:
  ## ----------
  ## * latticeGeometry: Open array specifying the global lattice geometry
  ## * haloDepth: (optional) Isotropic halo depth; default is 1.
  ## * quiet: (optional) Suppresses output of lattice information if `true`;
  ##     default is `false`.
  ## 
  ## Returns: 
  ## --------
  ##   `SimpleCubicLattice` object
  ## 
  ## Example:
  ## --------
  ## .. code-block:: nim
  ##   let lattice = [16, 16, 16, 32].newSimpleCubicLattice(
  ##     haloDepth = 1, # optional: default is already 1
  ##     quiet = false  # optional: default is already false
  ##   )
  let 
    g = latticeGeometry.toSeq(L)
    dp = g.partition(numRanks())
    lg = g / dp
    sp = lg.partition(numThreads(), startWithLast = lg[0] < lg[^1])
    vlg = lg / sp
    vlp = vlg.partition(numLanes(), startWithLast = vlg[0] < vlg[^1])
  return g.newSimpleCubicLattice(dp, sp, vlp, haloDepth = haloDepth, quiet = quiet)

proc newSimpleCubicLattice*[L,D,H: SomeInteger](
  latticeGeometry: openArray[L],
  distributedMemoryPartition: openArray[D],
  haloDepth: H = 1,
  quiet: bool = false
): SimpleCubicLattice = 
  ## `SimpleCubicLattice` constructor
  ## --------------------------------
  ## 
  ## TL;DR: Next-to-simplest `SimpleCubicLattice` constructor
  ## 
  ## The following attributes of `SimpleCubicLattice` are inferred.
  ## * Shared memory geometry inferred from shared memory rank number
  ## * Vector lane geometry inferred from SIMD lane number
  ## 
  ## Arguments:
  ## ----------
  ## * latticeGeometry: Open array specifying the global lattice geometry
  ## * distributedMemoryPartition: Open array specifying the distributed memory
  ##     partitioning of the global lattice geometry. The product of the entries
  ##     must equal the number of distributed memory ranks (i.e. `numRanks()`).
  ## * haloDepth: (optional) Isotropic halo depth; default is 1.
  ## * quiet: (optional) Suppresses output of lattice information if `true`;
  ##     default is `false`.
  ## 
  ## Returns: 
  ## --------
  ##   `SimpleCubicLattice` object
  ## 
  ## Example:
  ## --------
  ## .. code-block:: nim
  ##   let lattice = [16, 16, 16, 32].newSimpleCubicLattice(
  ##     [1, 1, 2, 2],  # product must equal numRanks() = 4 in this case
  ##     haloDepth = 1, # optional: default is already 1
  ##     quiet = false  # optional: default is already false
  ##   )
  let 
    g = latticeGeometry.toSeq(L)
    dp = distributedMemoryPartition.toSeq(D)
    lg = g / dp
    sp = lg.partition(numThreads(), startWithLast = lg[0] < lg[^1])
    vlg = lg / sp
    vlp = vlg.partition(numLanes(), startWithLast = vlg[0] < vlg[^1])
  return g.newSimpleCubicLattice(dp, sp, vlp, haloDepth = haloDepth, quiet = quiet)

proc newSimpleCubicLattice*[L,D,S,H: SomeInteger](
  latticeGeometry: openArray[L],
  distributedMemoryPartition: openArray[D],
  sharedMemoryPartition: openArray[S],
  haloDepth: H = 1,
  quiet: bool = false
): SimpleCubicLattice = 
  ## `SimpleCubicLattice` constructor
  ## --------------------------------
  ## 
  ## TL;DR: Next-to-next-to-simplest `SimpleCubicLattice` constructor
  ## 
  ## The following attributes of `SimpleCubicLattice` are inferred.
  ## * Vector lane geometry inferred from SIMD lane number
  ## 
  ## Arguments:
  ## ----------`
  ## * latticeGeometry: Open array specifying the global lattice geometry
  ## * distributedMemoryPartition: Open array specifying the distributed memory
  ##     partitioning of the global lattice geometry. The product of the entries
  ##     must equal the number of distributed memory ranks (i.e. `numRanks()`).
  ## * sharedMemoryPartition: Open array specifying the shared memory partitioning
  ##     of the local distributed memory lattice geometry. The product of the entries
  ##     must equal the number of shared memory threads (i.e. `numThreads()`).
  ## * haloDepth: (optional) Isotropic halo depth; default is 1.
  ## * quiet: (optional) Suppresses output of lattice information if `true`;
  ##     default is `false`.
  ## 
  ## Returns: 
  ## --------
  ##   `SimpleCubicLattice` object
  ## 
  ## Example:
  ## --------
  ## .. code-block:: nim
  ##   let lattice = [16, 16, 16, 32].newSimpleCubicLattice(
  ##     [1, 1, 2, 2],  # product must equal numRanks() = 4 in this case
  ##     [1, 2, 2, 2],  # product must equal numThreads() = 8 in this case
  ##     haloDepth = 1, # optional: default is already 1
  ##     quiet = false  # optional: default is already false
  ##   )
  let 
    g = latticeGeometry.toSeq(L)
    dp = distributedMemoryPartition.toSeq(D)
    lg = g / dp
    sp = sharedMemoryPartition.toSeq(S)
    vlg = lg / sp
    vlp = vlg.partition(numLanes(), startWithLast = vlg[0] < vlg[^1])
  return g.newSimpleCubicLattice(dp, sp, vlp, haloDepth = haloDepth, quiet = quiet)

#[ ??? frontend: simple cubic lattice dispatch helpers ??? ]#

#[ tests ]#

## TO-DO: clean up generics; likely inconsistent with so many possible
## choices for integer types in SimpleCubicLattice object
when isMainModule:
  reliq:
    # basic 4D lattice
    let lattice = [8, 8, 8, 16].newSimpleCubicLattice()

    # some 2D lattice checks
    if numRanks() == 8:
      let lattice2D = [16, 16].newSimpleCubicLattice()
      for lg in ["[3][7]", "[2][6]", "[1][5]", "[0][4]"]: print lg
      for rank in 0..<numRanks():
        if rank == myRank():
          var check = "rank: " & $rank 
          check &= " coordinate: " & $lattice2D.coordinate
          check &= " neighbors: " & $lattice2D.halo.neighbors
          echo check
    if numRanks() == 16:
      let lattice2D = [16, 16].newSimpleCubicLattice()
      for lg in [
        "[03][07][11][15]", 
        "[02][06][10][14]", 
        "[01][05][09][13]", 
        "[00][04][08][12]"
      ]: print lg
      for rank in 0..<numRanks():
        if rank == myRank():
          var check = "rank: " & $rank 
          check &= " coordinate: " & $lattice2D.coordinate
          check &= " neighbors: " & $lattice2D.halo.neighbors
          echo check