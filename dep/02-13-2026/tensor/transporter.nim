#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/tensor/transporter.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
]#

## Transporter - Parallel Transport with MPI Halo Exchange
## ========================================================
##
## This module provides parallel transport operations for lattice gauge theory.
## Unlike stencils (which are local), transporters handle MPI communication
## for accessing neighbors across rank boundaries.
##
## Key concepts:
## - **Shifter**: Shift fields in a direction (no gauge link multiplication)
## - **Transporter**: Parallel transport with gauge link multiplication (covariant)
## - **HaloBuffer**: Communication buffers for boundary exchange
##
## Design Philosophy:
## - Explicit direction handling with compile-time safety
## - Support for forward (+) and backward (-) transport
## - Covariant derivatives: D_μ ψ = (U_μ(x) ψ(x+μ) - ψ(x)) / a
## - AoSoA-aware for SIMD vectorization
## - Multi-device and MPI compatible
##
## Example usage:
## ```nim
## # Create shifters for all directions
## let shifters = newShifters(field, len=1)
##
## # Shift field forward in direction 0
## let shifted = shifters[0] ^* field
##
## # Create transporters with gauge links
## let transporters = newTransporters(gaugeField, field, len=1)
##
## # Parallel transport: U_μ(x) * ψ(x+μ)
## let transported = transporters[0] ^* field
##
## # Covariant derivative in direction μ
## let Dpsi = covariantDerivative(gaugeField, psi, mu)
## ```
##
## Inspired by QEX's transporters but with a cleaner API for LGT.

import lattice/stencil
export stencil  # Export direction types etc.

#[ ============================================================================
   Shift Direction - Enhanced direction type for transport
   ============================================================================ ]#

type
  ShiftDir* = object
    ## Direction and displacement for shifting/transport
    dim*: int      # Dimension (0..D-1)
    len*: int      # Displacement length (+ forward, - backward)

proc shift*(dim: int, len: int = 1): ShiftDir {.inline.} =
  ## Create a shift in dimension `dim` with length `len`
  ShiftDir(dim: dim, len: len)

proc forward*(dim: int): ShiftDir {.inline.} =
  ## Forward shift in dimension `dim` (length +1)
  shift(dim, +1)

proc backward*(dim: int): ShiftDir {.inline.} =
  ## Backward shift in dimension `dim` (length -1)
  shift(dim, -1)

proc `$`*(sd: ShiftDir): string =
  let sign = if sd.len >= 0: "+" else: ""
  "shift[" & $sd.dim & ", " & sign & $sd.len & "]"

#[ ============================================================================
   Halo Region - Describes boundary data needed from neighbors
   ============================================================================ ]#

type
  HaloRegion*[D: static int] = object
    ## Description of a halo region for communication
    dim*: int              # Dimension of the shift
    direction*: int        # +1 (forward) or -1 (backward)
    thickness*: int        # Number of planes in halo
    localGeom*: array[D, int]
    # Region bounds within padded local lattice
    lo*: array[D, int]     # Lower corner (inclusive)
    hi*: array[D, int]     # Upper corner (inclusive)
    nSites*: int           # Number of sites in this halo

proc newHaloRegion*[D: static int](
  dim: int,
  direction: int,
  thickness: int,
  localGeom: array[D, int]
): HaloRegion[D] =
  ## Create a halo region description
  result.dim = dim
  result.direction = direction
  result.thickness = thickness
  result.localGeom = localGeom
  
  # Compute region bounds
  result.nSites = 1
  for d in 0..<D:
    if d == dim:
      if direction > 0:
        # Forward halo: sites beyond local extent (ghost region)
        result.lo[d] = localGeom[d]
        result.hi[d] = localGeom[d] + thickness - 1
      else:
        # Backward halo: sites before local extent (ghost region)
        result.lo[d] = -thickness
        result.hi[d] = -1
      result.nSites *= thickness
    else:
      result.lo[d] = 0
      result.hi[d] = localGeom[d] - 1
      result.nSites *= localGeom[d]

#[ ============================================================================
   Shift Indices - Pre-computed indices for shift operations
   ============================================================================ ]#

type
  ShiftIndices*[D: static int] = object
    ## Pre-computed indices for a shift operation
    ##
    ## For each local site, stores where to get the shifted value:
    ## - If isLocal[i]: source is localIdx[i] (within this rank)
    ## - Else: source is from receive buffer at recvIdx[i]
    dim*: int
    len*: int
    localGeom*: array[D, int]
    nSites*: int
    
    # For local sites: index into local field
    localIdx*: seq[int32]
    isLocal*: seq[bool]
    
    # For boundary sites: info about communication
    sendSites*: seq[int32]    # Local sites to send
    recvSites*: seq[int32]    # Local sites receiving from neighbor
    nSendSites*: int
    nRecvSites*: int

proc computeStrides[D: static int](geom: array[D, int]): array[D, int] =
  result[0] = 1
  for d in 1..<D:
    result[d] = result[d-1] * geom[d-1]

proc lexToCoords[D: static int](idx: int, geom, strides: array[D, int]): array[D, int] =
  var remaining = idx
  for d in countdown(D-1, 0):
    result[d] = remaining div strides[d]
    remaining = remaining mod strides[d]

proc coordsToLex[D: static int](coords: array[D, int], strides: array[D, int]): int =
  for d in 0..<D:
    result += coords[d] * strides[d]

proc newShiftIndices*[D: static int](
  dim: int,
  len: int,
  localGeom: array[D, int]
): ShiftIndices[D] =
  ## Create pre-computed shift indices
  ##
  ## For shift in dimension `dim` by `len` sites.
  ## Positive len = forward shift (access site at x+len)
  ## Negative len = backward shift (access site at x-len)
  result.dim = dim
  result.len = len
  result.localGeom = localGeom
  
  result.nSites = 1
  for d in 0..<D:
    result.nSites *= localGeom[d]
  
  let strides = computeStrides(localGeom)
  
  result.localIdx = newSeq[int32](result.nSites)
  result.isLocal = newSeq[bool](result.nSites)
  result.sendSites = @[]
  result.recvSites = @[]
  
  for site in 0..<result.nSites:
    let coords = lexToCoords(site, localGeom, strides)
    
    var nbrCoords = coords
    nbrCoords[dim] = coords[dim] + len
    
    # Check if neighbor is local (within this rank)
    if nbrCoords[dim] >= 0 and nbrCoords[dim] < localGeom[dim]:
      # Local access
      result.isLocal[site] = true
      result.localIdx[site] = coordsToLex(nbrCoords, strides).int32
    else:
      # Boundary access - need communication
      result.isLocal[site] = false
      result.recvSites.add(site.int32)
      
      # Wrap for periodic BC (determines which site to send)
      nbrCoords[dim] = (nbrCoords[dim] + localGeom[dim]) mod localGeom[dim]
      result.localIdx[site] = coordsToLex(nbrCoords, strides).int32
  
  # Compute send sites (opposite direction)
  for site in 0..<result.nSites:
    let coords = lexToCoords(site, localGeom, strides)
    
    # If len > 0 (forward shift), we send from the "front" face
    # If len < 0 (backward shift), we send from the "back" face
    var needsSend = false
    if len > 0:
      # Forward shift: sites near lower boundary need to be sent backward
      if coords[dim] < abs(len):
        needsSend = true
    else:
      # Backward shift: sites near upper boundary need to be sent forward
      if coords[dim] >= localGeom[dim] - abs(len):
        needsSend = true
    
    if needsSend:
      result.sendSites.add(site.int32)
  
  result.nSendSites = result.sendSites.len
  result.nRecvSites = result.recvSites.len

#[ ============================================================================
   Shifter - Shift fields without gauge link multiplication
   ============================================================================ ]#

type
  Shifter*[D: static int, T] = object
    ## Shifter for moving fields in a direction
    ##
    ## Handles MPI communication for boundary sites.
    ## Use `^*` operator to apply: shifted = shifter ^* field
    dim*: int
    len*: int
    indices*: ShiftIndices[D]
    # Communication buffers (for MPI)
    sendBuf*: seq[T]
    recvBuf*: seq[T]

proc newShifter*[D: static int, T](
  localGeom: array[D, int],
  dim: int,
  len: int = 1
): Shifter[D, T] =
  ## Create a shifter for the given direction and length
  result.dim = dim
  result.len = len
  result.indices = newShiftIndices[D](dim, len, localGeom)
  
  # Allocate communication buffers
  if result.indices.nSendSites > 0:
    result.sendBuf = newSeq[T](result.indices.nSendSites)
  if result.indices.nRecvSites > 0:
    result.recvBuf = newSeq[T](result.indices.nRecvSites)

proc newShifters*[D: static int, T](
  localGeom: array[D, int],
  len: int = 1
): array[D, Shifter[D, T]] =
  ## Create shifters for all D directions
  for d in 0..<D:
    result[d] = newShifter[D, T](localGeom, d, len)

proc newBackwardShifters*[D: static int, T](
  localGeom: array[D, int],
  len: int = 1
): array[D, Shifter[D, T]] =
  ## Create backward shifters for all D directions
  for d in 0..<D:
    result[d] = newShifter[D, T](localGeom, d, -len)

#[ ============================================================================
   Transporter - Parallel transport with gauge links
   ============================================================================ ]#

type
  Transporter*[D: static int, U, F] = object
    ## Transporter for gauge-covariant field shifting
    ##
    ## Multiplies by gauge link during transport:
    ## - Forward: U_μ(x) * ψ(x+μ)
    ## - Backward: U_μ(x-μ)† * ψ(x-μ)
    dim*: int
    len*: int
    indices*: ShiftIndices[D]
    link*: ptr U           # Pointer to gauge link field
    # Communication buffers
    sendBuf*: seq[F]
    recvBuf*: seq[F]

proc newTransporter*[D: static int, U, F](
  localGeom: array[D, int],
  gaugeField: ptr U,
  dim: int,
  len: int = 1
): Transporter[D, U, F] =
  ## Create a transporter for the given direction
  result.dim = dim
  result.len = len
  result.link = gaugeField
  result.indices = newShiftIndices[D](dim, len, localGeom)
  
  if result.indices.nSendSites > 0:
    result.sendBuf = newSeq[F](result.indices.nSendSites)
  if result.indices.nRecvSites > 0:
    result.recvBuf = newSeq[F](result.indices.nRecvSites)

#[ ============================================================================
   Shift Operations - Apply shifters to fields
   ============================================================================ ]#

proc applyShiftLocal*[D: static int, T](
  shifter: Shifter[D, T],
  source: openArray[T],
  dest: var openArray[T]
) =
  ## Apply shift operation (local sites only, no MPI)
  ##
  ## For sites that need data from other ranks, uses periodic wrapping
  ## within the local domain (single-rank mode).
  let indices = shifter.indices
  for site in 0..<indices.nSites:
    dest[site] = source[indices.localIdx[site]]

template `^*`*[D: static int, T](shifter: Shifter[D, T], source: openArray[T]): seq[T] =
  ## Apply shifter to source field (single-rank mode)
  ##
  ## Returns shifted field: result[x] = source[x + shift]
  block:
    var result = newSeq[T](source.len)
    applyShiftLocal(shifter, source, result)
    result

#[ ============================================================================
   Covariant Derivative Operations
   ============================================================================ ]#

type
  CovariantDerivativeDir* = enum
    ## Direction type for covariant derivative
    cdForward,    # D_μ^+ = (U_μ(x) ψ(x+μ) - ψ(x)) / a
    cdBackward,   # D_μ^- = (ψ(x) - U_μ(x-μ)† ψ(x-μ)) / a  
    cdSymmetric   # D_μ = (D_μ^+ + D_μ^-) / 2

proc covariantShiftForward*[T](
  gaugeLink: T,  # U_μ(x)
  psiShifted: T  # ψ(x+μ)
): T {.inline.} =
  ## Forward covariant shift: U_μ(x) * ψ(x+μ)
  ##
  ## This is the basic building block for covariant derivatives
  ## and Wilson-Dirac operators.
  when compiles(gaugeLink * psiShifted):
    gaugeLink * psiShifted
  else:
    # Fallback for non-matrix types
    gaugeLink * psiShifted

proc covariantShiftBackward*[T](
  gaugeLinkShifted: T,  # U_μ(x-μ)†
  psiShifted: T         # ψ(x-μ)
): T {.inline.} =
  ## Backward covariant shift: U_μ(x-μ)† * ψ(x-μ)
  ##
  ## Note: gaugeLinkShifted should already be the adjoint of U_μ(x-μ)
  when compiles(gaugeLinkShifted * psiShifted):
    gaugeLinkShifted * psiShifted
  else:
    gaugeLinkShifted * psiShifted

#[ ============================================================================
   Stencil-based Transport Patterns
   ============================================================================ ]#

type
  TransportPattern*[D: static int] = object
    ## A pattern of transport operations
    ##
    ## Combines stencil geometry with gauge link information
    ## for efficient multi-direction transport.
    stencil*: Stencil[D]
    shiftIndices*: seq[ShiftIndices[D]]

proc newTransportPattern*[D: static int](
  localGeom: array[D, int],
  len: int = 1
): TransportPattern[D] =
  ## Create transport pattern for nearest-neighbor transport
  result.stencil = nearestNeighborStencil[D]()
  result.shiftIndices = newSeq[ShiftIndices[D]](2 * D)
  
  for d in 0..<D:
    result.shiftIndices[2*d] = newShiftIndices[D](d, +len, localGeom)
    result.shiftIndices[2*d + 1] = newShiftIndices[D](d, -len, localGeom)

proc forwardShiftIdx*(pattern: TransportPattern, dim: int): int {.inline.} =
  ## Get index of forward shift indices for dimension dim
  2 * dim

proc backwardShiftIdx*(pattern: TransportPattern, dim: int): int {.inline.} =
  ## Get index of backward shift indices for dimension dim
  2 * dim + 1

#[ ============================================================================
   Tests
   ============================================================================ ]#

when isMainModule:
  import std/unittest

  suite "ShiftDir":
    test "Forward and backward":
      let fwd = forward(2)
      let bwd = backward(2)
      check fwd.dim == 2
      check fwd.len == 1
      check bwd.dim == 2
      check bwd.len == -1

  suite "ShiftIndices":
    test "Create 2D shift indices forward":
      let si = newShiftIndices[2](0, 1, [4, 4])
      check si.nSites == 16
      check si.dim == 0
      check si.len == 1

    test "Local sites identified correctly":
      let si = newShiftIndices[2](0, 1, [4, 4])
      # Site (0,0): +x neighbor is (1,0) which is local
      check si.isLocal[0] == true
      # Site (3,0): +x neighbor is (4,0) = (0,0) wrapped, so boundary
      check si.isLocal[3] == false

    test "Send and receive sites counted":
      let si = newShiftIndices[2](0, 1, [4, 4])
      # For +1 shift in x: boundary sites are at x=3 (4 sites)
      check si.nRecvSites == 4
      # Send sites at x=0 (4 sites)
      check si.nSendSites == 4

    test "4D shift indices":
      let si = newShiftIndices[4](2, 1, [4, 4, 4, 8])
      check si.nSites == 512
      # Boundary in z direction: 4*4*8 = 128 sites
      check si.nRecvSites == 128
      check si.nSendSites == 128

  suite "Shifter":
    test "Create 2D shifter":
      let shifter = newShifter[2, float64]([4, 4], 0, 1)
      check shifter.dim == 0
      check shifter.len == 1

    test "Apply shift to 1D array":
      let shifter = newShifter[1, float64]([8], 0, 1)
      var source = newSeq[float64](8)
      for i in 0..<8:
        source[i] = float64(i)
      
      let shifted = shifter ^* source
      # shift[x+1]: result[0] = source[1], result[1] = source[2], etc.
      check shifted[0] == 1.0
      check shifted[1] == 2.0
      check shifted[6] == 7.0
      check shifted[7] == 0.0  # Wrapped

    test "Apply backward shift":
      let shifter = newShifter[1, float64]([8], 0, -1)
      var source = newSeq[float64](8)
      for i in 0..<8:
        source[i] = float64(i)
      
      let shifted = shifter ^* source
      # shift[x-1]: result[0] = source[7], result[1] = source[0], etc.
      check shifted[0] == 7.0  # Wrapped
      check shifted[1] == 0.0
      check shifted[7] == 6.0

    test "Create all-direction shifters":
      let shifters = newShifters[4, float64]([4, 4, 4, 8], 1)
      for d in 0..<4:
        check shifters[d].dim == d
        check shifters[d].len == 1

  suite "HaloRegion":
    test "Forward halo region":
      let halo = newHaloRegion[2](0, +1, 1, [4, 4])
      check halo.dim == 0
      check halo.direction == +1
      check halo.lo == [4, 0]
      check halo.hi == [4, 3]
      check halo.nSites == 4  # 1 * 4

    test "Backward halo region":
      let halo = newHaloRegion[2](0, -1, 1, [4, 4])
      check halo.lo == [-1, 0]
      check halo.hi == [-1, 3]
      check halo.nSites == 4

    test "Thick halo region":
      let halo = newHaloRegion[2](1, +1, 2, [4, 4])
      check halo.lo == [0, 4]
      check halo.hi == [3, 5]
      check halo.nSites == 8  # 4 * 2

  suite "TransportPattern":
    test "Create nearest-neighbor pattern":
      let pattern = newTransportPattern[4]([4, 4, 4, 8], 1)
      check pattern.stencil.nPoints == 8  # 2 * 4
      check pattern.shiftIndices.len == 8

    test "Forward and backward index accessors":
      let pattern = newTransportPattern[4]([4, 4, 4, 8], 1)
      # Dimension 2 (z)
      let fwdIdx = pattern.forwardShiftIdx(2)
      let bwdIdx = pattern.backwardShiftIdx(2)
      check fwdIdx == 4  # 2*2
      check bwdIdx == 5  # 2*2 + 1
      check pattern.shiftIndices[fwdIdx].len == +1
      check pattern.shiftIndices[bwdIdx].len == -1

  # ==========================================================================
  # Shift Correctness Tests
  # ==========================================================================
  
  suite "Shift Correctness (1D)":
    ## Verifies that shift operations produce correct results on 1D lattices.
    ## Every site stores its coordinate value as a float, so we can verify
    ## that shifting produces the expected permutation.
    
    const N = 8  # 1D lattice size
    
    setup:
      var field: array[N, float64]
      for i in 0..<N: field[i] = float64(i)
    
    test "Forward shift by 1: result[x] = source[x+1]":
      let s = newShifter[1, float64]([N], 0, +1)
      let shifted = s ^* field
      for i in 0..<N-1:
        check shifted[i] == float64(i + 1)
      check shifted[N-1] == 0.0  # Wrapped: (N-1)+1 mod N = 0
    
    test "Backward shift by 1: result[x] = source[x-1]":
      let s = newShifter[1, float64]([N], 0, -1)
      let shifted = s ^* field
      check shifted[0] == float64(N - 1)  # Wrapped: 0-1 mod N = N-1
      for i in 1..<N:
        check shifted[i] == float64(i - 1)
    
    test "Forward then backward = identity":
      let fwd = newShifter[1, float64]([N], 0, +1)
      let bwd = newShifter[1, float64]([N], 0, -1)
      let shifted = fwd ^* field
      let restored = bwd ^* shifted
      for i in 0..<N:
        check restored[i] == field[i]
    
    test "Backward then forward = identity":
      let fwd = newShifter[1, float64]([N], 0, +1)
      let bwd = newShifter[1, float64]([N], 0, -1)
      let shifted = bwd ^* field
      let restored = fwd ^* shifted
      for i in 0..<N:
        check restored[i] == field[i]
    
    test "Shift by 2 = double shift by 1":
      let s1 = newShifter[1, float64]([N], 0, +1)
      let s2 = newShifter[1, float64]([N], 0, +2)
      let doubleShift = s1 ^* (s1 ^* field)
      let singleShift2 = s2 ^* field
      for i in 0..<N:
        check doubleShift[i] == singleShift2[i]
    
    test "Shift by N = identity":
      let s = newShifter[1, float64]([N], 0, N)
      let shifted = s ^* field
      for i in 0..<N:
        check shifted[i] == field[i]
    
    test "Shift by -N = identity":
      let s = newShifter[1, float64]([N], 0, -N)
      let shifted = s ^* field
      for i in 0..<N:
        check shifted[i] == field[i]

  suite "Shift Correctness (2D)":
    ## Verifies shift operations on a 2D lattice using coordinate-encoding.
    ## Each site stores 100*x + y, so we can verify exact neighbor access.
    
    const Lx = 4
    const Ly = 6
    const nSites2D = Lx * Ly
    
    setup:
      # Encode coordinates: field[x + Lx*y] = 100*x + y
      var field2d: array[nSites2D, float64]
      for y in 0..<Ly:
        for x in 0..<Lx:
          field2d[x + Lx * y] = float64(100 * x + y)
    
    test "Shift +x: result[x,y] = source[x+1,y]":
      let s = newShifter[2, float64]([Lx, Ly], 0, +1)
      let shifted = s ^* field2d
      for y in 0..<Ly:
        for x in 0..<Lx:
          let nbrX = (x + 1) mod Lx
          check shifted[x + Lx * y] == float64(100 * nbrX + y)
    
    test "Shift -x: result[x,y] = source[x-1,y]":
      let s = newShifter[2, float64]([Lx, Ly], 0, -1)
      let shifted = s ^* field2d
      for y in 0..<Ly:
        for x in 0..<Lx:
          let nbrX = (x - 1 + Lx) mod Lx
          check shifted[x + Lx * y] == float64(100 * nbrX + y)
    
    test "Shift +y: result[x,y] = source[x,y+1]":
      let s = newShifter[2, float64]([Lx, Ly], 1, +1)
      let shifted = s ^* field2d
      for y in 0..<Ly:
        for x in 0..<Lx:
          let nbrY = (y + 1) mod Ly
          check shifted[x + Lx * y] == float64(100 * x + nbrY)
    
    test "Shift -y: result[x,y] = source[x,y-1]":
      let s = newShifter[2, float64]([Lx, Ly], 1, -1)
      let shifted = s ^* field2d
      for y in 0..<Ly:
        for x in 0..<Lx:
          let nbrY = (y - 1 + Ly) mod Ly
          check shifted[x + Lx * y] == float64(100 * x + nbrY)
    
    test "Shift +x then -x = identity (2D)":
      let fwd = newShifter[2, float64]([Lx, Ly], 0, +1)
      let bwd = newShifter[2, float64]([Lx, Ly], 0, -1)
      let restored = bwd ^* (fwd ^* field2d)
      for i in 0..<nSites2D:
        check restored[i] == field2d[i]
    
    test "Shift +y then -y = identity (2D)":
      let fwd = newShifter[2, float64]([Lx, Ly], 1, +1)
      let bwd = newShifter[2, float64]([Lx, Ly], 1, -1)
      let restored = bwd ^* (fwd ^* field2d)
      for i in 0..<nSites2D:
        check restored[i] == field2d[i]
    
    test "Shift +x and +y commute":
      let sx = newShifter[2, float64]([Lx, Ly], 0, +1)
      let sy = newShifter[2, float64]([Lx, Ly], 1, +1)
      let xy = sy ^* (sx ^* field2d)
      let yx = sx ^* (sy ^* field2d)
      for i in 0..<nSites2D:
        check xy[i] == yx[i]
    
    test "Forward/backward in different dims commute":
      let fwdX = newShifter[2, float64]([Lx, Ly], 0, +1)
      let bwdY = newShifter[2, float64]([Lx, Ly], 1, -1)
      let fwdX_bwdY = bwdY ^* (fwdX ^* field2d)
      let bwdY_fwdX = fwdX ^* (bwdY ^* field2d)
      for i in 0..<nSites2D:
        check fwdX_bwdY[i] == bwdY_fwdX[i]

  suite "Shift Correctness (4D)":
    ## Verifies shift operations on a 4D lattice (typical for LFT).
    ## Encodes coordinates as field[site] = 1000*x + 100*y + 10*z + t.
    
    const L4 = [4, 4, 4, 8]
    const nSites4D = L4[0] * L4[1] * L4[2] * L4[3]
    
    setup:
      var field4d = newSeq[float64](nSites4D)
      let strides4D = computeStrides(L4)
      for t in 0..<L4[3]:
        for z in 0..<L4[2]:
          for y in 0..<L4[1]:
            for x in 0..<L4[0]:
              let site = coordsToLex([x, y, z, t], strides4D)
              field4d[site] = float64(1000 * x + 100 * y + 10 * z + t)
    
    test "Shift +x: check all sites (4D)":
      let s = newShifter[4, float64](L4, 0, +1)
      let shifted = s ^* field4d
      let strides = computeStrides(L4)
      for t in 0..<L4[3]:
        for z in 0..<L4[2]:
          for y in 0..<L4[1]:
            for x in 0..<L4[0]:
              let site = coordsToLex([x, y, z, t], strides)
              let nbrX = (x + 1) mod L4[0]
              let expected = float64(1000 * nbrX + 100 * y + 10 * z + t)
              check shifted[site] == expected
    
    test "Shift -t: check all sites (4D)":
      let s = newShifter[4, float64](L4, 3, -1)
      let shifted = s ^* field4d
      let strides = computeStrides(L4)
      for t in 0..<L4[3]:
        for z in 0..<L4[2]:
          for y in 0..<L4[1]:
            for x in 0..<L4[0]:
              let site = coordsToLex([x, y, z, t], strides)
              let nbrT = (t - 1 + L4[3]) mod L4[3]
              let expected = float64(1000 * x + 100 * y + 10 * z + nbrT)
              check shifted[site] == expected
    
    test "Forward then backward = identity in all 4 dims":
      let strides = computeStrides(L4)
      for d in 0..<4:
        let fwd = newShifter[4, float64](L4, d, +1)
        let bwd = newShifter[4, float64](L4, d, -1)
        let restored = bwd ^* (fwd ^* field4d)
        for i in 0..<nSites4D:
          check restored[i] == field4d[i]
    
    test "All direction shifts commute (4D)":
      ## Test that S_x S_y = S_y S_x for all dimension pairs
      for d1 in 0..<4:
        for d2 in (d1+1)..<4:
          let s1 = newShifter[4, float64](L4, d1, +1)
          let s2 = newShifter[4, float64](L4, d2, +1)
          let path12 = s2 ^* (s1 ^* field4d)
          let path21 = s1 ^* (s2 ^* field4d)
          for i in 0..<nSites4D:
            check path12[i] == path21[i]
    
    test "Plaquette path returns to start (4D)":
      ## A plaquette in the (x,y) plane: +x, +y, -x, -y should give identity
      let fwdX = newShifter[4, float64](L4, 0, +1)
      let fwdY = newShifter[4, float64](L4, 1, +1)
      let bwdX = newShifter[4, float64](L4, 0, -1)
      let bwdY = newShifter[4, float64](L4, 1, -1)
      let plaquette = bwdY ^* (bwdX ^* (fwdY ^* (fwdX ^* field4d)))
      for i in 0..<nSites4D:
        check plaquette[i] == field4d[i]
    
    test "Rectangle path returns to start (4D)":
      ## A 2x1 rectangle: +x, +x, +y, -x, -x, -y should give identity
      let fwdX = newShifter[4, float64](L4, 0, +1)
      let fwdY = newShifter[4, float64](L4, 1, +1)
      let bwdX = newShifter[4, float64](L4, 0, -1)
      let bwdY = newShifter[4, float64](L4, 1, -1)
      let rect = bwdY ^* (bwdX ^* (bwdX ^* (fwdY ^* (fwdX ^* (fwdX ^* field4d)))))
      for i in 0..<nSites4D:
        check rect[i] == field4d[i]

  suite "ShiftIndices Correctness":
    ## Verifies the boundary/local classification and index mapping
    
    test "All local indices point to valid sites":
      let si = newShiftIndices[2](0, 1, [4, 6])
      for site in 0..<si.nSites:
        check si.localIdx[site] >= 0
        check si.localIdx[site] < si.nSites.int32
    
    test "Boundary sites are exactly at the shifted face":
      ## For +x shift on 4x4 grid, boundary sites are x=3 (need x=4 -> wraps)
      let si = newShiftIndices[2](0, 1, [4, 4])
      for site in si.recvSites:
        let x = site.int mod 4
        check x == 3  # Last column in x needs wrapping
    
    test "Send sites are at the opposite face":
      ## For +x shift, send sites are at x=0 (sent to left neighbor)
      let si = newShiftIndices[2](0, 1, [4, 4])
      for site in si.sendSites:
        let x = site.int mod 4
        check x == 0  # First column in x is sent
    
    test "Backward shift boundary at opposite face":
      ## For -x shift, boundary sites are x=0 (need x=-1 -> wraps)
      let si = newShiftIndices[2](0, -1, [4, 4])
      for site in si.recvSites:
        let x = site.int mod 4
        check x == 0  # First column needs wrapping
    
    test "Backward shift send sites at far face":
      ## For -x shift, send sites are at x=3 (sent to right neighbor)
      let si = newShiftIndices[2](0, -1, [4, 4])
      for site in si.sendSites:
        let x = site.int mod 4
        check x == 3  # Last column is sent
    
    test "Shift by 2: boundary is 2 planes thick":
      let si = newShiftIndices[2](0, 2, [6, 4])
      # For +2 shift on Lx=6, sites at x=4,5 need wrapping
      check si.nRecvSites == 2 * 4  # 2 planes * Ly
      for site in si.recvSites:
        let x = site.int mod 6
        check x >= 4
    
    test "Number of recv equals number of send":
      ## On periodic lattice, what goes out must come in
      for d in 0..<4:
        let si = newShiftIndices[4](d, 1, [4, 4, 4, 8])
        check si.nSendSites == si.nRecvSites
    
    test "Local + recv covers all sites":
      ## Every site either gets data locally or from recv buffer
      let si = newShiftIndices[2](0, 1, [4, 4])
      var nLocal = 0
      for site in 0..<si.nSites:
        if si.isLocal[site]: inc nLocal
      check nLocal + si.nRecvSites == si.nSites