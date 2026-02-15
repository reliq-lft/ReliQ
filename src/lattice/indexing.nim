#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/lattice/indexing.nim
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

## GA memory layout
## ================
## 
## GA's C API uses row-major (C-order) storage: the LAST dimension is the
## fastest varying in memory.  For a 7-dimensional GA with dims
## [d0, d1, d2, d3, d4, d5, d6], element (i0,i1,...,i6) is at flat index:
##   i0 * (d1*d2*d3*d4*d5*d6) + i1 * (d2*d3*d4*d5*d6) + ... + i6
##
## For a TensorField[D=4, R=2, T=float64] with shape [1,1], the GA has 7
## dimensions: [Lx, Ly, Lz, Lt, S0, S1, Cplx].
## The last R+1 dimensions are "inner" (tensor shape + complex component).
## For scalar real fields all inner dims are 1, so the inner stride factor is 1
## and flat lattice indexing works directly.
##
## For ghost-padded arrays the inner dims get padded too (ghost width ≥ 1 is
## required for ALL GA dimensions). To index into the ghost-padded array for
## a lattice site we must:
##   1. Compute the inner offset once (center of the padded inner dims)
##   2. Multiply each lattice stride by the inner block size

proc innerPaddedBlockSize*(
  R: static[int], 
  shape: openArray[int], 
  complexFactor: int, 
  ghostWidth: int
): int =
  ## Compute the padded inner block size for ghost-padded arrays
  ##
  ## Each inner dimension of size S gets padded to ``S + 2*ghostWidth``.
  ## The block size is the product of all padded inner dimensions
  ## (R tensor dimensions + 1 complex/real dimension).
  result = 1
  for r in 0..<R: result *= (shape[r] + 2 * ghostWidth)
  result *= (complexFactor + 2 * ghostWidth)

proc innerPaddedOffset*(R: static[int], shape: openArray[int], complexFactor: int, ghostWidth: int): int =
  ## Compute the flat offset to the center element of the padded inner block
  ##
  ## In the ghost-padded inner block, each inner dimension of padded size P_i
  ## has the real data at index ``ghostWidth``.  The offset from the start
  ## of the padded block to the center (first real element) is:
  ## ``sum_{i=0..R} ghostWidth * stride_i``
  ## where stride_i is the product of padded sizes of dimensions i+1..R.
  var paddedSizes: seq[int] = @[]
  var stride = 1
  
  # Build padded sizes for all R+1 inner dims (R tensor dims + 1 complex dim)
  for r in 0..<R: paddedSizes.add(shape[r] + 2 * ghostWidth)
  paddedSizes.add(complexFactor + 2 * ghostWidth)

  # Accumulate offset in row-major order (last dim fastest)
  result = 0
  for i in countdown(paddedSizes.len - 1, 0):
    result += ghostWidth * stride
    stride *= paddedSizes[i]

proc lexToCoords*[D: static[int]](idx: int, geom: array[D, int]): array[D, int] =
  ## Convert lexicographic index to D-dimensional coordinates
  ##
  ## Uses GA C-order convention: last dimension is fastest varying.
  ## ``idx = coords[0] * stride0 + coords[1] * stride1 + ... + coords[D-1]``
  var remaining = idx
  for d in countdown(D-1, 0):
    result[d] = remaining mod geom[d]
    remaining = remaining div geom[d]

proc coordsToLex*[D: static[int]](coords: array[D, int], geom: array[D, int]): int =
  ## Convert D-dimensional coordinates to lexicographic index
  ##
  ## GA C-order: last dimension is fastest varying.
  result = 0
  var stride = 1
  for d in countdown(D-1, 0):
    result += coords[d] * stride
    stride *= geom[d]

proc coordsToPaddedLex*[D: static[int]](
  coords: array[D, int],
  paddedGeom: array[D, int],
  ghostWidth: array[D, int],
  innerBlockSize: int
): int =
  ## Convert local lattice coordinates to a flat index in the ghost-padded array
  ##
  ## The padded array has D lattice dimensions (each padded by ``2*ghostWidth``)
  ## followed by R+1 inner dimensions (also padded). This function computes
  ## the index for the center element of the inner block at the given lattice site.
  ##
  ## Parameters:
  ## - coords: Local lattice coordinates (can be negative for backward ghost)
  ## - paddedGeom: Padded lattice geometry (``localGeom + 2*ghostWidth`` per dim)
  ## - ghostWidth: Ghost width per lattice dimension
  ## - innerBlockSize: Product of all padded inner dimensions
  ##
  ## Returns:
  ## Flat index into the padded data array (not including inner offset)
  result = 0
  var stride = innerBlockSize
  for d in countdown(D-1, 0):
    result += (coords[d] + ghostWidth[d]) * stride
    stride *= paddedGeom[d]

proc localSiteOffset*[D: static[int]](
  coords: array[D, int],
  paddedGeom: array[D, int],
  innerBlockSize: int
): int =
  ## Convert local lattice coordinates to a flat offset in the local data pointer
  ##
  ## The local pointer from NGA_Access points into the padded memory at the start
  ## of the local region. The memory layout still uses padded strides for the
  ## lattice dimensions (which carry ghost regions). The inner block at each
  ## lattice site is contiguous with ``innerBlockSize`` elements.
  ##
  ## Parameters:
  ## - coords: Local lattice coordinates
  ## - paddedGeom: Padded lattice geometry (``localGeom + 2*ghostWidth`` per dim)
  ## - innerBlockSize: Number of contiguous inner elements per lattice site
  ##
  ## Returns:
  ## Flat offset from local pointer p[0] to the start of the inner block
  ## for the given site.
  result = 0
  var stride = innerBlockSize
  for d in countdown(D-1, 0):
    result += coords[d] * stride
    stride *= paddedGeom[d]