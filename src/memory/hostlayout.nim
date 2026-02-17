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

import utils/[private]
import lattice/[indexing]

proc innerBlockSize*(
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
  result = complexFactor + 2 * ghostWidth
  for r in 0..<R: result *= (shape[r] + 2 * ghostWidth)

proc innerPaddedOffset*(
  R: static[int]; 
  shape: openArray[int]; 
  complexFactor: int; 
  ghostWidth: int
): int =
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

proc strides*[D: static[int]](grid: array[D, int]): array[D, int] =
  assert D > 0, "Grid must have at least one dimension"
  result[D-1] = 1
  for d in countdown(D-2, 0): result[d] = result[d+1] * grid[d+1]

proc elementOffsets*[R: static[int]](
  shape: array[R, int]; 
  isComplex: bool; 
  ghostWidth: int = 1
): seq[int] = 
  ## Build the elemOffsets table for a tensor with given shape and type.
  ##
  ## GA's inner dimensions [S0, S1, ..., S_{R-1}, Cplx] are row-major
  ## with ghost padding innerGhostWidth on every inner dimension.
  ## Returns offsets[0..elemsPerSite-1] where each entry is the flat
  ## offset from the site's base pointer (after innerPaddedOffset) to
  ## the e-th real element.
  let cf = (if isComplex: 2 else: 1)
  let paddedCf = cf + 2 * ghostWidth
  let elementsPerSite = product(shape) * paddedCf

  # calculate stride
  var strides: array[R+1, int]
  strides[R] = 1
  for r in countdown(R-1, 0):
    let extent = (if r == R-1: paddedCf else: shape[r+1] + 2 * ghostWidth)
    strides[r] = strides[r+1] * extent
  
  # calculate offsets
  result = newSeq[int](elementsPerSite)
  for elem in 0..<elementsPerSite:
    var remaining = elem
    var idx: array[R, int]
    let cPart = remaining mod cf

    # decompose elem into row-major order
    remaining = remaining div cf
    for r in countdown(R-1, 0):
      idx[r] = remaining mod shape[r]
      remaining = remaining div shape[r]
    
    # calculate offset using padded strides
    var offset = 0
    for r in 0..<R: offset += idx[r] * strides[r]
    result[elem] = offset

proc siteOffset*[D: static[int]](
  coords: array[D, int]; 
  geom: array[D, int]; 
  innerBlockSize: int
): int =
  ## Compute the flat offset for a given lattice site in the non-padded layout
  ##
  ## This is used for indexing into the local data pointer when there is no
  ## ghost padding. The offset is simply the lexicographic index of the site
  ## multiplied by the inner block size (number of elements per site).
  var offset = 0
  var stride = innerBlockSize
  for d in countdown(D-1, 0):
    offset += coords[d] * stride
    stride *= geom[d]

proc siteOffsets*[R: static[int]](
  grid: array[R, int];
  innerBlockSize: int
): seq[int] =
  let strides = strides(grid)
  let numSites = product(grid)

  result = newSeq[int](numSites)
  for n in 0..<numSites: # TODO: target for threading
    let coords = n.lexToCoords(grid, strides)
    result[n] = coords.siteOffset(grid, innerBlockSize)