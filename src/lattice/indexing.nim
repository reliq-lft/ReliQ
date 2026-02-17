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

proc lexToCoords*[D: static[int]](idx: int, geom: array[D, int]): array[D, int] =
  ## Convert lexicographic index to D-dimensional coordinates
  ##
  ## Uses GA C-order convention: last dimension is fastest varying.
  ## ``idx = coords[0] * stride0 + coords[1] * stride1 + ... + coords[D-1]``
  var remaining = idx
  for d in countdown(D-1, 0):
    result[d] = remaining mod geom[d]
    remaining = remaining div geom[d]

proc lexToCoords*[D: static[int]](
  idx: int; 
  geom: array[D, int]; # not used, but needed to disambiguate from geom-based overload
  strides: array[D, int]
): array[D, int] =
  ## Convert lexicographic index to D-dimensional coordinates using precomputed strides
  ##
  ## This version takes precomputed strides for efficiency when called in a
  ## tight loop. The relationship is:
  ## ``idx = coords[0] * strides[0] + coords[1] * strides[1] + ... + coords[D-1] * strides[D-1]``
  var remaining = idx
  for d in 0..<D:
    result[d] = remaining div strides[d]
    remaining = remaining mod strides[d]

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