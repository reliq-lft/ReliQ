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

proc flatToCoords*[D: static[int]](
  idx: int,
  dims: array[D, int]
): array[D, int] =
  ## Convert flat index to D-dimensional coordinates
  ##
  ## Parameters:
  ## - `idx`: Flat index (0..numSites-1)
  ## - `dims`: Dimensions of the local lattice portion
  ##
  ## Returns:
  ## Array of D coordinates
  var remaining = idx
  for i in countdown(D-1, 0):
    result[i] = remaining mod dims[i]
    remaining = remaining div dims[i]

proc coordsToFlat*[D: static[int]](
  coords: array[D, int],
  dims: array[D, int]
): int =
  ## Convert D-dimensional coordinates to flat index
  ##
  ## Parameters:
  ## - `coords`: Array of D coordinates
  ## - `dims`: Dimensions of the local lattice portion
  ##
  ## Returns:
  ## Flat index
  result = 0
  var stride = 1
  for i in countdown(D-1, 0):
    result += coords[i] * stride
    stride *= dims[i]

proc localToGlobalCoords*[D: static[int]](
  localCoords: array[D, int],
  lo: array[D, int]
): array[D, int] =
  ## Convert local coordinates to global coordinates
  ##
  ## Parameters:
  ## - `localCoords`: Local coordinates within the process's portion
  ## - `lo`: Lower bounds of this process's global array portion
  ##
  ## Returns:
  ## Global coordinates
  for i in 0..<D:
    result[i] = localCoords[i] + lo[i]

proc globalToLocalCoords*[D: static[int]](
  globalCoords: array[D, int],
  lo: array[D, int]
): array[D, int] =
  ## Convert global coordinates to local coordinates
  ##
  ## Parameters:
  ## - `globalCoords`: Global coordinates in the full lattice
  ## - `lo`: Lower bounds of this process's global array portion
  ##
  ## Returns:
  ## Local coordinates within this process's portion
  for i in 0..<D:
    result[i] = globalCoords[i] - lo[i]

proc localFlatToGlobalFlat*[D: static[int]](
  localIdx: int,
  localDims: array[D, int],
  globalDims: array[D, int],
  lo: array[D, int]
): int =
  ## Convert local flat index to global flat index
  ##
  ## This function converts a flat index within a process's local portion
  ## to the corresponding flat index in the global lattice.
  ##
  ## Parameters:
  ## - `localIdx`: Flat index within local portion (0..numLocalSites-1)
  ## - `localDims`: Dimensions of the local lattice portion
  ## - `globalDims`: Dimensions of the full global lattice
  ## - `lo`: Lower bounds of this process's global array portion
  ##
  ## Returns:
  ## Global flat index
  ##
  ## Example:
  ## ```nim
  ## # For a process with lo = [0, 8], localDims = [4, 4], globalDims = [4, 16]
  ## # local index 0 -> local coords [0, 0] -> global coords [0, 8] -> global index 8
  ## let globalIdx = localFlatToGlobalFlat(0, [4, 4], [4, 16], [0, 8])
  ## ```
  let localCoords = flatToCoords(localIdx, localDims)
  let globalCoords = localToGlobalCoords(localCoords, lo)
  result = coordsToFlat(globalCoords, globalDims)

proc globalFlatToLocalFlat*[D: static[int]](
  globalIdx: int,
  localDims: array[D, int],
  globalDims: array[D, int],
  lo: array[D, int]
): int =
  ## Convert global flat index to local flat index
  ##
  ## This function converts a flat index in the global lattice to the
  ## corresponding flat index within a process's local portion.
  ##
  ## Parameters:
  ## - `globalIdx`: Flat index in global lattice (0..numGlobalSites-1)
  ## - `localDims`: Dimensions of the local lattice portion
  ## - `globalDims`: Dimensions of the full global lattice
  ## - `lo`: Lower bounds of this process's global array portion
  ##
  ## Returns:
  ## Local flat index within this process's portion
  let globalCoords = flatToCoords(globalIdx, globalDims)
  let localCoords = globalToLocalCoords(globalCoords, lo)
  result = coordsToFlat(localCoords, localDims)