#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/lattice.nim
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

include indexing
include simplecubiclattice

## Lattice Concept
## ===============
##
## Defines the `Lattice` concept that any lattice type must satisfy.
## This is in its own module to avoid circular dependencies between
## lattice.nim and lattice/stencil.nim.

type Lattice*[D: static[int]] = concept x
  ## Lattice concept
  ##
  ## A concept that any lattice type must satisfy.
  ##
  ## Requirements:
  ## - Must have `globalGrid`, `mpiGrid`, and `ghostGrid` fields of type `array[D, int]`.
  x.globalGrid is array[D, int]
  x.mpiGrid is array[D, int]
  x.ghostGrid is array[D, int]

proc numGlobalSites*[D: static[int], L: Lattice[D]](lat: L): int =
  ## Total global lattice volume (same as numSites for single-rank)
  result = 1
  for d in 0..<D:
    result *= lat.globalGrid[d]

template all*(lat: untyped): untyped =
  ## Returns a range over all local sites: ``0 ..< numSites``.
  ## Use with ``each`` and ``reduce`` loops:
  ##   ``for n in each lattice.all:``
  0..<lat.numLocalSites()

proc newPaddedLattice*[D: static[int], L: Lattice[D]](lat: L; ghostGrid: array[D, int]): L =
  ## Create a new lattice with the same global and MPI grid as `lat` but with the specified `ghostGrid`.
  ## This is used for creating padded lattices for stencil access.
  when L is SimpleCubicLattice[D]: 
    newSimpleCubicLattice(lat.globalGrid, lat.mpiGrid, ghostGrid)
  else: L(globalGrid: lat.globalGrid, mpiGrid: lat.mpiGrid, ghostGrid: ghostGrid)