#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/lattice/latticeconcept.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
]#

import simplecubiclattice

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

proc numSites*[D: static[int], L: Lattice[D]](lat: L): int =
  ## Total number of local lattice sites on this MPI rank.
  ## When mpiGrid has valid values (> 0), divides globalGrid by mpiGrid.
  ## When mpiGrid has auto-detect sentinels (<= 0), returns globalVolume
  ## (correct for single-rank; multi-rank code should use view/stencil counts).
  result = 1
  for d in 0..<D:
    if lat.mpiGrid[d] > 0: result *= lat.globalGrid[d] div lat.mpiGrid[d]
    else: result *= lat.globalGrid[d]

proc globalVolume*[D: static[int], L: Lattice[D]](lat: L): int =
  ## Total global lattice volume (same as numSites for single-rank)
  result = 1
  for d in 0..<D:
    result *= lat.globalGrid[d]

template all*(lat: untyped): untyped =
  ## Returns a range over all local sites: ``0 ..< numSites``.
  ## Use with ``each`` and ``reduce`` loops:
  ##   ``for n in each lattice.all:``
  0 ..< lat.numSites()

template newPaddedLattice*[D: static[int], L: Lattice[D]](lat: L; ghostGrid: array[D, int]): untyped =
  ## Create a new lattice with the same global and MPI grid as `lat` but with the specified `ghostGrid`.
  ## This is used for creating padded lattices for stencil access.
  when L is SimpleCubicLattice[D]: 
    newSimpleCubicLattice(lat.globalGrid, lat.mpiGrid, ghostGrid)
  else: L(globalGrid: lat.globalGrid, mpiGrid: lat.mpiGrid, ghostGrid: ghostGrid)
