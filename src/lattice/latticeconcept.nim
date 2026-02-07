#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/lattice/latticeconcept.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
]#

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
