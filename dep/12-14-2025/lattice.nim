import reliq
import lattice/[simplecubiclattice]

export simplecubiclattice

type Lattice* = concept x
  ## Concept interface for lattice types
  ##
  ## A type satisfies this concept if it has a `dimensions` field
  ## that is an array of integers.
  x.dimensions is array

test:
  let lattice = newSimpleCubicLattice([8, 8, 8, 16])
  assert(lattice is Lattice, "SimpleCubicLattice not Lattice")