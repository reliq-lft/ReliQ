import lattice
import field/[simplecubicfield]

export simplecubicfield

type Field* = concept x
  ## Concept interface for field types
  ##
  ## A type satisfies this concept if it has a `lattice` field
  ## that is a `Lattice`.
  x.lattice is Lattice