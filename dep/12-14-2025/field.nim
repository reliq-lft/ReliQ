import lattice
import field/[scalarfield, tensorfield]

export scalarfield
export tensorfield

type Field* = concept x
  ## Concept interface for field types
  ##
  ## A type satisfies this concept if it has a `lattice` field
  ## that is a `Lattice`.
  x.lattice is Lattice