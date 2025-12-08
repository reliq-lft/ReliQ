type SimpleCubicLattice*[D: static[int]] = object
  ## Simple cubic lattice implementation
  ##
  ## Represents a simple cubic lattice with equal dimensions in all directions.
  dimensions*: array[D, int]
  mpiGrid*: array[D, int]
  ghostGrid*: array[D, int]

proc defaultMPIGrid[D: static[int]](): array[D, int] =
  for i in 0..<D: result[i] = -1

proc defaultGhostGrid[D: static[int]](): array[D, int] =
  for i in 0..<D: result[i] = 0

proc newSimpleCubicLattice*[D: static[int]](
  dimensions: array[D, int],
  mpiGrid: array[D, int] = defaultMPIGrid[D](),
  ghostGrid: array[D, int] = defaultGhostGrid[D]()
): SimpleCubicLattice[D] =
  ## Create a new SimpleCubicLattice
  ##
  ## Parameters:
  ## - `dims`: Dimensions of the lattice in each direction
  ## - `mpiGrid`: MPI grid configuration
  ## - `ghostGrid`: Ghost cell configuration
  ##
  ## Returns:
  ## A new SimpleCubicLattice instance
  ## 
  ## Example:
  ## ```nim
  ## let lattice = newSimpleCubicLattice(
  ##   [8, 8, 8, 16], 
  ##   [1, 1, 1, numRanks()], 
  ##   [1, 1, 1, 1]
  ## )
  ## ```
  ## 
  ## Note:
  ## Last two parameters have default values for convenience. If not specified, 
  ## `mpiGrid` uses algorithm provided by GlobalArrays for arranging lattice into
  ## MPI ranks. `ghostGrid` defaults to no ghost cells.
  SimpleCubicLattice[D](
    dimensions: dimensions, 
    mpiGrid: mpiGrid, 
    ghostGrid: ghostGrid
  )