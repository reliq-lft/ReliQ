#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/lattice/simplecubiclattice.nim
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

type SimpleCubicLattice*[D: static[int]] = object
  ## Simple cubic lattice implementation
  ##
  ## Represents a simple cubic lattice with equal dimensions in all directions.
  globalGrid*: array[D, int]
  mpiGrid*: array[D, int]
  ghostGrid*: array[D, int]

proc defaultMPIGrid[D: static[int]](): array[D, int] =
  for i in 0..<D: result[i] = -1

proc defaultGhostGrid[D: static[int]](): array[D, int] =
  for i in 0..<D: result[i] = 0

proc newSimpleCubicLattice*[D: static[int]](
  globalGrid: array[D, int],
  mpiGrid: array[D, int] = defaultMPIGrid[D](),
  ghostGrid: array[D, int] = defaultGhostGrid[D]()
): SimpleCubicLattice[D] =
  ## Create a new SimpleCubicLattice
  ##
  ## Parameters:
  ## - `globalGrid`: Dimensions of the lattice in each direction
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
    globalGrid: globalGrid, 
    mpiGrid: mpiGrid, 
    ghostGrid: ghostGrid
  )