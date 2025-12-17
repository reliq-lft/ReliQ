#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/bridge/localarray.nim
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

import globalarrays
import kokkos

import globalarrays/[gawrap]
import utils/[complex]

type LocalArray*[D: static[int], T] = object
  ## Wrapper around Kokkos View for local portion of GlobalArray
  ##
  ## Holds both the view and necessary metadata for accessing
  ## the local portion of a distributed GlobalArray.
  lo: array[D, cint]
  hi: array[D, cint]
  ld: array[D-1, cint]
  released: bool
  when isComplex(T):
    handleRe: cint
    handleIm: cint
    dataRe: ptr UncheckedArray[float64]
    dataIm: ptr UncheckedArray[float64]
    view: ComplexView[D, float64]
  else:
    handle: cint
    data: ptr UncheckedArray[T]
    view: View[D, T]

#[ constructor ]#

proc newLocalArray*[D: static[int], T](ga: GlobalArray[D, T]): LocalArray[D, T] =
  ## Create a new real LocalArray from a GlobalArray
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray to create the LocalArray from
  ##
  ## Returns:
  ## A new LocalArray instance
  ##
  ## Example:
  ## ```nim
  ## let lattice = newSimpleCubicLattice([8, 8, 8, 16])
  ## let field = lattice.newField: float
  ## let localView = field.newLocalArray
  ## ```
  var lo: array[D, cint]
  var hi: array[D, cint]
  var ld: array[D-1, cint]
  var p: pointer

  var latticeGrid: array[D, int]
  var ghostGrid: array[D, int]

  NGA_Distribution(ga.getHandle(), GA_Nodeid(), addr lo[0], addr hi[0])
  NGA_Access(ga.getHandle(), addr lo[0], addr hi[0], addr p, addr ld[0])
  
  # Debug: print memory address and range info
  echo "Process ", GA_Nodeid(), ": memory ptr=", cast[uint](p), " range=[", lo[0], ",", lo[1], ",", lo[2], ",", lo[3], "] to [", hi[0], ",", hi[1], ",", hi[2], ",", hi[3], "]"

  for i in 0..<D: 
    latticeGrid[i] = int(hi[i] - lo[i] + 1)
  
  result = LocalArray[D, T](
    lo: lo,
    hi: hi,
    ld: ld,
    released: false,
    handle: ga.getHandle(),
    data: cast[ptr UncheckedArray[T]](p)
  )

  result.view = newView(result.data, latticeGrid, ghostGrid)

proc newLocalArray*[D: static[int], T](
  gaRe: GlobalArray[D, T],
  gaIm: GlobalArray[D, T]
): LocalArray[D, Complex[T]] =
  ## Create a new complex LocalArray from a GlobalArray
  ##
  ## Parameters:
  ## - `gaRe`: The GlobalArray for the real part to create the LocalArray from
  ## - `gaIm`: The GlobalArray for the imaginary part to create the LocalArray from
  ##
  ## Returns:
  ## A new LocalArray instance
  ##
  ## Example:
  ## ```nim
  ## let lattice = newSimpleCubicLattice([8, 8, 8, 16])
  ## let field = lattice.newField: float
  ## let localView = field.newLocalArray
  ## ```
  var lo: array[D, cint]
  var hi: array[D, cint]
  var ld: array[D-1, cint]
  var pRe: pointer
  var pIm: pointer

  var latticeGrid: array[D, int]
  var ghostGrid: array[D, int]

  NGA_Distribution(gaRe.getHandle(), GA_Nodeid(), addr lo[0], addr hi[0])
  
  NGA_Access(gaRe.getHandle(), addr lo[0], addr hi[0], addr pRe, addr ld[0])
  NGA_Access(gaIm.getHandle(), addr lo[0], addr hi[0], addr pIm, addr ld[0])

  for i in 0..<D: 
    latticeGrid[i] = int(hi[i] - lo[i] + 1)

  result = LocalArray[D, Complex[T]](
    lo: lo,
    hi: hi,
    ld: ld,
    released: false,
    handleRe: gaRe.getHandle(),
    handleIm: gaIm.getHandle(),
    dataRe: cast[ptr UncheckedArray[T]](pRe),
    dataIm: cast[ptr UncheckedArray[T]](pIm)
  )

  result.view = newView(result.dataRe, result.dataIm, latticeGrid, ghostGrid)

#[ move semantics ]#

proc `=destroy`*[D: static[int], T](la: LocalArray[D, T]) =
  ## Destructor for LocalArray
  ##
  ## Releases the access to the underlying GlobalArray
  ## Parameters:
  ## - `la`: The LocalArray instance to destroy
  if not la.released:
    when isComplex(T):
      if la.handleRe > 0: NGA_Release(la.handleRe, addr la.lo[0], addr la.hi[0])
      if la.handleIm > 0: NGA_Release(la.handleIm, addr la.lo[0], addr la.hi[0])
    else: 
      if la.handle > 0: NGA_Release(la.handle, addr la.lo[0], addr la.hi[0])

proc `=copy`*[D: static[int], T](dest: var LocalArray[D, T], src: LocalArray[D, T]) {.error.}

#[ accessors ]#

proc numSites*[D: static[int], T](la: LocalArray[D, T]): int =
  ## Get the number of local sites in the LocalArray
  ##
  ## Parameters:
  ## - `la`: The LocalArray instance
  ##
  ## Returns:
  ## The number of local sites
  result = 1
  for i in 0..<D:
    result *= int(la.hi[i] - la.lo[i] + 1)

template `[]`*[D: static[int], T](la: LocalArray[D, T]; n: SomeInteger): T =
  la.view[n]

template `[]=`*[D: static[int], T](
  la: LocalArray[D, T]; 
  n: SomeInteger; 
  value: T
) = la.view[n] = value

template `[]=`*[D: static[int], T](
  la: LocalArray[D, T]; 
  n: SomeInteger; 
  value: SomeNumber
) = la.view[n] = value

#[ unit test ]#

when isMainModule:
  block:
    initGlobalArrays()
    initKokkos()

    let latticeGrid = [8, 8, 8, 8*GA_Nnodes()]
    let mpiGrid = [1, 1, 1, GA_Nnodes()]
    let ghostGrid = [0, 0, 0, 0]
    let gaRe = newGlobalArray(latticeGrid, mpiGrid, ghostGrid): float64
    let gaIm = newGlobalArray(latticeGrid, mpiGrid, ghostGrid): float64
    var localView = newLocalArray(gaRe)

    for i in 0..<localView.numSites():
      localView[i] = float64(i + 1)

    for i in 0..<localView.numSites():
      assert localView[i] == float64(i + 1)

    var localViewC = newLocalArray(gaRe, gaIm)

    for i in 0..<localViewC.numSites():
      localViewC[i] = complex(float64(i + 1), float64(-(i + 1)))

    for i in 0..<localViewC.numSites():
      assert localViewC[i].re == float64(i + 1)
      assert localViewC[i].im == float64(-(i + 1))
    
    for i in 0..<localViewC.numSites():
      localViewC[i] = float64(i + 2)
    
    for i in 0..<localViewC.numSites():
      assert localViewC[i].re == float64(i + 2)
      assert localViewC[i].im == float64(0)

    finalizeKokkos()
    finalizeGlobalArrays()