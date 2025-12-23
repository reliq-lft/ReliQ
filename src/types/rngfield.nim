#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/types/rngfield.nim
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

import reliq

type RandomNumberGenerator = enum
  rkKokkosXorShift64,
  rkKokkosXorShift1024

type
  RandomNumberGeneratorField[R: static[RandomNumberGenerator], D: static[int]] = object
    lattice*: SimpleCubicLattice[D]
    seed*: uint64
    when R == rkKokkosXorShift64:
      rng*: ptr UncheckedArray[KokkosXorShiftRNG[64]]
    elif R == rkKokkosXorShift1024:
      rng*: ptr UncheckedArray[KokkosXorShiftRNG[1024]]
    else:
      rng*: ptr UncheckedArray[KokkosXorShiftRNG[1024]]

#[ constructor ]#

template newRandomNumberGeneratorField*[R: static[RandomNumberGenerator]](
  lattice: SimpleCubicLattice;
  globalSeed: uint64
): RandomNumberGeneratorField[R, lattice.D] =
  ## Create a new RNG field for the given lattice and RNG type.
  ##
  ## Params:
  ## - `lattice`: The lattice for which to create the RNG field
  ## - `globalSeed`: The global seed for the RNGs
  ##
  ## Returns:
  ## A new RandomNumberGeneratorField instance
  var localDimensions: array[lattice.D, int]
  let dimensions = lattice.dimensions
  let mpiGrid = lattice.mpiGrid
  let ghostGrid = lattice.ghostGrid
  let dummy = newGlobalArray(dimensions, mpiGrid, ghostGrid): float64
  let numSites = dummy.numSites()
  let (lo, hi) = dummy.getBounds()

  var result = RandomNumberGeneratorField[R, lattice.D](lattice: lattice, seed: globalSeed)
  for i in 0..<lattice.D: localDimensions[i] = hi[i] - lo[i] + 1

  when R == rkKokkosXorShift64:
    let size = numSites * sizeof(KokkosXorShiftRNG[64])
    result.rng = cast[ptr UncheckedArray[KokkosXorShiftRNG[64]]](alloc(size))    
    for n in 0..<numSites:
      let localCoords = flatToCoords(n, localDimensions)
      let globalCoords = localToGlobalCoords(localCoords, lo)
      let localSeed = siteSeed(globalSeed, globalCoords)
      result.rng[n] = newKokkosXorShiftRNG(64, localSeed)
  else:
    let size = numSites * sizeof(KokkosXorShiftRNG[1024])
    result.rng = cast[ptr UncheckedArray[KokkosXorShiftRNG[1024]]](alloc(size))
    for n in 0..<numSites:
      let localCoords = flatToCoords(n, localDimensions)
      let globalCoords = localToGlobalCoords(localCoords, lo)
      let localSeed = siteSeed(globalSeed, globalCoords)
      result.rng[n] = newKokkosXorShiftRNG(1024, localSeed)
  
  result

#[ sampling ]#

proc uniform*[R: static[RandomNumberGenerator], D: static[int], T](
  rngField: RandomNumberGeneratorField[R, D];
  n: SomeInteger,
  start, stop: T
): T {.inline.} =
  ## Generate a uniform random number in [start, stop) for site n
  ##
  ## Params:
  ## - `rngField`: The RNG field
  ## - `n`: The flat index of the lattice site
  ## - `start`: The start of the uniform range
  ## - `stop`: The end of the uniform range
  ##
  ## Returns:
  ## A uniform random number in [start, stop)
  when R == rkKokkosXorShift64: return rngField.rng[n].uniform(start, stop)
  else: return rngField.rng[n].uniform(start, stop)

proc gaussian*[R: static[RandomNumberGenerator], D: static[int]](
  rngField: RandomNumberGeneratorField[R, D];
  n: SomeInteger;
  T: typedesc
): T {.inline.} =
  ## Generate a Gaussian random number with given mean and stddev for site n
  ##
  ## Params:
  ## - `rngField`: The RNG field
  ## - `n`: The flat index of the lattice site
  ## - `mean`: The mean of the Gaussian distribution
  ## - `sdev`: The standard deviation of the Gaussian distribution
  ##
  ## Returns:
  ## A Gaussian random number
  let mean = 0.0
  let sdev = 1.0
  when isComplex(T):
    let realPart = rngField.rng[n].gaussian(mean, sdev)
    let imagPart = rngField.rng[n].gaussian(mean, sdev)
    when isComplex32(T):
      return complex(realPart.float32, imagPart.float32)
    elif isComplex64(T):
      return complex(realPart.float64, imagPart.float64)
  else:
    when rngField.R == rkKokkosXorShift64: 
      return rngField.rng[n].gaussian(mean, sdev)
    else: return rngField.rng[n].gaussian(mean, sdev)

#[ unit tests ]#

test:
  let dimensions = [8, 8, 8, 16]
  let lattice = dimensions.newSimpleCubicLattice()
  var rngField1 = newRandomNumberGeneratorField[rkKokkosXorShift64](lattice, 123456789'u64)
  var rngField2 = newRandomNumberGeneratorField[rkKokkosXorShift1024](lattice, 123456789'u64)

  # Get the number of local sites for this process
  let mpiGrid = lattice.mpiGrid
  let ghostGrid = lattice.ghostGrid
  let dummy = newGlobalArray(dimensions, mpiGrid, ghostGrid): float64
  let numLocalSites = dummy.numSites()

  # Test using local indices only
  for i in every 0..<numLocalSites:
    let r1 = rngField1.uniform(i, 0.0'f64, 1.0'f64)
    let r2 = rngField2.uniform(i, 0.0'f64, 1.0'f64)
    assert r1 >= 0.0'f64 and r1 < 1.0'f64
    assert r2 >= 0.0'f64 and r2 < 1.0'f64

  # Test Gaussian sampling
  for i in every 0..<numLocalSites:
    let g1 = rngField1.gaussian(i): float64
    let g2 = rngField2.gaussian(i): Complex64