#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/kokkos/kokkosrng.nim
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

import kokkostypes

when isMainModule:
  import kokkosbase

type KokkosXorShiftRNG*[B: static[int]] = object
  seed: uint64
  when B == 64: 
    handle: KokkosRNG64
  elif B == 1024:
    handle: KokkosRNG1024

#[ constructors/destructors ]#

proc newKokkosXorShiftRNG*(
  bits: static[int]; 
  seed: uint64
): KokkosXorShiftRNG[bits] =
  ## Create a new Kokkos RNG pool with the specified bit size (64 or 1024).
  ##
  ## Params:
  ## - `bits`: Bit size of the RNG (64 or 1024)
  ## - `seed`: Seed for the RNG pool
  ##
  ## Returns:
  ## A pointer to the created Kokkos RNG pool.
  when bits == 64: 
    return KokkosXorShiftRNG[bits](
      seed: seed, 
      handle: newRNG64(seed)
    )
  elif bits == 1024:
    return KokkosXorShiftRNG[bits](
      seed: seed, 
      handle: newRNG1024(seed)
    )
  else: raise newException(ValueError, "unsupported RNG bit size.")

proc `=destroy`*[B: static[int]](rng: var KokkosXorShiftRNG[B]) =
  ## Destroy the Kokkos RNG pool.
  when B == 64: destroyRNG64(rng.handle)
  elif B == 1024: destroyRNG1024(rng.handle)

#[ draw ]#

proc uniform*[B: static[int], T](
  rng: KokkosXorShiftRNG[B];
  start, stop: T
): T =
  ## Draw a uniform random number from the Kokkos RNG pool.
  ##
  ## Params:
  ## - `start`: Start of the uniform distribution
  ## - `stop`: End of the uniform distribution
  ##
  ## Returns:
  ## A random number drawn from the uniform distribution.
  let handle = rng.handle
  when (T is int) or (T is int32):
    {.emit: """
    auto generator = `handle`->get_state();
    `result` = generator.rand(`start`, `stop`);
    `handle`->free_state(generator);
    """.}
  elif T is uint32:
    {.emit: """
    auto generator = `handle`->get_state();
    `result` = generator.urand(`start`, `stop`);
    `handle`->free_state(generator);
    """.}
  elif T is int64:
    {.emit: """
    auto generator = `handle`->get_state();
    `result` = static_cast<int64_t>(generator.rand64(static_cast<uint64_t>(`start`), static_cast<uint64_t>(`stop`)));
    `handle`->free_state(generator);
    """.}
  elif T is uint64:
    {.emit: """
    auto generator = `handle`->get_state();
    `result` = generator.urand64(`start`, `stop`);
    `handle`->free_state(generator);
    """.}
  elif T is float32:
    {.emit: """
    auto generator = `handle`->get_state();
    `result` = generator.frand(`start`, `stop`);
    `handle`->free_state(generator);
    """.}
  elif T is float64:
    {.emit: """
    auto generator = `handle`->get_state();
    `result` = generator.drand(`start`, `stop`);
    `handle`->free_state(generator);
    """.}
  else: {.error: "uniform() only supports float32, float64, int32, uint32, int64, and uint64 types".}

proc gaussian*[B: static[int], T](
  rng: KokkosXorShiftRNG[B];
  mean, sdev: T
): T =
  ## Draw a Gaussian random number from the Kokkos RNG pool.
  ##
  ## Params:
  ## - `mean`: Mean of the Gaussian distribution
  ## - `stddev`: Standard deviation of the Gaussian distribution
  ##
  ## Returns:
  ## A random number drawn from the Gaussian distribution.
  let handle = rng.handle
  {.emit: """
  auto generator = `handle`->get_state();
  `result` = static_cast<`T`>(generator.normal(`mean`, `sdev`));
  `handle`->free_state(generator);
  """.}

when isMainModule:
  initKokkos()

  block:
    let lattice = [8, 8, 8, 16]
    let seed = 123456789'u64
    let rng64 = newKokkosXorShiftRNG(64, seed)
    let rng1024 = newKokkosXorShiftRNG(1024, seed)

    for i in 0..<10:
      echo "uniform (int): ", rng64.uniform(0, 100)
      echo "uniform (float): ", rng1024.uniform(0.0, 1.0)
    
    for i in 0..<10:
      echo "gaussian: ", rng64.gaussian(0.0, 1.0)

  finalizeKokkos()
