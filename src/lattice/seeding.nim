#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/lattice/seeding.nim
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

proc siteSeed*[D: static[int]](
  globalSeed: uint64,
  globalCoords: array[D, int]
): uint64 =
  ## Generate a deterministic per-site seed from global seed and site coordinates.
  ## Uses a simple hash function based on the FNV-1a algorithm.
  ##
  ## This ensures:
  ## - Each site gets a unique seed
  ## - The seed is deterministic and reproducible
  ## - The seed is independent of MPI rank decomposition
  ##
  ## Parameters:
  ## - `globalSeed`: The master seed for the entire simulation
  ## - `globalCoords`: Global coordinates of the lattice site
  ##
  ## Returns:
  ## A unique uint64 seed for this site
  
  # FNV-1a hash constants (64-bit)
  const FNV_PRIME = 1099511628211'u64
  const FNV_OFFSET = 14695981039346656037'u64
  
  result = FNV_OFFSET
  
  # Hash in the global seed - process each byte
  # FNV-1a: XOR then multiply for each octet
  for i in 0..<8:
    let octet = uint64((globalSeed shr (i * 8)) and 0xFF)
    result = result xor octet
    result = result * FNV_PRIME
  
  # Hash in each coordinate - process each byte of the int
  for coord in globalCoords:
    # Convert coord to uint64 (handle negative via two's complement)
    let coordU64 = cast[uint64](int64(coord))
    # Process each of the 8 bytes
    for i in 0..<8:
      let octet = (coordU64 shr (i * 8)) and 0xFF
      result = result xor octet
      result = result * FNV_PRIME

when isMainModule:
  import std/sets
  
  echo "Testing FNV-1a implementation in siteSeed..."
  
  # Test 1: Determinism - same input produces same output
  block:
    let seed1 = siteSeed(12345'u64, [0, 0, 0, 0])
    let seed2 = siteSeed(12345'u64, [0, 0, 0, 0])
    assert seed1 == seed2, "Determinism test failed: same inputs should produce same output"
    echo "✓ Test 1 passed: Determinism verified"
  
  # Test 2: Uniqueness - different global seeds produce different outputs
  block:
    let seed1 = siteSeed(12345'u64, [0, 0, 0, 0])
    let seed2 = siteSeed(54321'u64, [0, 0, 0, 0])
    assert seed1 != seed2, "Uniqueness test failed: different global seeds should produce different outputs"
    echo "✓ Test 2 passed: Different global seeds produce different outputs"
  
  # Test 3: Uniqueness - different coordinates produce different outputs
  block:
    let seed1 = siteSeed(12345'u64, [0, 0, 0, 0])
    let seed2 = siteSeed(12345'u64, [1, 0, 0, 0])
    let seed3 = siteSeed(12345'u64, [0, 1, 0, 0])
    let seed4 = siteSeed(12345'u64, [0, 0, 1, 0])
    let seed5 = siteSeed(12345'u64, [0, 0, 0, 1])
    assert seed1 != seed2 and seed1 != seed3 and seed1 != seed4 and seed1 != seed5
    assert seed2 != seed3 and seed2 != seed4 and seed2 != seed5
    assert seed3 != seed4 and seed3 != seed5
    assert seed4 != seed5
    echo "✓ Test 3 passed: Different coordinates produce different outputs"
  
  # Test 4: Negative coordinates handled correctly
  block:
    let seed1 = siteSeed(12345'u64, [-1, -1, -1, -1])
    let seed2 = siteSeed(12345'u64, [1, 1, 1, 1])
    assert seed1 != seed2, "Negative coordinates test failed"
    # Should be deterministic with negatives
    let seed3 = siteSeed(12345'u64, [-1, -1, -1, -1])
    assert seed1 == seed3, "Negative coordinates should be deterministic"
    echo "✓ Test 4 passed: Negative coordinates handled correctly"
  
  # Test 5: Order matters - coordinate order affects result
  block:
    let seed1 = siteSeed(12345'u64, [1, 2, 3, 4])
    let seed2 = siteSeed(12345'u64, [4, 3, 2, 1])
    assert seed1 != seed2, "Coordinate order should matter"
    echo "✓ Test 5 passed: Coordinate order affects output"
  
  # Test 6: Large lattice uniqueness test
  block:
    var seeds: HashSet[uint64]
    const latticeSize = 16
    var collisions = 0
    
    for x in 0..<latticeSize:
      for y in 0..<latticeSize:
        for z in 0..<latticeSize:
          for t in 0..<latticeSize:
            let seed = siteSeed(12345'u64, [x, y, z, t])
            if seed in seeds:
              collisions += 1
            seeds.incl(seed)
    
    let totalSites = latticeSize * latticeSize * latticeSize * latticeSize
    assert collisions == 0, "Found " & $collisions & " collisions in " & $totalSites & " sites"
    echo "✓ Test 6 passed: No collisions in ", totalSites, " sites (16^4 lattice)"
  
  # Test 7: Distribution test - seeds should be well distributed
  block:
    var lowBits: HashSet[uint64]
    var highBits: HashSet[uint64]
    
    for i in 0..<1000:
      let seed = siteSeed(12345'u64, [i, 0, 0, 0])
      lowBits.incl(seed and 0xFFFF)
      highBits.incl((seed shr 48) and 0xFFFF)
    
    # Should have good distribution in both low and high bits
    assert lowBits.len > 900, "Poor distribution in low bits: " & $lowBits.len & "/1000"
    assert highBits.len > 900, "Poor distribution in high bits: " & $highBits.len & "/1000"
    echo "✓ Test 7 passed: Good bit distribution (low: ", lowBits.len, "/1000, high: ", highBits.len, "/1000)"
  
  # Test 8: FNV-1a constants verification
  block:
    const FNV_PRIME = 1099511628211'u64
    const FNV_OFFSET = 14695981039346656037'u64
    
    # These are the official 64-bit FNV-1a constants
    assert FNV_PRIME == 0x100000001b3'u64, "FNV_PRIME incorrect"
    assert FNV_OFFSET == 0xcbf29ce484222325'u64, "FNV_OFFSET incorrect"
    echo "✓ Test 8 passed: FNV-1a constants verified"
  
  echo "\nAll tests passed! ✓"
