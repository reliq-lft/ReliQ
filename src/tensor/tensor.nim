#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/types/tensor.nim
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

import globaltensor
import localtensor
import tensorview
import lattice/stencil

export globaltensor
export localtensor
export tensorview
export stencil

when isMainModule:
  import std/unittest
  import lattice
  import parallel
  import utils/[complex]

  suite "Tensor Module Integration":
    test "Stencil is exported":
      # Verify stencil types are accessible
      let pattern = nearestNeighborStencil[4]()
      check pattern.nPoints == 8

    test "LatticeStencil from Lattice":
      let lat = newSimpleCubicLattice([8, 8, 8, 16])
      let stencil = newLatticeStencil(nearestNeighborStencil[4](), lat)
      check stencil.nSites > 0

    test "LatticeStencil with ghost regions":
      let pattern = nearestNeighborStencil[2]()
      let stencil = newLatticeStencil(pattern, [4, 4], [1, 1])
      check stencil.nLocalSites == 16
      check stencil.nPaddedSites == 36
      check stencil.paddedGeom == [6, 6]

    test "Stencil shift API":
      let pattern = nearestNeighborStencil[4]()
      let stencil = newLatticeStencil(pattern, [4, 4, 4, 4])
      # Test shift() produces valid neighbor indices
      for site in stencil.sites:
        let fwdX = stencil.fwd(site, 0)
        let bwdT = stencil.bwd(site, 3)
        check fwdX.idx >= 0 and fwdX.idx < stencil.nPaddedSites
        check bwdT.idx >= 0 and bwdT.idx < stencil.nPaddedSites

  # MPI-based tests (require parallel context)
  parallel:
    let dim: array[4, int] = [8, 8, 8, 16]
    let lat = newSimpleCubicLattice(dim)

    block: # GA objects must be destroyed before finalizeGA
      var cf1 = lat.newTensorField([3, 3]): Complex64
      var cf2 = lat.newTensorField([3, 3]): Complex64
      var cf3 = lat.newTensorField([3, 3]): Complex64

      block:
        var lcf1 = cf1.newLocalTensorField()
        var lcf2 = cf2.newLocalTensorField()
        var lcf3 = cf3.newLocalTensorField()

        for i in all 0..<lcf1.numElements():
          lcf1[i] = complex64(float64(i), 0.0)
          lcf2[i] = complex64(float64(i * 2), 0.0)

        # Create views for reading and writing.
        # Views with iokWrite have buffers allocated automatically.
        # The same is true for views with iokRead and iokReadWrite; however,
        # they also have their data synchronized from the parent tensor field.
        var vcf1 = lcf1.newTensorFieldView(iokRead)
        var vcf2 = lcf2.newTensorFieldView(iokRead)
        var vcf3 = lcf3.newTensorFieldView(iokWrite)

        for n in each 0..<vcf1.numSites():
          vcf3[n] = vcf1[n] + vcf2[n]
      
      # Views destroyed here when going out of scope. 
      # Views with iokRead have their buffers deallocated automatically.
      # The same is true for views with iokWrite and iokReadWrite; however,
      # they also have their data synchronized back to the parent tensor field.

    # Stencil tests with MPI - Unified API
    block:
      # Create a scalar field and stencil
      var field = lat.newTensorField([1]): float64
      let stencil = newLatticeStencil(nearestNeighborStencil[4](), lat)
      
      # Verify stencil dimensions
      assert stencil.nSites > 0, "Stencil should have sites"
      assert stencil.nPoints == 8, "Nearest neighbor stencil should have 8 points"
      
      # Test shift API
      for site in 0..<min(10, stencil.nSites):
        let fwd = stencil.fwd(site, 0)
        let bwd = stencil.bwd(site, 0)
        assert fwd.idx >= 0, "Forward shift should be valid"
        assert bwd.idx >= 0, "Backward shift should be valid"

    # Test stencil construction and basic neighbor lookup
    block:
      # Create stencil for this lattice (no ghost width for local periodic BC)
      let stencil = newLatticeStencil(nearestNeighborStencil[4](), lat)
      
      # Verify stencil dimensions
      assert stencil.nSites > 0, "Stencil should have sites"
      assert stencil.nPoints == 8, "Nearest neighbor stencil should have 8 points in 4D"
      
      # Test that all neighbor indices are valid
      for site in 0..<min(100, stencil.nSites):
        for p in 0..<stencil.nPoints:
          let nbr = stencil.neighbor(site, p)
          assert nbr >= 0 and nbr < stencil.nPaddedSites, "Neighbor should be valid"

    # Test stencil shift() API
    block:
      let stencil = newLatticeStencil(nearestNeighborStencil[4](), lat)
      
      for site in 0..<min(50, stencil.nSites):
        for dir in 0..<4:
          let fwd = stencil.fwd(site, dir)
          let bwd = stencil.bwd(site, dir)
          assert fwd.idx >= 0, "Forward shift should be valid"
          assert bwd.idx >= 0, "Backward shift should be valid"

    # End of GA block - tensor fields destroyed before finalizeGA
