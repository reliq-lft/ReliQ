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

export globaltensor
export localtensor
export tensorview

when isMainModule:
  import lattice
  import parallel
  import utils/[complex]

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
    # End of GA block - tensor fields destroyed before finalizeGA

