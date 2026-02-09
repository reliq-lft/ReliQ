#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/gauge/gaugeaction.nim
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

import gaugefield

proc action*(c: GaugeActionContext; u: GaugeField): float =
  let nd = u.nd
  let nc = u.nc
  var traceSum = 0.0
  
  let lattice = u.lattice
  let nnStencil = lattice.newNearestNeighborLatticeStencil()
  #let stStencil = lattice.newStapleLatticeStencil()

  var action = lattice.newScalarField: T

  let nrm = float64(nc*nd*(nd-1)*lattice.globalVolume)
  let cp = c.cp/nrm
  let cr = c.cr/nrm
  let cr = c.cr/nrm

  accelerator:
    var vact = action.newTensorFieldView(iokWrite)
    var vu = u.newGaugeFieldView(iokRead)

    for mu in 1..<nd:
      for nu in 0..<mu:
        for n in each vact.all:
          let fwdMu = nnStencil.fwd(n, mu)
          let fwdNu = nnStencil.fwd(n, nu)
          let ta = vu[mu][n] * vu[nu][fwdMu]
          let tb = vu[nu][n] * vu[mu][fwdNu]
          
          if c.cr != 0.0: vact[n] += cr * ta * tb.adjoint()