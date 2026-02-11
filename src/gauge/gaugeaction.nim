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

import lattice
import parallel
import tensor/[tensor]
import utils/[complex]
import globalarrays/[gabase, gawrap]
import profile/[profile]

proc action*[D: static[int], L: Lattice[D], T](
  c: GaugeActionContext; 
  u: GaugeField[D, L, T]
): float =
  initProfiler("GaugeAction")
  
  tic("Setup")
  
  let nd = u.nd
  let nc = u.nc
  var traceSum = 0.0
  
  let lattice = u.lattice
  let stencil = lattice.newStapleLatticeStencil()

  var action = lattice.newScalarField: T

  let nrm = float64(nc*nd*(nd-1)*lattice.globalVolume)
  let cp = c.beta*c.cp/nrm
  let cr = c.beta*c.cr/nrm
  let cpg = c.beta*c.cpg/nrm

  var actionSum = 0.0

  toc()

  accelerator:
    tic("ViewCreation")

    var vact = action.newScalarFieldView(iokWrite)
    var vu = u.newGaugeFieldView(iokRead)

    toc()
    tic("ActionLoop")

    for mu in 1..<nd:
      for nu in 0..<mu:
        for n in each vact.all:
          let fwdMu = stencil.fwd(n, mu)
          let fwdNu = stencil.fwd(n, nu)

          let ta = vu[mu][n] * vu[nu][fwdMu]
          let tb = vu[nu][n] * vu[mu][fwdNu]

          var id = siteIdentity(vu.TensorSiteProxy)

          addFLOP 2*matMulFLOP(nc)
          
          # plaquette
          if c.cp != 0.0: 
            vact[n] += cp * trace(id - ta * tb.adjoint())

            addFLOP matMulFLOP(nc) + matAddSubFLOP(nc) + traceFLOP(nc) + 1

          # rectangle
          if c.cr != 0.0:
            let bwdMu = stencil.bwd(n, mu)
            let bwdNu = stencil.bwd(n, nu)
            let bwdMuFwdNu = stencil.corner(n, -1, mu, +1, nu)
            let bwdNuFwdMu = stencil.corner(n, -1, nu, +1, mu)
            
            let tc = vu[nu][bwdNu].adjoint() * vu[mu][bwdNu] * vu[nu][bwdNuFwdMu]
            let td = tb * vu[nu][fwdNu].adjoint()

            let te = ta * vu[nu][fwdNu].adjoint()
            let tf = vu[mu][bwdMu].adjoint() * vu[nu][bwdMu] * vu[mu][bwdMuFwdNu]
            
            vact[n] += cr * trace(2.0*id - te * tf.adjoint() - tc * td.adjoint())

            addFLOP 6*matMulFLOP(nc) + 2*matAddSubFLOP(nc) + 2*traceFLOP(nc) + 2
          
          # parallelogram
          if c.cpg != 0.0: discard
    
    toc()
    tic("Reduction")
    
    for n in reduce vact.all:
      actionSum += vact[n].re
      addFLOP(1)
    
    toc()
  
  result = nrm*actionSum
  
  finalizeProfiler()

when isMainModule:
  import io/[gaugeio]

  const SampleILDGFile = "src/io/sample/ildg.lat"

  parallel:
    let dim: array[4, int] = [8, 8, 8, 16]
    let lat = newSimpleCubicLattice(dim)
    let ctx = newGaugeFieldContext(gkSU3, rkFundamental)
    let act = GaugeActionContext(beta: 6.0, cp: 1.0, cr: 1.0, cpg: 0.0)
    var ua = lat.newGaugeField(ctx)
  
    ua.readGaugeField(SampleILDGFile)

    echo act.action(ua)