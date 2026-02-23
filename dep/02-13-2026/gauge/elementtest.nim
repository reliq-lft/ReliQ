#[
  Element-level operation test
  
  Tests that user-defined element-level operations on proxy types
  (for-loops, element read/write, conj) produce correct results
  by comparing against the hardcoded transpiler primitives.
]#

import gaugefield

import lattice
import parallel
import tensor/[tensor]
import types/[complex]
import globalarrays/[gabase, gawrap]

proc elementTest[D: static[int], L: Lattice[D], T](
  u: GaugeField[D, L, T]
) =
    let nd = u.nd
    let nc = u.nc
    let lattice = u.lattice
    let stencil = lattice.newStapleLatticeStencil()

    # Test 1: element-level adjoint via for-loop + conj
    # Compare hardcoded adjoint vs element-level adjoint
    var actHardcoded = lattice.newScalarField: T
    var actElement = lattice.newScalarField: T

    var sumHard = 0.0
    var sumElem = 0.0

    accelerator:
      var vHard = actHardcoded.newScalarFieldView(iokWrite)
      var vElem = actElement.newScalarFieldView(iokWrite)
      var vu = u.newGaugeFieldView(iokRead)

      for mu in 1..<nd:
        for nu in 0..<mu:
          for n in each vHard.all:
            let fwdMu = stencil.fwd(n, mu)
            let fwdNu = stencil.fwd(n, nu)

            let ta = vu[mu][n] * vu[nu][fwdMu]
            let tb = vu[nu][n] * vu[mu][fwdNu]

            # Hardcoded transpiler adjoint
            let tbAdj = tb.adjoint()
            vHard[n] += trace(ta * tbAdj)

          for n in each vElem.all:
            let fwdMu = stencil.fwd(n, mu)
            let fwdNu = stencil.fwd(n, nu)

            let ta = vu[mu][n] * vu[nu][fwdMu]
            let tb = vu[nu][n] * vu[mu][fwdNu]

            # Element-level adjoint using for-loop + conj
            var tbAdj = siteIdentity(vu.TensorSiteProxy)
            for i in 0..<nc:
              for j in 0..<nc:
                tbAdj[i,j] = conj(tb[j,i])

            vElem[n] += trace(ta * tbAdj)

      # Reduce both and compare
      var sumHard = 0.0
      var sumElem = 0.0

      for n in reduce vHard.all:
        sumHard += vHard[n].re

      for n in reduce vElem.all:
        sumElem += vElem[n].re

    echo "Hardcoded adjoint sum: ", sumHard
    echo "Element-level adjoint sum: ", sumElem

    let diff = abs(sumHard - sumElem)
    if diff < 1e-10:
      echo "Element-level adjoint: [OK]"
    else:
      echo "Element-level adjoint: [FAIL] diff = ", diff

when isMainModule:
  parallel:
    let dim: array[4, int] = [8, 8, 8, 16]
    let lat = newSimpleCubicLattice(dim)
    let ctx = newGaugeFieldContext(gkSU3, rkFundamental)
    var ua = lat.newGaugeField(ctx)
  
    ua.setToIdentity()

    elementTest(ua)
