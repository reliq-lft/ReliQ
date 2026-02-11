#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/gauge/gaugefield.nim
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

## The code in this module should be considered the idiomatic syntax for ReliQ;
## i.e., it should be considered the "right way" to use tensor types in ReliQ

import lattice
import parallel
import tensor/[tensor]
import utils/[complex]
import globalarrays/[gabase, gawrap]

type
  SinglePrecision* = object
  DoublePrecision* = object

  GroupKind* = enum
    gkSU2,
    gkSU3,
    gkSU4,
    gkSU5

  RepresentationKind* = enum
    rkFundamental,
    rkAdjoint

  PrecisionKind* = enum
    pkSingle,
    pkDouble

  GaugeFieldContext*[T] = object
    group*: GroupKind
    representation*: RepresentationKind
    precision*: PrecisionKind

type
  GaugeField*[D: static[int], L: Lattice[D], T] = object
    ctx*: GaugeFieldContext[T]
    field*: array[D, TensorField[D, 2, L, T]]
  
  LocalGaugeField*[D: static[int], L: Lattice[D], T] = object
    ctx*: GaugeFieldContext[T]
    field*: array[D, LocalTensorField[D, 2, L, T]]
  
  GaugeFieldView*[D: static[int], L: Lattice[D], T] = object
    ctx*: GaugeFieldContext[T]
    field*: array[D, TensorFieldView[L, T]]

type
  GaugeActionContext* = object
    beta*: float64
    cp*: float64
    cr*: float64
    cpg*: float64

#[ context constructor and accessors ]#

template newGaugeFieldContext*[T](
  grp: GroupKind; 
  rep: RepresentationKind;
  prc: T = DoublePrecision()
): untyped =
  case grp
  of gkSU2, gkSU3, gkSU4, gkSU5:
    when T is SinglePrecision:
      GaugeFieldContext[Complex32](group: grp, representation: rep, precision: pkSingle)
    else:
      GaugeFieldContext[Complex64](group: grp, representation: rep, precision: pkDouble)

func nc*(group: GroupKind, representation: RepresentationKind): int =
  ## Get number of colors from the gauge group and representation
  ## Note that adjoint represented as matrices, so this does not correspond to dimension of the representation
  case group
  of gkSU2:
    case representation
    of rkFundamental: 2
    of rkAdjoint: 2
  of gkSU3:
    case representation
    of rkFundamental: 3
    of rkAdjoint: 3
  of gkSU4:
    case representation
    of rkFundamental: 4
    of rkAdjoint: 4
  of gkSU5: 
    case representation
    of rkFundamental: 5
    of rkAdjoint: 5

func nc*[T](ctx: GaugeFieldContext[T]): int =
  ## Get number of colors from the gauge field context
  nc(ctx.group, ctx.representation)

#[ gauge field view constructor ]#

proc newGaugeField*[D: static[int], L: Lattice[D], T](
  lattice: L; 
  ctx: GaugeFieldContext[T]
): GaugeField[D, L, T] = 
  result.ctx = ctx
  let n = ctx.nc
  for mu in 0..<D: result.field[mu] = lattice.newTensorField([n, n]): T

proc newLocalGaugeField*[D: static[int], L: Lattice[D], T](
  u: GaugeField[D, L, T]
): LocalGaugeField[D, L, T] = 
  result.ctx = u.ctx
  for mu in 0..<D: 
    result.field[mu] = u.field[mu].newLocalTensorField()

template newGaugeFieldView*[D: static[int], L: Lattice[D], T](
  u: GaugeField[D, L, T];
  iok: IOKind
): GaugeFieldView[D, L, T] =
  block:
    var gfv: GaugeFieldView[D, L, T]
    gfv.ctx = u.ctx
    for mu in 0..<D:
      gfv.field[mu] = u.field[mu].newTensorFieldView(iok)
    gfv

#[ accessors ]#

proc lattice*[D: static[int], L: Lattice[D], T](u: GaugeField[D, L, T]): L =
  u.field[0].lattice

proc `[]`*[D: static[int], L: Lattice[D], T](gaugeField: var GaugeField[D, L, T], mu: int): var TensorField[D, 2, L, T] =
  ## Get a mutable reference to the gauge field tensor for a given direction
  gaugeField.field[mu]

proc `[]`*[D: static[int], L: Lattice[D], T](gaugeField: GaugeField[D, L, T], mu: int): lent TensorField[D, 2, L, T] =
  ## Get a read-only reference to the gauge field tensor for a given direction
  gaugeField.field[mu]

proc `[]=`*[D: static[int], L: Lattice[D], T](gaugeField: var GaugeField[D, L, T], mu: int, value: TensorField[D, 2, L, T]) =
  ## Set the gauge field tensor for a given direction
  gaugeField.field[mu] = value

proc `[]`*[D: static[int], L: Lattice[D], T](gaugeFieldView: GaugeFieldView[D, L, T], mu: int): lent TensorFieldView[L, T] =
  ## Get a read-only view of the gauge field tensor for a given direction
  gaugeFieldView.field[mu]

proc `[]=`*[D: static[int], L: Lattice[D], T](gaugeFieldView: var GaugeFieldView[D, L, T], mu: int, value: TensorFieldView[L, T]) =
  ## Set the gauge field tensor view for a given direction
  gaugeFieldView.field[mu] = value

proc `[]`*[D: static[int], L: Lattice[D], T](gaugeFieldView: var GaugeFieldView[D, L, T], mu: int): var TensorFieldView[L, T] =
  ## Get a mutable view of the gauge field tensor for a given direction
  gaugeFieldView.field[mu]

template TensorSiteProxy*[D: static[int], L: Lattice[D], T](gfv: GaugeFieldView[D, L, T]): typedesc =
  ## Return the TensorSiteProxy type matching this gauge field view
  sitetensor.TensorSiteProxy[L, T]

proc nc*(gaugeField: GaugeField): int =
  ## Get number of colors from the gauge field
  nc(gaugeField.ctx)

proc nd*[D: static[int], L: Lattice[D], T](gaugeField: GaugeField[D, L, T]): int =
  ## Get number of dimensions from the gauge field
  D

#[ methods ]#

proc setToIdentity*[D: static[int], L: Lattice[D], T](u: var GaugeField[D, L, T]) =
  for mu in 0..<D:
    var umu = u[mu].newLocalTensorField()
    let ns = umu.numSites()
    let nc = umu.shape[0]
    for n in all 0..<ns:
      for i in 0..<nc:
        when T is Complex32: umu[n][i, i] = complex32(1.0, 0.0)
        elif T is Complex64: umu[n][i, i] = complex64(1.0, 0.0)
        else: umu[n][i, i] = T(1)

proc linkTrace*[D: static[int], L: Lattice[D], T](u: GaugeField[D, L, T]): float64 =
  ## Compute the average link trace of a gauge configuration.
  let lattice = u.lattice
  let nc = u.nc
  var traceSum = 0.0

  accelerator:
    var vu = u.newGaugeFieldView(iokRead)
    for mu in 0..<D:
      for n in reduce vu[mu].all:
        traceSum += trace(vu[mu][n]).re
  
  traceSum /= float64(D * nc * lattice.globalVolume)
  return traceSum

proc plaquette*[D: static[int], L: Lattice[D], T](u: GaugeField[D, L, T]): float64 =
  ## Compute the average plaquette of a gauge configuration.
  
  let nc = u.nc
  var traceSum = 0.0
  
  let lattice = u.lattice
  let stencil = lattice.newNearestNeighborLatticeStencil()

  var plaq = lattice.newTensorField([nc, nc]): T

  accelerator:
    var vu = u.newGaugeFieldView(iokRead)
    var vplaq = plaq.newTensorFieldView(iokWrite)

    for mu in 1..<D:
      for nu in 0..<mu:
        for n in each vplaq.all:
          let fwdMu = stencil.fwd(n, mu)
          let fwdNu = stencil.fwd(n, nu)
          let ta = vu[mu][n] * vu[nu][fwdMu]
          let tb = vu[nu][n] * vu[mu][fwdNu]
          vplaq[n] += ta * tb.adjoint()
    
    for n in reduce vplaq.all:
      traceSum += trace(vplaq[n]).re
  
  traceSum /= float(6*nc*lattice.globalVolume)
  return traceSum

when isMainModule:
  parallel:
    let dim: array[4, int] = [8, 8, 8, 16]
    let lat = newSimpleCubicLattice(dim)
    let ctx = newGaugeFieldContext(gkSU3, rkFundamental)
    var ua = lat.newGaugeField(ctx)
  
    ua.setToIdentity()

    let traceVal = ua.linkTrace()
    echo "Link trace: ", traceVal
    if abs(traceVal - 1.0) < 1e-10:
      echo "Link trace test: [OK]"
    else:
      echo "Link trace test: [FAILED] expected 1.0 got ", traceVal
    
    let plaqVal = ua.plaquette()
    echo "Plaquette: ", plaqVal
    if abs(plaqVal - 1.0) < 1e-10:
      echo "Plaquette test: [OK]"
    else:
      echo "Plaquette test: [FAILED] expected 1.0 got ", plaqVal
  