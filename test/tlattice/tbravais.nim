#[ 
  QXX lattice field theory framework: github.com/ctpeterson/QXX
  Source file: test/tlattice/tbravais.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 Curtis Taylor Peterson
  
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

import std/[system]

import tlatticeconcept
import tupcxx

type
  SimpleCubicLattice = object
    latticeGeometry: seq[GeometryType]
    rankGeometry: seq[GeometryType]
    globalPtr: upcxx_global_ptr[CoordinateType]
    sites: seq[CoordinateType]

template toGeomSeq[T](ilg, irg: openArray[T]): untyped =
  assert(tlg.len == trg.len)
  let nd = tlg.len
  var (olg, org) = (newSeq[GeometryType](nd), newSeq[GeometryType](nd))
  for idx in 0..<nd: 
    (olg[idx], org[idx]) = (GeometryType(ilg[idx]), GeometryType(irg[idx]))
  (olg, org)

proc newSimpleCubicLattice*(
  latticeGeometry, rankGeometry: openArray[SomeInteger]
): SimpleCubicLattice =
  let (lg, rg) = toGeomSeq(latticeGeometry, rankGeometry)
