#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/field/simplecubicfield.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  Notes:
  * What kind of gain (if any) could be achieved by considering inter-node
  connectivity in partitioning of lattice?

  MIT License
  
  Copyright (c) 2025 Curtis Taylor Peterson
  
  Permission is hereby granted, free of chadge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, medge, publish, distribute, sublicense, and/or sell
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

import backend
import utils
import lattice/[simplecubiclattice]

#[ frontend: simple cubic field type definition ]#

type
  SimpleCubicField*[T] = object
    ## Field on simple cubic Bravais lattice
    ##
    ## <in need of documentation>
    lattice: ref SimpleCubicLattice
    field: GlobalPointer[T]

#[ frontend: SimpleCubicField constructors ]#

proc newField*(lattice: SimpleCubicLattice; T: typedesc): SimpleCubicField[T] =
  ## Create new field on simple cubic Bravais lattice
  ##
  ## <in need of documentation>
  let numLocalSites = csize_t(lattice.numLocalSites)
  result = SimpleCubicField[T](field: newGlobalPointerArray(numLocalSites, T))
  new(result.lattice)
  result.lattice[] = lattice  

#[ frontend: basic SimpleCubicField methods ]#

proc `[]`*[T](f: SimpleCubicField[T]; n: int): T =
  ## Access field value at given local site
  ##
  ## This is for testing; it is most certainly not optimal or efficient
  ## <in need of documentation>
  let lPtr = f.field.local()
  let nIdx = cint(n)
  var value: T
  {.emit: """value_1 = lPtr_1[nIdx_1];""".}
  return value

when isMainModule:
  import runtime
  import lattice/[latticeconcept]
  const verbosity = 1
  reliq:
    let l = newSimpleCubicLattice([8, 8, 8, 16])
    var field = l.newField(float)

    if compiles(Lattice(l, seq[int], float)): 
      print "SimpleCubicLattice conforms to Lattice concept"
    else: print "SimpleCubicLattice does not conform to Lattice concept"

    for n in l.sites(): 
      if verbosity > 1: 
        echo "[" & $myRank() & "] " & "site: ", n, " coord: ", field[n]