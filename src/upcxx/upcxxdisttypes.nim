#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/upcxx/upcxxdisttypes.nim
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

import upcxxbase
import upcxx/[upcxxglobalptr]

# include upcxx header
upcxx: discard

#[ frontend: distributed types ]#

const DOBJ = "upcxx::dist_object"
const GPTR = "upcxx::global_ptr"

type
  DistributedObject*[T] {.importcpp: DOBJ & "<" & GPTR & "<'*0>>", upcxx.} = object
    ## UPC++ distributed object
    ## 
    ## <in need of documentation>

type
  DistributedArray*[T] = object
    ## Distributed array
    ## 
    ## <in need of documentation>
    gPtr: GlobalPointer[T]
    dObj: ptr DistributedObject[T]

#[ frontend: constructors ]#

proc newDistributedObject*[T](gPtr: GlobalPointer[T]): DistributedObject[T]
  {.constructor: DOBJ & "<" & GPTR & "<'*0>>(#)", upcxx.}

proc newDistributedArray*(size: SomeInteger; T: typedesc): DistributedArray[T] =
  ## Create new distributed array of given size
  ##
  ## Returns: DistributedArray
  let gPtr = newGlobalPointerArray(size div numRanks()): T
  let dObj = gPtr.newDistributedObject()
  return DistributedArray[T](gPtr: gPtr, dObj: addr dObj)

#[ frontend: methods ]#

proc global*[T](distArr: var DistributedArray[T]): GlobalPointer[T] =
  ## Return global pointer to distributed array
  return distArr.gPtr

## Return local pointer to distributed object
proc local*[T](distObj: var DistributedObject[T]): ptr T
  {.importcpp: "#->local()", upcxx.}

proc local*[T](distArr: var DistributedArray[T]): ptr T =
  ## Return local pointer to distributed array
  return distArr.gPtr.local()

#[ tests ]#

when isMainModule:
  import runtime
  reliq: 
    block:
      let gPtr = newGlobalPointerArray(64): float
      var distObj = gPtr.newDistributedObject()
      let localObj = distObj.local()
      var distArr = newDistributedArray(100): float
      let globalArr = distArr.global()
      let localArr = distArr.local()

    # simple halo exchange test
    # file:///home/curtyp/Downloads/A_Newcomer_In_The_PGAS_World_--_UPC_vs_UPC_A_Compa.pdf
    var
      coord1first = newSeq[GlobalPointer[float]](numRanks())
      coord1last = newSeq[GlobalPointer[float]](numRanks())