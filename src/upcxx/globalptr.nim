#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/upcxx/globalptr.nim
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

# include upcxx headers
upcxx: discard

type # UPC++ global pointer: accessible to all ranks; downcasts to ordinary pointer
  GlobalPointer*[T] {.importcpp: "upcxx::global_ptr", upcxx.} = object

# upcxx::global_ptr constructor: single data type
proc newGlobalPointer*[T](t: T): GlobalPointer[T] 
  {.constructor, importcpp: "upcxx::new_<'*0>(#)", upcxx.}

# upcxx::global_ptr destructor: single data type
proc deleteGlobalPointer*[T](global_ptr: GlobalPointer[T])
  {.importcpp: "upcxx::delete_(#)", upcxx.}

# return upcxx::global_ptr to new array
proc newGlobalPointerArray[T](size: csize_t): GlobalPointer[T]
  {.importcpp: "upcxx::new_array<'*0>(#)", upcxx.}
proc newGlobalPointerArray*(size: csize_t, T: typedesc): GlobalPointer[T] = 
  return newGlobalPointerArray[T](size)  

# upcxx::global_ptr destructor: array of data type
proc deleteGlobalPointerArray*[T](global_ptr: GlobalPointer[T])
  {.importcpp: "upcxx::delete_array(#)", upcxx.}

# downcasts upcxx::global_ptr to ordinary pointer
proc local*[T](global_ptr: GlobalPointer[T]): ptr T {.importcpp: "#.local()", upcxx.}

when isMainModule:
  upcxxInit()
  
  let sq = newSeq[int](10)
  var gptr = newGlobalPointer(sq)

  var lptr = gptr.local()

  gptr.deleteGlobalPointer()

  let sz: csize_t = 10
  var gptrArr = newGlobalPointerArray(sz, int)

  gptrArr.deleteGlobalPointerArray()
  
  upcxxFinalize()