#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/upcxx/upcxxdefs.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  Notes:
  * This source is intended to be encapsulated entirely within ReliQ's backend

  Resources:
  * UPC++ Programmer's Guide: https://upcxx.lbl.gov/docs/html/guide.html
  * Graph algorithms in UPC++: https://tinyurl.com/nv8mr7h9
    - Has useful information on interleaved communication/computation

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

#[ static compile-time information gathering and processing ]#

const # path to upcxx header and configuration/metadata script
  UPCXXP = "/home/curtyp/Software/upcxx-2023.9.0/build"
  UPCXXM = UPCXXP & "/bin/upcxx-meta "
const # execute configuration/metadata script to get compilation flags
  UPCXXPP = staticExec(UPCXXM & "PPFLAGS")
  UPCXXC =  staticExec(UPCXXM & "CFLAGS")
  UPCXXL =  staticExec(UPCXXM & "LIBFLAGS")

# informs user of upcxx build location
static: echo "UPCXX: " & UPCXXP

# pass compiler flags from metadata
{.passC: UPCXXPP & " " & UPCXXC.}
{.passL: UPCXXL.}

# shorten pragmas referencing upcxx and ReliQ wrapper headers
{.pragma: upcxx, header: "<upcxx/upcxx.hpp>".}

#[ Nim wrappers of UPC++ types ]#

type # UPC++ global pointer: accessible to all ranks; downcasts to ordinary pointer
  upcxx_global_ptr*[T] {.importcpp: "upcxx::global_ptr", upcxx.} = object

#[ initialize/finalize UPC++ runtime ]#

proc upcxx_init* {.importcpp: "upcxx::init()", upcxx.}
proc upcxx_finalize* {.importcpp: "upcxx::finalize()", upcxx.}

#[ get process rank and total number of ranks ]#

proc upcxx_rank_me*: cint {.importcpp: "upcxx::rank_me()", upcxx.}
proc upcxx_rank_n*: cint {.importcpp: "upcxx::rank_n()", upcxx.}

#[ upcxx::global_ptr constructors, destructors, and methods ]#

# upcxx::global_ptr constructor: single data type
proc upcxx_new*[T]: upcxx_global_ptr[T] 
  {.constructor, importcpp: "upcxx::new_<'*0>(upcxx::rank_me())", upcxx.}

# upcxx::global_ptr destructor: single data type
proc upcxx_delete*[T](global_ptr: upcxx_global_ptr[T])
  {.importcpp: "upcxx::delete_(#)", upcxx.}

# return upcxx::global_ptr to new array
proc upcxx_new_array*[T](size: csize_t): upcxx_global_ptr[T]
  {.importcpp: "upcxx::new_array<'*0>(#)", upcxx.}

# upcxx::global_ptr destructor: array of data type
proc upcxx_delete_array*[T](global_ptr: upcxx_global_ptr[T])
  {.importcpp: "upcxx::delete_array(#)", upcxx.}

# downcasts upcxx::global_ptr to ordinary pointer
proc upcxx_local*[T](global_ptr: upcxx_global_ptr[T]): ptr T
  {.importcpp: "#.local()", upcxx.}

#[ misc. UPC++ procedures ]#

#proc upcxx_barrier* {.importcpp: "upcxx::barrier()", upcxx.}

#[ tests ]#

when isMainModule: 
  # nim cpp upcxxdefs.nim
  # upcxx-run -n 4 -localhost upcxxdefs

  upcxx_init()

  #[ hello world: https://tinyurl.com/2x37bay2 ]#

  echo "Hello world from process " & $upcxx_rank_me() &
    " out of " & $upcxx_rank_n() & " processes"
  
  var gptr = upcxx_new[int]()

  #upcxx_barrier()

  var lptr = gptr.upcxx_local()

  upcxx_delete(gptr)

  #[ test construction of upcxx::global_ptr to array ]#

  let sz: csize_t = 10
  var gptrArr = upcxx_new_array[int](sz)

  upcxx_delete_array(gptrArr)
  
  upcxx_finalize()