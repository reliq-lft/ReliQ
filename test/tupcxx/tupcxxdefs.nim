#[ 
  QXX lattice field theory framework: github.com/ctpeterson/QXX
  Source file: test/tupcxx/tupcxxdefs.nim
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

const 
  UPCXXP = "/home/curtyp/Software/upcxx-2023.9.0/build"
  UPCXXH = "upcxx/upcxx.hpp"
  UPCXX = UPCXXP & "/include/" & UPCXXH
  UPCXXM = UPCXXP & "/bin/upcxx-meta "
const 
  UPCXXPP = staticExec(UPCXXM & "PPFLAGS")
  UPCXXC =  staticExec(UPCXXM & "CFLAGS")
  UPCXXL =  staticExec(UPCXXM & "LIBFLAGS")

static: echo "UPCXX: " & UPCXXP

{.passC: UPCXXPP & " " & UPCXXC.}
{.passL: UPCXXL.}
{.pragma: upcxx, header: UPCXX.}

type
  upcxx_global_ptr*[T] {.importcpp: "upcxx::global_ptr", upcxx.} = object

proc upcxx_init*() {.importcpp: "upcxx::init()", upcxx.}
proc upcxx_finalize*() {.importcpp: "upcxx::finalize()", upcxx.}

proc upcxx_rank_me*(): cint {.importcpp: "upcxx::rank_me()", upcxx.}
proc upcxx_rank_n*(): cint {.importcpp: "upcxx::rank_n()", upcxx.}

proc upcxx_barrier*() {.importcpp: "upcxx::barrier()", upcxx.}

proc upcxx_new*(T: typedesc): upcxx_global_ptr[T] 
  {.constructor, importcpp: "upcxx::new_<#>(upcxx::rank_me())", upcxx.}
proc upcxx_delete*[T](global_ptr: upcxx_global_ptr[T])
  {.importcpp: "upcxx::delete_(#)", upcxx.}

proc upcxx_local*[T](global_ptr: upcxx_global_ptr[T]): ptr T
  {.importcpp: "#.local()", upcxx.}

when isMainModule: # upcxx-run -n 4 -localhost tupcxxdefs
  upcxx_init()

  echo "Hello world from process " & $upcxx_rank_me() &
    " out of " & $upcxx_rank_n() & " processes"
  
  var gptr = upcxx_new(int)

  upcxx_barrier()

  var lptr = gptr.upcxx_local()

  upcxx_delete(gptr)
  
  upcxx_finalize()