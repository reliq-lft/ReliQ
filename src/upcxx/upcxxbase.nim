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

# shorten pragmas referencing upcxx and ReliQ wrapper headers
{.pragma: upcxx, header: "<upcxx/upcxx.hpp>".}

# template returning UPC++ include through pragma
template upcxx*(pragmas: untyped): untyped = 
  {.pragma: upcxx, header: "<upcxx/upcxx.hpp>".}
  pragmas

# initialize/finalize UPC++ runtime
proc upcxxInit* {.importcpp: "upcxx::init()", inline, upcxx.}
proc upcxxFinalize* {.importcpp: "upcxx::finalize()", inline, upcxx.}

# get process rank and total number of ranks
proc myRank*: cint {.importcpp: "upcxx::rank_me()", upcxx.}
proc numRanks*: cint {.importcpp: "upcxx::rank_n()", upcxx.}

# set barrier
proc upcxxBarrier* {.importcpp: "upcxx::barrier()", upcxx.}

when isMainModule:
  upcxxInit()

  echo "Hello world from process " & $myRank() &
    " out of " & $numRanks() & " processes"
  
  upcxxFinalize()