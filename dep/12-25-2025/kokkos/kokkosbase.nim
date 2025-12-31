#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/kokkos/kokkosbase.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  Notes:
  * Kokkos GitHub: https://github.com/kokkos/kokkos
  * Kokkos wiki: https://kokkos.org/kokkos-core-wiki/
  * UPC++ + Kokkos: https://tinyurl.com/4cvza7v2
  * Lectures on Kokkos: https://tinyurl.com/dhbrr7yn

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

import utils/[commandline]

template Kokkos*(pragmas: untyped): untyped = 
  {.pragma: kokkos, header: "<Kokkos_Core.hpp>".}
  pragmas
  
Kokkos: discard

#[ frontend: runtime initializers/finalizers ]#

proc initKokkos*(argc: cint; argv: cstringArray)
  {.importcpp: "Kokkos::initialize(#, #)", inline, kokkos.}

proc initKokkos* {.inline.} =
  let 
    argc = cargc()
    argv = cargv(argc)
  initKokkos(argc, argv)
  deallocCStringArray(argv)

proc finalizeKokkos* {.importcpp: "Kokkos::finalize()", inline, kokkos.}

#[ misc ]#

proc numThreads*: cint {.importcpp: "Kokkos::num_threads()", inline, kokkos.}

proc localBarrier* {.importcpp: "Kokkos::fence()", inline, kokkos.}

#[ unit tests ]#

when isMainModule:
  block:
    initKokkos()
    echo "Kokkos initialized"
    
    echo "Number of Kokkos threads: ", numThreads()
    
    finalizeKokkos()
    echo "Kokkos finalized"