#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/kokkos/kokkosdefs.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  Notes:
  * Kokkos GitHub: https://github.com/kokkos/kokkos
  * Kokkos wiki: https://kokkos.org/kokkos-core-wiki/
  * UPC++ + Kokkos: https://tinyurl.com/4cvza7v2
  * Lecutres on Kokkos: https://tinyurl.com/dhbrr7yn

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

import utils/[nimutils]

#[ static compile-time information gathering and processing ]#

# Kokkos and information gathering
const # vvvv will be figured out by configuration scipt vvvv
  SpackP = "/home/curtyp/spack/opt/spack"
  KokkosP = SpackP & "/linux-zen4/kokkos-4.6.01-gun7pi2qgo4bhxhiaic5sk33rf2wrqmo"
const 
  KokkosC = "-I" & KokkosP & "/include"
  KokkosL = "-L" & KokkosP & "/lib"

# informs user of Kokkos build location
static: echo "Kokkos: " & KokkosP

# pass compiler flags
{.passC: KokkosC.}
{.passL: KokkosL.}

# shorten pragmas pointing to Kokkos headers
{.pragma: kokkos_core, header: "<Kokkos_Core.hpp>".}

#[ initialize/finalize Kokkos runtime ]#

# initializes Kokkos runtime
proc kokkos_init(argc: cint; argv: cstringArray)
  {.importcpp: "Kokkos::initialize(#, #)", kokkos_core.}
proc kokkos_init* =
  let 
    argc = cargc()
    argv = cargv(argc)
  kokkos_init(argc, argv)
  deallocCStringArray(argv)

# finalizes Kokkos runtime
proc kokkos_finalize* {.importcpp: "Kokkos::finalize()", kokkos_core.}

#[ tests ]#

when isMainModule: 
  # nim cpp --path:/home/curtyp/Software/ReliQ/src kokkosdefs
  # local test: 

  kokkos_init()

  kokkos_finalize()