#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/backend.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
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

proc reliqInit* {.inline.} =
  ## Initializes ReliQ's runtime environment
  ## Author: Curtis Taylor Peterson
  ## 
  ## Initialize ReliQ runtime environment. This is currently
  ## composed of two keys steps:
  ## 
  ## a.) Initialize UPC++ runtime environemnt
  ## b.) Initialize Kokkos runtime environment
  upcxxInit()
  kokkosInit()

proc reliqFinalize* {.inline.} =
  ## Initializes ReliQ's runtime environment
  ## Author: Curtis Taylor Peterson
  ## 
  ## Finalizes ReliQ runtime environment. This is currently
  ## composed of two keys steps:
  ## 
  ## a.) Finalize Kokkos runtime environment
  ## b.) Finalize UPC++ runtime environemnt
  kokkosFinalize()
  upcxxFinalize()

template reliq*(work: untyped): untyped =
  ## Encapsulates ReliQ's runtime environment in workload block
  ## Author: Curtis Taylor Peterson
  reliqInit()
  work
  reliqFinalize()

when isMainModule:
  import utils

  const encapsulated = true

  proc hello_world() =
    echo "Hello world from process" + $myRank() +
      "out of" + $numRanks() + "processes"
    print "I should only print once"

  if not encapsulated:
    reliqInit()
    hello_world()
    reliqFinalize()
  else: 
    reliq: 
      hello_world()