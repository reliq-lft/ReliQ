#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/parallel.nim
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

import globalarrays/[globalarrays]
import openmp/[omplocal]

export globalarrays
export omplocal 

# Backend selection - matches tensorview.nim's UseSycl/UseOpenMP flags
const UseSycl* {.booldefine.} = false
const UseOpenMP* {.booldefine.} = false

when UseOpenMP:
  import openmp/[openmp]
  export openmp
elif UseSycl:
  import sycl/[sycl]
  export sycl
else:
  import opencl/[opencl]
  export opencl

template parallel*(body: untyped): untyped =
  bind UseOpenMP, UseSycl
  gaParallel:
    when UseOpenMP:
      ompParallel: body
    elif UseSycl:
      syclParallel: body
    else:
      clParallel: body

template accelerator*(body: untyped): untyped =
  ## Scoping block for GPU/accelerator operations.
  ## TensorFieldViews are created and ``each`` loops run within this block.
  ## On exit, views are synced/destroyed in order.
  bind UseOpenMP, UseSycl
  block:
    when UseOpenMP:
      ompParallel: body
    elif UseSycl:
      syclParallel: body
    else:
      clParallel: body

template local*(body: untyped): untyped =
  ## Scoping block for CPU-local operations.
  ## LocalTensorFields are created and accessed within this block.
  block:
    body