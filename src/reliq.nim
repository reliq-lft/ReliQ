#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/reliq.nim
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

const UseSycl* {.booldefine.} = false
const UseOpenMP* {.booldefine.} = false

import ga/[ga]
import memory/[bufferpool]
import memory/[coherence]

# <***> will be where view is exported <***>
when defined(UseOpenMP):
  import openmp/[openmp]
elif defined(UseOpenCL):
  import opencl/[opencl]
else:
  import opencl/[opencl]

when defined(UseOpenMP):
  type
    DeviceBuffer* = pointer
    DeviceQueue* = pointer
elif defined(UseOpenCL):
  type
    DeviceBuffer* = PMem
    DeviceQueue* = PCommandQueue
else:
  type
    DeviceBuffer* = PMem
    DeviceQueue* = PCommandQueue
    
var globalBufferPool* {.inject.} = newBufferPool()
var globalCoherenceManager* {.inject.} = newCoherenceManager()

template reliq*(body: untyped): untyped =
  gaParallel:
    block: body
  globalBufferPool.drain()

template accelerator*(body: untyped): untyped =
  ## Scoping block for GPU/accelerator operations.
  ## TensorFieldViews are created and ``each`` loops run within this block.
  ## On exit, views are synced/destroyed in order.
  bind UseOpenMP, UseOpenCL
  when defined(UseOpenMP):
    ompParallel:
      block: body
  elif defined(UseOpenCL):
    oclParallel:
      block: body
  else:
    oclParallel:
      block: body

template local*(body: untyped): untyped =
  ## Scoping block for CPU-local operations.
  ## LocalTensorFields are created and accessed within this block.
  block: body
