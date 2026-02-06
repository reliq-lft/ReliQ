#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/globalarrays/gawrap.nim
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

import gabase

GlobalArrays: discard

var 
  C_INT* {.importc: "C_INT", ga.}: cint
  C_LONGLONG* {.importc: "C_LONGLONG", ga.}: cint
  C_FLOAT* {.importc: "C_FLOAT", ga.}: cint
  C_DBL* {.importc: "C_DBL", ga.}: cint

# used for constructing global array

proc GA_Create_handle*: cint {.importc: "GA_Create_handle", ga.}

proc GA_Allocate*(g_a: cint): cint {.importc: "GA_Allocate", ga.}

proc GA_Set_name*(g_a: cint, name: ptr cchar) {.importc: "GA_Set_array_name", ga.}

proc GA_Set_data*(
  g_a: cint,  
  array_ndim: cint, 
  array_dims: ptr cint, 
  data_type: cint
) {.importc: "GA_Set_data", ga.}

proc GA_Set_chunk*(g_a: cint, chunk_dims: ptr cint) {.importc: "GA_Set_chunk", ga.}

proc GA_Set_ghosts*(g_a: cint, widths: ptr cint) {.importc: "GA_Set_ghosts", ga.}

proc GA_Destroy*(g_a: cint) {.importc: "GA_Destroy", ga.}

proc GA_Copy*(g_a: cint, g_b: cint) {.importc: "GA_Copy", ga.}

# accessing global array data

proc NGA_Distribution*(g_a: cint, pid: cint, lo: ptr cint, hi: ptr cint) 
  {.importc: "NGA_Distribution", ga.}

proc NGA_Access*(g_a: cint, lo: ptr cint, hi: ptr cint, p: ptr pointer, ld: ptr cint) 
  {.importc: "NGA_Access", ga.}

proc NGA_Access_ghosts*(g_a: cint, dims: ptr cint, p: ptr pointer, ld: ptr cint) 
  {.importc: "NGA_Access_ghosts", ga.}

# releasing global array data

proc NGA_Release*(g_a: cint, lo: ptr cint, hi: ptr cint) 
  {.importc: "NGA_Release", ga.}

# global array halo exchange

proc GA_Update_ghost_dir*(
  g_a: cint, 
  dir: cint, 
  side: cint,
  update_corners: cint
) {.importc: "NGA_Update_ghost_dir", ga.}

# synchronization, barriers, fences

proc GA_Init_fence*() {.importc: "GA_Init_fence", ga.}

proc GA_Fence*() {.importc: "GA_Fence", ga.}

proc GA_Sync*() {.importc: "GA_Sync", ga.}

# MPI information

proc GA_Nodeid*(): cint {.importc: "GA_Nodeid", ga.}

proc GA_Nnodes*(): cint {.importc: "GA_Nnodes", ga.}

#[ GlobalArrays global operations ]#

proc GA_Brdcst*(buf: pointer, n: cint, root: cint) 
  {.importc: "GA_Brdcst", ga, discardable.}

proc GA_Igop*(x: ptr int32, n: cint, op: cstring) 
  {.importc: "GA_Igop", ga, discardable.}

proc GA_Lgop*(x: ptr int64, n: cint, op: cstring) 
  {.importc: "GA_Lgop", ga, discardable.}

proc GA_Fgop*(x: ptr float32, n: cint, op: cstring) 
  {.importc: "GA_Fgop", ga, discardable.}

proc GA_Dgop*(x: ptr float64, n: cint, op: cstring) 
  {.importc: "GA_Dgop", ga, discardable.}

#[ derived wrappers ]#

proc toGAType*(t: typedesc[int32]): cint = C_INT

proc toGAType*(t: typedesc[int64]): cint = C_LONGLONG

proc toGAType*(t: typedesc[float32]): cint = C_FLOAT

proc toGAType*(t: typedesc[float64]): cint = C_DBL

proc newHandle*(): cint =
  result = GA_Create_handle()
  GA_Sync()

proc setName*(handle: cint; name: cstring) =
  handle.GA_Set_name(cast[ptr cchar](name))
  GA_Sync()

proc setData*[D: static[int]](handle: cint; dims: array[D, cint]; T: typedesc) =
  handle.GA_Set_data(cint(D), addr dims[0], toGAType(T)) 
  GA_Sync()

proc setChunk*[D: static[int]](handle: cint; chunks: array[D, cint]) =
  handle.GA_Set_chunk(addr chunks[0]) 
  GA_Sync()

proc setGhosts*[D: static[int]](handle: cint; widths: array[D, cint]) =
  handle.GA_Set_ghosts(addr widths[0]) 
  GA_Sync()

proc alloc*(handle: cint) =
  let status = handle.GA_Allocate()
  GA_Sync()
  if status == 0:
    raise newException(ValueError, "Global Array allocation failed.")

proc localIndices*[D: static[int]](handle: cint; lo, hi: var array[D, cint]) =
  NGA_Distribution(handle, GA_Nodeid(), addr lo[0], addr hi[0])

#[ unit tests ]#

when isMainModule:
  echo "no tests for gawrap"