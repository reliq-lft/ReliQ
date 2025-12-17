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

# synchronization

proc GA_Sync*() {.importc: "GA_Sync", ga.}

# MPI information

proc GA_Nodeid*(): cint {.importc: "GA_Nodeid", ga.}

proc GA_Nnodes*(): cint {.importc: "GA_Nnodes", ga.}

#[ unit tests ]#

when isMainModule:
  echo "no tests for gawrap"