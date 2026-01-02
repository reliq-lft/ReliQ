#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/kokkos/kokkostypes.nim
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

import kokkosbase

Kokkos: 
  {.pragma: kokkostypes, header: "../kokkos/kokkostypes.hpp".}
  {.pragma: rng, header: "<Kokkos_Random.hpp>".}

const 
  XORSHIFT64 = "Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>"
  XORSHIFT1024 = "Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>"

type KokkosHandle* = pointer

type 
  KokkosRNG64Obj {.importcpp: XORSHIFT64, rng.} = object
  KokkosRNG64* = ptr KokkosRNG64Obj

type 
  KokkosRNG1024Obj {.importcpp: XORSHIFT1024, rng.} = object
  KokkosRNG1024* = ptr KokkosRNG1024Obj

#[ View wrappers ]#

proc createViewInt32*(data: pointer, rank: csize_t, dims: ptr csize_t): pointer 
  {.importc: "create_kokkos_view<int>", kokkostypes, nodecl.}

proc createViewInt64*(data: pointer, rank: csize_t, dims: ptr csize_t): pointer 
  {.importc: "create_kokkos_view<long long>", kokkostypes, nodecl.}

proc createViewFloat32*(data: pointer, rank: csize_t, dims: ptr csize_t): pointer 
  {.importc: "create_kokkos_view<float>", kokkostypes, nodecl.}

proc createViewFloat64*(data: pointer, rank: csize_t, dims: ptr csize_t): pointer 
  {.importc: "create_kokkos_view<double>", kokkostypes, nodecl.}

proc destroyViewInt32*(handle: pointer, rank: csize_t) 
  {.importc: "destroy_kokkos_view<int>", kokkostypes, nodecl.}

proc destroyViewInt64*(handle: pointer, rank: csize_t) 
  {.importc: "destroy_kokkos_view<long long>", kokkostypes, nodecl.}

proc destroyViewFloat32*(handle: pointer, rank: csize_t) 
  {.importc: "destroy_kokkos_view<float>", kokkostypes, nodecl.}

proc destroyViewFloat64*(handle: pointer, rank: csize_t) 
  {.importc: "destroy_kokkos_view<double>", kokkostypes, nodecl.}

proc viewGetInt32*(handle: pointer, rank: csize_t, indices: ptr csize_t): cint 
  {.importc: "view_get<int>", kokkostypes, nodecl.}

proc viewGetInt64*(handle: pointer, rank: csize_t, indices: ptr csize_t): clonglong 
  {.importc: "view_get<long long>", kokkostypes, nodecl.}

proc viewGetFloat32*(handle: pointer, rank: csize_t, indices: ptr csize_t): cfloat 
  {.importc: "view_get<float>", kokkostypes, nodecl.}

proc viewGetFloat64*(handle: pointer, rank: csize_t, indices: ptr csize_t): cdouble 
  {.importc: "view_get<double>", kokkostypes, nodecl.}

proc viewSetInt32*(handle: pointer, rank: csize_t, indices: ptr csize_t, value: cint) 
  {.importc: "view_set<int>", kokkostypes, nodecl.}

proc viewSetInt64*(handle: pointer, rank: csize_t, indices: ptr csize_t, value: clonglong) 
  {.importc: "view_set<long long>", kokkostypes, nodecl.}

proc viewSetFloat32*(handle: pointer, rank: csize_t, indices: ptr csize_t, value: cfloat) 
  {.importc: "view_set<float>", kokkostypes, nodecl.}

proc viewSetFloat64*(handle: pointer, rank: csize_t, indices: ptr csize_t, value: cdouble) 
  {.importc: "view_set<double>", kokkostypes, nodecl.}

#[ RNG wrappers ]#

proc newRNG64*(seed: uint64): KokkosRNG64
  {.importcpp: "new " & XORSHIFT64 & "(#)", rng.}

proc newRNG1024*(seed: uint64): KokkosRNG1024
  {.importcpp: "new " & XORSHIFT1024 & "(#)", rng.}

proc destroyRNG64*(rng: KokkosRNG64)
  {.importcpp: "delete #", rng.}

proc destroyRNG1024*(rng: KokkosRNG1024)
  {.importcpp: "delete #", rng.}