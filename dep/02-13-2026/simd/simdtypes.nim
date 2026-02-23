#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/simd/simdtypes.nim
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

## SIMD Vector Types
## 
## This module provides generic SIMD vector wrapper types that abstract
## over different SIMD widths (SSE, AVX2, AVX-512) and support runtime
## SIMD width selection within the AoSoA memory layout.
##
## The `SimdVec[N, T]` type represents N values of type T that can be
## processed in parallel using SIMD instructions.
##
## Reference: QEX simdWrap.nim and nimsimd for design patterns

import x86wrap
import avx2wrap
import avx512wrap

export x86wrap

type
  SimdVec*[N: static[int], T] = object
    ## Generic SIMD vector holding N elements of type T
    ## N is the number of SIMD lanes (e.g., 4, 8, 16)
    ## T is the scalar type (float32, float64, int32, int64)
    data*: array[N, T]

  # Type aliases for common configurations
  SimdF32x4* = SimdVec[4, float32]   ## SSE: 4 x float32
  SimdF32x8* = SimdVec[8, float32]   ## AVX2: 8 x float32
  SimdF32x16* = SimdVec[16, float32] ## AVX-512: 16 x float32
  
  SimdF64x2* = SimdVec[2, float64]   ## SSE: 2 x float64
  SimdF64x4* = SimdVec[4, float64]   ## AVX2: 4 x float64
  SimdF64x8* = SimdVec[8, float64]   ## AVX-512: 8 x float64
  
  SimdI32x4* = SimdVec[4, int32]     ## SSE: 4 x int32
  SimdI32x8* = SimdVec[8, int32]     ## AVX2: 8 x int32
  SimdI32x16* = SimdVec[16, int32]   ## AVX-512: 16 x int32
  
  SimdI64x2* = SimdVec[2, int64]     ## SSE: 2 x int64
  SimdI64x4* = SimdVec[4, int64]     ## AVX2: 4 x int64
  SimdI64x8* = SimdVec[8, int64]     ## AVX-512: 8 x int64

#[ ============================================================================
   Runtime SIMD Vector for Dynamic Width
   ============================================================================ ]#

type
  SimdVecDyn*[T] = object
    ## Dynamic SIMD vector with runtime-determined width
    ## Used when SIMD width is not known at compile time
    width*: int
    data*: seq[T]

proc newSimdVecDyn*[T](width: int): SimdVecDyn[T] =
  ## Create a zero-initialized dynamic SIMD vector
  result.width = width
  result.data = newSeq[T](width)

proc newSimdVecDyn*[T](width: int, val: T): SimdVecDyn[T] =
  ## Create a dynamic SIMD vector filled with a scalar value
  result.width = width
  result.data = newSeq[T](width)
  for i in 0..<width:
    result.data[i] = val

#[ ============================================================================
   Constructors and Basic Operations for SimdVec[N, T]
   ============================================================================ ]#

proc zero*[N: static[int], T](): SimdVec[N, T] {.inline.} =
  ## Create a zero-initialized SIMD vector
  discard  # Arrays are zero-initialized by default

proc splat*[N: static[int], T](val: T): SimdVec[N, T] {.inline.} =
  ## Broadcast a scalar to all SIMD lanes
  for i in 0..<N:
    result.data[i] = val

proc `[]`*[N: static[int], T](v: SimdVec[N, T], i: int): T {.inline.} =
  ## Access individual lane
  v.data[i]

proc `[]=`*[N: static[int], T](v: var SimdVec[N, T], i: int, val: T) {.inline.} =
  ## Set individual lane
  v.data[i] = val

proc len*[N: static[int], T](v: SimdVec[N, T]): int {.inline.} =
  ## Return number of SIMD lanes
  N

#[ ============================================================================
   Arithmetic Operations - Generic Implementation
   ============================================================================ ]#

proc `+`*[N: static[int], T](a, b: SimdVec[N, T]): SimdVec[N, T] {.inline.} =
  ## Element-wise addition
  for i in 0..<N:
    result.data[i] = a.data[i] + b.data[i]

proc `-`*[N: static[int], T](a, b: SimdVec[N, T]): SimdVec[N, T] {.inline.} =
  ## Element-wise subtraction
  for i in 0..<N:
    result.data[i] = a.data[i] - b.data[i]

proc `*`*[N: static[int], T](a, b: SimdVec[N, T]): SimdVec[N, T] {.inline.} =
  ## Element-wise multiplication
  for i in 0..<N:
    result.data[i] = a.data[i] * b.data[i]

proc `/`*[N: static[int], T](a, b: SimdVec[N, T]): SimdVec[N, T] {.inline.} =
  ## Element-wise division
  for i in 0..<N:
    result.data[i] = a.data[i] / b.data[i]

proc `-`*[N: static[int], T](a: SimdVec[N, T]): SimdVec[N, T] {.inline.} =
  ## Unary negation
  for i in 0..<N:
    result.data[i] = -a.data[i]

# Scalar operations
proc `+`*[N: static[int], T](a: SimdVec[N, T], b: T): SimdVec[N, T] {.inline.} =
  for i in 0..<N: result.data[i] = a.data[i] + b

proc `+`*[N: static[int], T](a: T, b: SimdVec[N, T]): SimdVec[N, T] {.inline.} =
  for i in 0..<N: result.data[i] = a + b.data[i]

proc `-`*[N: static[int], T](a: SimdVec[N, T], b: T): SimdVec[N, T] {.inline.} =
  for i in 0..<N: result.data[i] = a.data[i] - b

proc `-`*[N: static[int], T](a: T, b: SimdVec[N, T]): SimdVec[N, T] {.inline.} =
  for i in 0..<N: result.data[i] = a - b.data[i]

proc `*`*[N: static[int], T](a: SimdVec[N, T], b: T): SimdVec[N, T] {.inline.} =
  for i in 0..<N: result.data[i] = a.data[i] * b

proc `*`*[N: static[int], T](a: T, b: SimdVec[N, T]): SimdVec[N, T] {.inline.} =
  for i in 0..<N: result.data[i] = a * b.data[i]

proc `/`*[N: static[int], T](a: SimdVec[N, T], b: T): SimdVec[N, T] {.inline.} =
  for i in 0..<N: result.data[i] = a.data[i] / b

# In-place operations
proc `+=`*[N: static[int], T](a: var SimdVec[N, T], b: SimdVec[N, T]) {.inline.} =
  for i in 0..<N: a.data[i] += b.data[i]

proc `-=`*[N: static[int], T](a: var SimdVec[N, T], b: SimdVec[N, T]) {.inline.} =
  for i in 0..<N: a.data[i] -= b.data[i]

proc `*=`*[N: static[int], T](a: var SimdVec[N, T], b: SimdVec[N, T]) {.inline.} =
  for i in 0..<N: a.data[i] *= b.data[i]

proc `/=`*[N: static[int], T](a: var SimdVec[N, T], b: SimdVec[N, T]) {.inline.} =
  for i in 0..<N: a.data[i] /= b.data[i]

proc `+=`*[N: static[int], T](a: var SimdVec[N, T], b: T) {.inline.} =
  for i in 0..<N: a.data[i] += b

proc `-=`*[N: static[int], T](a: var SimdVec[N, T], b: T) {.inline.} =
  for i in 0..<N: a.data[i] -= b

proc `*=`*[N: static[int], T](a: var SimdVec[N, T], b: T) {.inline.} =
  for i in 0..<N: a.data[i] *= b

proc `/=`*[N: static[int], T](a: var SimdVec[N, T], b: T) {.inline.} =
  for i in 0..<N: a.data[i] /= b

#[ ============================================================================
   Reduction Operations
   ============================================================================ ]#

proc sum*[N: static[int], T](v: SimdVec[N, T]): T {.inline.} =
  ## Sum all lanes
  result = v.data[0]
  for i in 1..<N:
    result += v.data[i]

proc product*[N: static[int], T](v: SimdVec[N, T]): T {.inline.} =
  ## Multiply all lanes
  result = v.data[0]
  for i in 1..<N:
    result *= v.data[i]

proc min*[N: static[int], T](v: SimdVec[N, T]): T {.inline.} =
  ## Find minimum across all lanes
  result = v.data[0]
  for i in 1..<N:
    if v.data[i] < result:
      result = v.data[i]

proc max*[N: static[int], T](v: SimdVec[N, T]): T {.inline.} =
  ## Find maximum across all lanes
  result = v.data[0]
  for i in 1..<N:
    if v.data[i] > result:
      result = v.data[i]

#[ ============================================================================
   Load/Store Operations
   ============================================================================ ]#

proc load*[N: static[int], T](p: ptr T): SimdVec[N, T] {.inline.} =
  ## Load N consecutive values from memory
  let arr = cast[ptr UncheckedArray[T]](p)
  for i in 0..<N:
    result.data[i] = arr[i]

proc store*[N: static[int], T](v: SimdVec[N, T], p: ptr T) {.inline.} =
  ## Store N values to consecutive memory locations
  let arr = cast[ptr UncheckedArray[T]](p)
  for i in 0..<N:
    arr[i] = v.data[i]

proc loadStrided*[N: static[int], T](p: ptr T, stride: int): SimdVec[N, T] {.inline.} =
  ## Load N values with given stride (gather)
  let arr = cast[ptr UncheckedArray[T]](p)
  for i in 0..<N:
    result.data[i] = arr[i * stride]

proc storeStrided*[N: static[int], T](v: SimdVec[N, T], p: ptr T, stride: int) {.inline.} =
  ## Store N values with given stride (scatter)
  let arr = cast[ptr UncheckedArray[T]](p)
  for i in 0..<N:
    arr[i * stride] = v.data[i]

#[ ============================================================================
   AVX2 Specialized Operations (8 x float32, 4 x float64)
   ============================================================================ ]#

when defined(AVX2) or defined(avx2):
  # float32 x 8 - AVX2
  proc add*(a, b: SimdF32x8): SimdF32x8 {.inline.} =
    let ma = cast[m256](a.data)
    let mb = cast[m256](b.data)
    let mr = mm256_add_ps(ma, mb)
    result.data = cast[array[8, float32]](mr)
  
  proc sub*(a, b: SimdF32x8): SimdF32x8 {.inline.} =
    let ma = cast[m256](a.data)
    let mb = cast[m256](b.data)
    let mr = mm256_sub_ps(ma, mb)
    result.data = cast[array[8, float32]](mr)
  
  proc mul*(a, b: SimdF32x8): SimdF32x8 {.inline.} =
    let ma = cast[m256](a.data)
    let mb = cast[m256](b.data)
    let mr = mm256_mul_ps(ma, mb)
    result.data = cast[array[8, float32]](mr)
  
  proc madd*(a, b, c: SimdF32x8): SimdF32x8 {.inline.} =
    ## Fused multiply-add: a * b + c
    let ma = cast[m256](a.data)
    let mb = cast[m256](b.data)
    let mc = cast[m256](c.data)
    let mr = mm256_fmadd_ps(ma, mb, mc)
    result.data = cast[array[8, float32]](mr)
  
  # float64 x 4 - AVX2
  proc add*(a, b: SimdF64x4): SimdF64x4 {.inline.} =
    let ma = cast[m256d](a.data)
    let mb = cast[m256d](b.data)
    let mr = mm256_add_pd(ma, mb)
    result.data = cast[array[4, float64]](mr)
  
  proc sub*(a, b: SimdF64x4): SimdF64x4 {.inline.} =
    let ma = cast[m256d](a.data)
    let mb = cast[m256d](b.data)
    let mr = mm256_sub_pd(ma, mb)
    result.data = cast[array[4, float64]](mr)
  
  proc mul*(a, b: SimdF64x4): SimdF64x4 {.inline.} =
    let ma = cast[m256d](a.data)
    let mb = cast[m256d](b.data)
    let mr = mm256_mul_pd(ma, mb)
    result.data = cast[array[4, float64]](mr)
  
  proc madd*(a, b, c: SimdF64x4): SimdF64x4 {.inline.} =
    ## Fused multiply-add: a * b + c
    let ma = cast[m256d](a.data)
    let mb = cast[m256d](b.data)
    let mc = cast[m256d](c.data)
    let mr = mm256_fmadd_pd(ma, mb, mc)
    result.data = cast[array[4, float64]](mr)

#[ ============================================================================
   AVX-512 Specialized Operations (16 x float32, 8 x float64)
   ============================================================================ ]#

when defined(AVX512) or defined(avx512):
  # float32 x 16 - AVX-512
  proc add*(a, b: SimdF32x16): SimdF32x16 {.inline.} =
    let ma = cast[m512](a.data)
    let mb = cast[m512](b.data)
    let mr = mm512_add_ps(ma, mb)
    result.data = cast[array[16, float32]](mr)
  
  proc sub*(a, b: SimdF32x16): SimdF32x16 {.inline.} =
    let ma = cast[m512](a.data)
    let mb = cast[m512](b.data)
    let mr = mm512_sub_ps(ma, mb)
    result.data = cast[array[16, float32]](mr)
  
  proc mul*(a, b: SimdF32x16): SimdF32x16 {.inline.} =
    let ma = cast[m512](a.data)
    let mb = cast[m512](b.data)
    let mr = mm512_mul_ps(ma, mb)
    result.data = cast[array[16, float32]](mr)
  
  proc madd*(a, b, c: SimdF32x16): SimdF32x16 {.inline.} =
    let ma = cast[m512](a.data)
    let mb = cast[m512](b.data)
    let mc = cast[m512](c.data)
    let mr = mm512_fmadd_ps(ma, mb, mc)
    result.data = cast[array[16, float32]](mr)
  
  # float64 x 8 - AVX-512
  proc add*(a, b: SimdF64x8): SimdF64x8 {.inline.} =
    let ma = cast[m512d](a.data)
    let mb = cast[m512d](b.data)
    let mr = mm512_add_pd(ma, mb)
    result.data = cast[array[8, float64]](mr)
  
  proc sub*(a, b: SimdF64x8): SimdF64x8 {.inline.} =
    let ma = cast[m512d](a.data)
    let mb = cast[m512d](b.data)
    let mr = mm512_sub_pd(ma, mb)
    result.data = cast[array[8, float64]](mr)
  
  proc mul*(a, b: SimdF64x8): SimdF64x8 {.inline.} =
    let ma = cast[m512d](a.data)
    let mb = cast[m512d](b.data)
    let mr = mm512_mul_pd(ma, mb)
    result.data = cast[array[8, float64]](mr)
  
  proc madd*(a, b, c: SimdF64x8): SimdF64x8 {.inline.} =
    let ma = cast[m512d](a.data)
    let mb = cast[m512d](b.data)
    let mc = cast[m512d](c.data)
    let mr = mm512_fmadd_pd(ma, mb, mc)
    result.data = cast[array[8, float64]](mr)

#[ ============================================================================
   Dynamic SIMD Vector Operations
   ============================================================================ ]#

proc `+`*[T](a, b: SimdVecDyn[T]): SimdVecDyn[T] {.inline.} =
  assert a.width == b.width
  result.width = a.width
  result.data = newSeq[T](a.width)
  for i in 0..<a.width:
    result.data[i] = a.data[i] + b.data[i]

proc `-`*[T](a, b: SimdVecDyn[T]): SimdVecDyn[T] {.inline.} =
  assert a.width == b.width
  result.width = a.width
  result.data = newSeq[T](a.width)
  for i in 0..<a.width:
    result.data[i] = a.data[i] - b.data[i]

proc `*`*[T](a, b: SimdVecDyn[T]): SimdVecDyn[T] {.inline.} =
  assert a.width == b.width
  result.width = a.width
  result.data = newSeq[T](a.width)
  for i in 0..<a.width:
    result.data[i] = a.data[i] * b.data[i]

proc `/`*[T](a, b: SimdVecDyn[T]): SimdVecDyn[T] {.inline.} =
  assert a.width == b.width
  result.width = a.width
  result.data = newSeq[T](a.width)
  for i in 0..<a.width:
    result.data[i] = a.data[i] / b.data[i]

proc `*`*[T](a: T, b: SimdVecDyn[T]): SimdVecDyn[T] {.inline.} =
  result.width = b.width
  result.data = newSeq[T](b.width)
  for i in 0..<b.width:
    result.data[i] = a * b.data[i]

proc `*`*[T](a: SimdVecDyn[T], b: T): SimdVecDyn[T] {.inline.} =
  result.width = a.width
  result.data = newSeq[T](a.width)
  for i in 0..<a.width:
    result.data[i] = a.data[i] * b

proc `+`*[T](a: SimdVecDyn[T], b: T): SimdVecDyn[T] {.inline.} =
  result.width = a.width
  result.data = newSeq[T](a.width)
  for i in 0..<a.width:
    result.data[i] = a.data[i] + b

proc `+`*[T](a: T, b: SimdVecDyn[T]): SimdVecDyn[T] {.inline.} =
  result.width = b.width
  result.data = newSeq[T](b.width)
  for i in 0..<b.width:
    result.data[i] = a + b.data[i]

proc `-`*[T](a: SimdVecDyn[T], b: T): SimdVecDyn[T] {.inline.} =
  result.width = a.width
  result.data = newSeq[T](a.width)
  for i in 0..<a.width:
    result.data[i] = a.data[i] - b

proc `/`*[T](a: SimdVecDyn[T], b: T): SimdVecDyn[T] {.inline.} =
  result.width = a.width
  result.data = newSeq[T](a.width)
  for i in 0..<a.width:
    result.data[i] = a.data[i] / b

proc sum*[T](v: SimdVecDyn[T]): T {.inline.} =
  result = v.data[0]
  for i in 1..<v.width:
    result += v.data[i]

#[ ============================================================================
   String Representation
   ============================================================================ ]#

proc `$`*[N: static[int], T](v: SimdVec[N, T]): string =
  result = "SimdVec[" & $N & "]("
  for i in 0..<N:
    if i > 0: result &= ", "
    result &= $v.data[i]
  result &= ")"

proc `$`*[T](v: SimdVecDyn[T]): string =
  result = "SimdVecDyn[" & $v.width & "]("
  for i in 0..<v.width:
    if i > 0: result &= ", "
    result &= $v.data[i]
  result &= ")"

#[ ============================================================================
   Tests
   ============================================================================ ]#

when isMainModule:
  import std/unittest
  import std/strutils
  
  suite "SimdVec Types":
    
    test "SimdVec construction and access":
      var v: SimdF64x4
      v[0] = 1.0
      v[1] = 2.0
      v[2] = 3.0
      v[3] = 4.0
      
      check v[0] == 1.0
      check v[3] == 4.0
      check v.len == 4
    
    test "Splat broadcast":
      let v = splat[8, float32](3.14'f32)
      for i in 0..<8:
        check v[i] == 3.14'f32
    
    test "Vector addition":
      var a, b: SimdF64x4
      for i in 0..<4:
        a[i] = float64(i)
        b[i] = float64(i * 2)
      
      let c = a + b
      for i in 0..<4:
        check c[i] == float64(i * 3)
    
    test "Vector multiplication":
      var a, b: SimdF32x8
      for i in 0..<8:
        a[i] = 2.0'f32
        b[i] = float32(i)
      
      let c = a * b
      for i in 0..<8:
        check c[i] == 2.0'f32 * float32(i)
    
    test "Scalar multiplication":
      var v: SimdF64x4
      for i in 0..<4:
        v[i] = float64(i + 1)
      
      let scaled = 2.0 * v
      for i in 0..<4:
        check scaled[i] == 2.0 * float64(i + 1)
    
    test "Sum reduction":
      var v: SimdF64x4
      v[0] = 1.0; v[1] = 2.0; v[2] = 3.0; v[3] = 4.0
      check sum(v) == 10.0
    
    test "Load and store":
      var arr: array[8, float32] = [1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
      let v = load[8, float32](addr arr[0])
      
      for i in 0..<8:
        check v[i] == arr[i]
      
      var out_arr: array[8, float32]
      store(v, addr out_arr[0])
      for i in 0..<8:
        check out_arr[i] == arr[i]
    
    test "Dynamic SIMD vector":
      var a = newSimdVecDyn[float64](6)
      var b = newSimdVecDyn[float64](6)
      
      for i in 0..<6:
        a.data[i] = float64(i)
        b.data[i] = float64(i * 2)
      
      let c = a + b
      check c.width == 6
      for i in 0..<6:
        check c.data[i] == float64(i * 3)
    
    test "In-place operations":
      var v: SimdF64x4
      for i in 0..<4: v[i] = float64(i)
      
      v += splat[4, float64](1.0)
      for i in 0..<4:
        check v[i] == float64(i + 1)
    
    test "String representation":
      var v: SimdF32x4
      v[0] = 1.0'f32; v[1] = 2.0'f32; v[2] = 3.0'f32; v[3] = 4.0'f32
      let s = $v
      check "SimdVec[4]" in s
      check "1.0" in s
