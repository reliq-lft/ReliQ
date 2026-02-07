#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/tensor/sitetensor.nim
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

## Site tensor types for lattice field theory
##
## These types represent tensors at a single lattice site with compile-time
## known dimensions. They are designed for use with OpenCL kernel generation,
## where operations are translated to inline C code.

import std/strformat
import std/strutils  # for join

# Backend selection - same as tensorview.nim
const UseSycl* {.booldefine.} = false
const UseOpenMP* {.booldefine.} = false

when UseOpenMP:
  import ../simd/simdlayout

#[ Vector type - compile-time sized ]#

type Vec*[N: static[int], T] = object
  ## A vector with compile-time known size N and element type T
  ## Storage is a flat array of N elements
  data*: array[N, T]

proc `[]`*[N: static[int], T](v: Vec[N, T], i: int): T {.inline.} =
  ## Element access
  v.data[i]

proc `[]=`*[N: static[int], T](v: var Vec[N, T], i: int, val: T) {.inline.} =
  ## Element assignment
  v.data[i] = val

proc `+`*[N: static[int], T](a, b: Vec[N, T]): Vec[N, T] {.inline.} =
  ## Vector addition
  for i in 0..<N:
    result.data[i] = a.data[i] + b.data[i]

proc `-`*[N: static[int], T](a, b: Vec[N, T]): Vec[N, T] {.inline.} =
  ## Vector subtraction
  for i in 0..<N:
    result.data[i] = a.data[i] - b.data[i]

proc `*`*[N: static[int], T](a: T, v: Vec[N, T]): Vec[N, T] {.inline.} =
  ## Scalar-vector multiplication
  for i in 0..<N:
    result.data[i] = a * v.data[i]

proc `*`*[N: static[int], T](v: Vec[N, T], a: T): Vec[N, T] {.inline.} =
  ## Vector-scalar multiplication
  for i in 0..<N:
    result.data[i] = v.data[i] * a

proc dot*[N: static[int], T](a, b: Vec[N, T]): T {.inline.} =
  ## Dot product
  result = T(0)
  for i in 0..<N:
    result = result + a.data[i] * b.data[i]

#[ Matrix type - compile-time sized, row-major storage ]#

type Mat*[N: static[int], M: static[int], T] = object
  ## A matrix with compile-time known dimensions N x M and element type T
  ## Storage is row-major: data[i * M + j] = element at row i, column j
  data*: array[N * M, T]

proc `[]`*[N: static[int], M: static[int], T](m: Mat[N, M, T], i, j: int): T {.inline.} =
  ## Element access at (row i, column j)
  m.data[i * M + j]

proc `[]=`*[N: static[int], M: static[int], T](m: var Mat[N, M, T], i, j: int, val: T) {.inline.} =
  ## Element assignment at (row i, column j)
  m.data[i * M + j] = val

proc `+`*[N: static[int], M: static[int], T](a, b: Mat[N, M, T]): Mat[N, M, T] {.inline.} =
  ## Matrix addition
  for i in 0..<N*M:
    result.data[i] = a.data[i] + b.data[i]

proc `-`*[N: static[int], M: static[int], T](a, b: Mat[N, M, T]): Mat[N, M, T] {.inline.} =
  ## Matrix subtraction
  for i in 0..<N*M:
    result.data[i] = a.data[i] - b.data[i]

proc `*`*[N: static[int], M: static[int], T](a: T, m: Mat[N, M, T]): Mat[N, M, T] {.inline.} =
  ## Scalar-matrix multiplication
  for i in 0..<N*M:
    result.data[i] = a * m.data[i]

proc `*`*[N: static[int], M: static[int], T](m: Mat[N, M, T], a: T): Mat[N, M, T] {.inline.} =
  ## Matrix-scalar multiplication
  for i in 0..<N*M:
    result.data[i] = m.data[i] * a

proc `*`*[N: static[int], K: static[int], M: static[int], T](a: Mat[N, K, T], b: Mat[K, M, T]): Mat[N, M, T] {.inline.} =
  ## Matrix-matrix multiplication: (N x K) * (K x M) -> (N x M)
  ## C[i,j] = sum_k A[i,k] * B[k,j]
  for i in 0..<N:
    for j in 0..<M:
      var sum = T(0)
      for k in 0..<K:
        sum = sum + a[i, k] * b[k, j]
      result[i, j] = sum

proc `*`*[N: static[int], M: static[int], T](m: Mat[N, M, T], v: Vec[M, T]): Vec[N, T] {.inline.} =
  ## Matrix-vector multiplication: (N x M) * (M) -> (N)
  for i in 0..<N:
    var sum = T(0)
    for j in 0..<M:
      sum = sum + m[i, j] * v[j]
    result[i] = sum

proc transpose*[N: static[int], M: static[int], T](m: Mat[N, M, T]): Mat[M, N, T] {.inline.} =
  ## Matrix transpose: (N x M) -> (M x N)
  for i in 0..<N:
    for j in 0..<M:
      result[j, i] = m[i, j]

proc trace*[N: static[int], T](m: Mat[N, N, T]): T {.inline.} =
  ## Trace of a square matrix
  result = T(0)
  for i in 0..<N:
    result = result + m[i, i]

proc identity*[N: static[int], T](): Mat[N, N, T] {.inline.} =
  ## Identity matrix
  for i in 0..<N:
    result[i, i] = T(1)

#[ Type helpers for TensorFieldView integration ]#

template isVec*(T: typedesc): bool =
  ## Check if type is a Vec
  T is Vec

template isMat*(T: typedesc): bool =
  ## Check if type is a Mat
  T is Mat

template isSiteTensor*(T: typedesc): bool =
  ## Check if type is a site tensor (Vec or Mat)
  isVec(T) or isMat(T)

template elementType*[N: static[int], T](v: typedesc[Vec[N, T]]): typedesc =
  ## Get the element type of a Vec
  T

template elementType*[N: static[int], M: static[int], T](m: typedesc[Mat[N, M, T]]): typedesc =
  ## Get the element type of a Mat
  T

template numElements*[N: static[int], T](v: typedesc[Vec[N, T]]): int =
  ## Get the number of elements in a Vec
  N

template numElements*[N: static[int], M: static[int], T](m: typedesc[Mat[N, M, T]]): int =
  ## Get the number of elements in a Mat
  N * M

# Compile-time dimension accessors
template vecSize*[N: static[int], T](v: typedesc[Vec[N, T]]): int =
  ## Get the size N of a Vec[N, T]
  N

template matRows*[N: static[int], M: static[int], T](m: typedesc[Mat[N, M, T]]): int =
  ## Get the row count N of a Mat[N, M, T]
  N

template matCols*[N: static[int], M: static[int], T](m: typedesc[Mat[N, M, T]]): int =
  ## Get the column count M of a Mat[N, M, T]
  M

# Storage type extraction (for complex support)
template storageType*(T: typedesc): typedesc =
  ## Get the underlying storage type for a site tensor or scalar
  ## For scalars, returns T. For Vec/Mat, returns the element type.
  when T is Vec:
    elementType(T)
  elif T is Mat:
    elementType(T)
  else:
    T

#[ Convenience type aliases ]#

type
  Vec2*[T] = Vec[2, T]
  Vec3*[T] = Vec[3, T]
  Vec4*[T] = Vec[4, T]
  
  Mat2*[T] = Mat[2, 2, T]
  Mat3*[T] = Mat[3, 3, T]
  Mat4*[T] = Mat[4, 4, T]
  
  # Common physics types
  Vec2f* = Vec[2, float32]
  Vec3f* = Vec[3, float32]
  Vec4f* = Vec[4, float32]
  Vec2d* = Vec[2, float64]
  Vec3d* = Vec[3, float64]
  Vec4d* = Vec[4, float64]
  
  Mat2f* = Mat[2, 2, float32]
  Mat3f* = Mat[3, 3, float32]
  Mat4f* = Mat[4, 4, float32]
  Mat2d* = Mat[2, 2, float64]
  Mat3d* = Mat[3, 3, float64]
  Mat4d* = Mat[4, 4, float64]

#[ Site tensor proxy and operation marker types for OpenCL codegen ]#
# These types tell the OpenCL codegen what operation to generate.
# They are phantoms - never executed at runtime, the `each` macro rewrites them.
# Parameterized by L (lattice type) and T (element type) for integration with TensorFieldView.

type
  TensorSiteProxy*[L, T] = object
    ## Proxy type returned by TensorFieldView[] - represents site tensor at index
    ## Operations on this proxy generate appropriate OpenCL kernel code
    ## Also supports runtime use for printing: stores actual tensor data when accessed
    view*: pointer  # Pointer to view
    site*: int
    # Runtime data for printing support
    runtimeData*: RuntimeSiteData[T]
    # OpenMP-specific fields for direct memory access
    when UseOpenMP:
      hostPtr*: pointer      # Direct host memory pointer
      shape*: seq[int]       # Tensor shape
      elemsPerSite*: int     # Elements per site
      hasSimdLayout*: bool   # True if data uses AoSoA layout
      # SIMD layout fields for AoSoA indexing
      simdLayoutPtr*: pointer  # Pointer to SimdLatticeLayout
      nSitesInner*: int        # Number of inner sites (simd lane count)

  TensorElementProxy*[L, T] = object
    ## Proxy type returned by TensorSiteProxy[] - represents single element at site
    ## Used for element-level read/write: mView[n][i,j] = value
    view*: pointer
    site*: int
    indices*: array[4, int]  # Up to 4D tensor element indices
    numIndices*: int
    when UseOpenMP:
      hostPtr*: pointer      # Direct host memory pointer
      shape*: seq[int]       # Tensor shape
      elemsPerSite*: int     # Elements per site

  # Runtime-accessible proxy for CPU fallback (print support)
  RuntimeSiteData*[T] = object
    ## Holds actual data for a site tensor - used for CPU fallback with printing
    data*: seq[T]           # Flat array of tensor elements  
    shape*: seq[int]        # Tensor shape (e.g., [2,2] for 2x2 matrix)
    rank*: int              # 0=scalar, 1=vector, 2=matrix, etc.

  # Operation result marker types
  MatMulResult*[L, T] = object
    ## Result of matrix-matrix multiplication: mat1[n] * mat2[n]
    proxyA*, proxyB*: TensorSiteProxy[L, T]
    when UseOpenMP:
      computed*: seq[T]     # Computed result for chained operations
      hasComputed*: bool    # Whether computed buffer is populated

  MatAddResult*[L, T] = object
    ## Result of matrix addition/subtraction: mat1[n] +/- mat2[n]
    proxyA*, proxyB*: TensorSiteProxy[L, T]
    isSubtraction*: bool  # true for subtraction, false for addition
    when UseOpenMP:
      # For chained expressions, one or both operands may be from computed results
      computedA*, computedB*: seq[T]
      hasComputedA*, hasComputedB*: bool

  VecAddResult*[L, T] = object
    ## Result of vector addition/subtraction: vec1[n] +/- vec2[n]
    proxyA*, proxyB*: TensorSiteProxy[L, T]
    isSubtraction*: bool  # true for subtraction, false for addition
    when UseOpenMP:
      computedA*, computedB*: seq[T]
      hasComputedA*, hasComputedB*: bool

  MatVecResult*[L, T] = object
    ## Result of matrix-vector multiplication: mat[n] * vec[n]
    proxyMat*, proxyVec*: TensorSiteProxy[L, T]

  ScalarMulResult*[L, T] = object
    ## Result of scalar multiplication: scalar * mat[n] or mat[n] * scalar
    proxy*: TensorSiteProxy[L, T]
    scalar*: T

  ScalarAddResult*[L, T] = object
    ## Result of scalar addition: scalar + mat[n] or mat[n] + scalar  
    proxy*: TensorSiteProxy[L, T]
    scalar*: T

  #[ LocalTensorField Site Proxy Types - for CPU-only "for all" loops ]#
  
  LocalSiteProxy*[D: static[int], R: static[int], L, T] = object
    ## Proxy type returned by LocalTensorField[] for "for all" loops
    ## Provides CPU-only access to tensor sites without GPU acceleration
    hostPtr*: pointer        # Direct host memory pointer
    site*: int               # Site index
    shape*: array[R, int]    # Tensor shape
    elemsPerSite*: int       # Elements per site

  LocalAddResult*[D: static[int], R: static[int], L, T] = object
    ## Result of local tensor addition/subtraction
    proxyA*, proxyB*: LocalSiteProxy[D, R, L, T]
    isSubtraction*: bool

  LocalMulResult*[D: static[int], R: static[int], L, T] = object
    ## Result of local tensor multiplication
    proxyA*, proxyB*: LocalSiteProxy[D, R, L, T]

  LocalScalarMulResult*[D: static[int], R: static[int], L, T] = object
    ## Result of local scalar * tensor
    proxy*: LocalSiteProxy[D, R, L, T]
    scalar*: T

  LocalScalarAddResult*[D: static[int], R: static[int], L, T] = object
    ## Result of local tensor + scalar
    proxy*: LocalSiteProxy[D, R, L, T]
    scalar*: T

#[ TensorSiteProxy operators for element access ]#

when UseOpenMP:
  # OpenMP backend: real implementations that access host memory directly
  
  proc computeElementIndex(shape: seq[int], indices: varargs[int]): int {.inline.} =
    ## Compute flat array index from tensor indices (row-major order)
    result = 0
    var stride = 1
    for i in countdown(shape.len - 1, 0):
      if i < indices.len:
        result += indices[i] * stride
      stride *= shape[i]
  
  # Helper to compute element offset for proxy (handles AoS and AoSoA)
  proc proxyElemOffset[L, T](proxy: TensorSiteProxy[L, T], elemIdx: int): int {.inline.} =
    if proxy.hasSimdLayout:
      let layout = cast[ptr SimdLatticeLayout](proxy.simdLayoutPtr)
      let (outer, inner) = localToOuterInner(proxy.site, layout[])
      outer * proxy.nSitesInner * proxy.elemsPerSite + elemIdx * proxy.nSitesInner + inner
    else:
      proxy.site * proxy.elemsPerSite + elemIdx
  
  # TensorSiteProxy[] for vectors: view[n][i] - read element
  proc `[]`*[L, T](proxy: TensorSiteProxy[L, T], i: int): T {.inline.} =
    let hostData = cast[ptr UncheckedArray[T]](proxy.hostPtr)
    hostData[proxy.proxyElemOffset(i)]
  
  # TensorSiteProxy[] for matrices: view[n][i, j] - read element
  proc `[]`*[L, T](proxy: TensorSiteProxy[L, T], i, j: int): T {.inline.} =
    let hostData = cast[ptr UncheckedArray[T]](proxy.hostPtr)
    let cols = if proxy.shape.len > 1: proxy.shape[1] else: 1
    let localIdx = i * cols + j
    hostData[proxy.proxyElemOffset(localIdx)]
  
  # TensorSiteProxy[] for 3D tensors: view[n][i, j, k] - read element
  proc `[]`*[L, T](proxy: TensorSiteProxy[L, T], i, j, k: int): T {.inline.} =
    let hostData = cast[ptr UncheckedArray[T]](proxy.hostPtr)
    let localIdx = computeElementIndex(proxy.shape, i, j, k)
    hostData[proxy.proxyElemOffset(localIdx)]
  
  # TensorSiteProxy[]= for vectors: view[n][i] = value
  proc `[]=`*[L, T](proxy: TensorSiteProxy[L, T], i: int, value: T) {.inline.} =
    let hostData = cast[ptr UncheckedArray[T]](proxy.hostPtr)
    hostData[proxy.proxyElemOffset(i)] = value
  
  # TensorSiteProxy[]= for matrices: view[n][i, j] = value
  proc `[]=`*[L, T](proxy: TensorSiteProxy[L, T], i, j: int, value: T) {.inline.} =
    let hostData = cast[ptr UncheckedArray[T]](proxy.hostPtr)
    let cols = if proxy.shape.len > 1: proxy.shape[1] else: 1
    let localIdx = i * cols + j
    hostData[proxy.proxyElemOffset(localIdx)] = value
  
  # TensorSiteProxy[]= for 3D tensors: view[n][i, j, k] = value
  proc `[]=`*[L, T](proxy: TensorSiteProxy[L, T], i, j, k: int, value: T) {.inline.} =
    let hostData = cast[ptr UncheckedArray[T]](proxy.hostPtr)
    let localIdx = computeElementIndex(proxy.shape, i, j, k)
    let elemIdx = proxy.site * proxy.elemsPerSite + localIdx
    hostData[elemIdx] = value

else:
  # OpenCL/SYCL backend: phantom operators for kernel codegen
  
  # TensorSiteProxy[] returns element proxy for element-level access
  # For vectors: view[n][i]
  proc `[]`*[L, T](proxy: TensorSiteProxy[L, T], i: int): TensorElementProxy[L, T] =
    raise newException(Defect, "TensorSiteProxy[i] phantom operator")

  # For matrices: view[n][i, j]
  proc `[]`*[L, T](proxy: TensorSiteProxy[L, T], i, j: int): TensorElementProxy[L, T] =
    raise newException(Defect, "TensorSiteProxy[i,j] phantom operator")

  # For 3D tensors: view[n][i, j, k]
  proc `[]`*[L, T](proxy: TensorSiteProxy[L, T], i, j, k: int): TensorElementProxy[L, T] =
    raise newException(Defect, "TensorSiteProxy[i,j,k] phantom operator")

  # TensorSiteProxy[]= for writing individual elements
  proc `[]=`*[L, T](proxy: TensorSiteProxy[L, T], i: int, value: T) =
    raise newException(Defect, "TensorSiteProxy[i]= phantom operator")

  proc `[]=`*[L, T](proxy: TensorSiteProxy[L, T], i, j: int, value: T) =
    raise newException(Defect, "TensorSiteProxy[i,j]= phantom operator")

  proc `[]=`*[L, T](proxy: TensorSiteProxy[L, T], i, j, k: int, value: T) =
    raise newException(Defect, "TensorSiteProxy[i,j,k]= phantom operator")

#[ Operators on TensorSiteProxy - generate marker types for codegen ]#

when UseOpenMP:
  # OpenMP backend: operators return populated result types for direct execution
  
  proc `*`*[L, T](a, b: TensorSiteProxy[L, T]): MatMulResult[L, T] {.inline.} =
    ## Matrix multiplication - returns result type with proxy info
    result.proxyA = a
    result.proxyB = b
  
  proc `+`*[L, T](a, b: TensorSiteProxy[L, T]): MatAddResult[L, T] {.inline.} =
    ## Matrix/tensor addition
    result.proxyA = a
    result.proxyB = b
    result.isSubtraction = false
  
  proc `-`*[L, T](a, b: TensorSiteProxy[L, T]): MatAddResult[L, T] {.inline.} =
    ## Matrix/tensor subtraction
    result.proxyA = a
    result.proxyB = b
    result.isSubtraction = true
  
  proc `*`*[L, T](scalar: T, proxy: TensorSiteProxy[L, T]): ScalarMulResult[L, T] {.inline.} =
    result.proxy = proxy
    result.scalar = scalar
  
  proc `*`*[L, T](proxy: TensorSiteProxy[L, T], scalar: T): ScalarMulResult[L, T] {.inline.} =
    result.proxy = proxy
    result.scalar = scalar
  
  proc `+`*[L, T](scalar: T, proxy: TensorSiteProxy[L, T]): ScalarAddResult[L, T] {.inline.} =
    result.proxy = proxy
    result.scalar = scalar
  
  proc `+`*[L, T](proxy: TensorSiteProxy[L, T], scalar: T): ScalarAddResult[L, T] {.inline.} =
    result.proxy = proxy
    result.scalar = scalar

  # Helper to compute matrix multiply and store result
  proc computeMatMul*[L, T](m: var MatMulResult[L, T]) =
    ## Compute the matrix multiply and store in computed buffer
    let elemsPerSite = m.proxyA.elemsPerSite
    
    let rows = m.proxyA.shape[0]
    let cols = if m.proxyB.shape.len > 1: m.proxyB.shape[1] else: 1
    let innerDim = if m.proxyA.shape.len > 1: m.proxyA.shape[1] else: 1
    
    m.computed = newSeq[T](elemsPerSite)
    for i in 0..<rows:
      for j in 0..<cols:
        var sum: T = T(0)
        for k in 0..<innerDim:
          let aElemIdx = i * innerDim + k
          let bElemIdx = k * cols + j
          let srcAData = cast[ptr UncheckedArray[T]](m.proxyA.hostPtr)
          let srcBData = cast[ptr UncheckedArray[T]](m.proxyB.hostPtr)
          sum = sum + srcAData[m.proxyA.proxyElemOffset(aElemIdx)] * srcBData[m.proxyB.proxyElemOffset(bElemIdx)]
        m.computed[i * cols + j] = sum
    m.hasComputed = true

  # Chaining operators for OpenMP
  proc `+`*[L, T](a, b: MatMulResult[L, T]): MatAddResult[L, T] {.inline.} =
    ## A*B + C*D - compute both multiplications and store for later addition
    var ma = a
    var mb = b
    computeMatMul(ma)
    computeMatMul(mb)
    result.proxyA = a.proxyA  # Keep shape info
    result.proxyB = b.proxyA
    result.isSubtraction = false
    result.computedA = ma.computed
    result.computedB = mb.computed
    result.hasComputedA = true
    result.hasComputedB = true

  proc `-`*[L, T](a, b: MatMulResult[L, T]): MatAddResult[L, T] {.inline.} =
    ## A*B - C*D - compute both multiplications and store for later subtraction
    var ma = a
    var mb = b
    computeMatMul(ma)
    computeMatMul(mb)
    result.proxyA = a.proxyA  # Keep shape info
    result.proxyB = b.proxyA
    result.isSubtraction = true
    result.computedA = ma.computed
    result.computedB = mb.computed
    result.hasComputedA = true
    result.hasComputedB = true

  proc `-`*[L, T](a: MatAddResult[L, T], b: TensorSiteProxy[L, T]): MatAddResult[L, T] {.inline.} =
    ## (A*B + C*D) - E[n] - first compute A+B, then subtract E
    result.proxyA = a.proxyA  # Keep shape info
    result.proxyB = b
    result.isSubtraction = true
    result.hasComputedB = false  # B (E) comes from proxy, not computed
    
    # If both sides of previous result were computed, combine them first
    if a.hasComputedA and a.hasComputedB:
      let n = a.computedA.len
      result.computedA = newSeq[T](n)
      if a.isSubtraction:
        for i in 0..<n:
          result.computedA[i] = a.computedA[i] - a.computedB[i]
      else:
        for i in 0..<n:
          result.computedA[i] = a.computedA[i] + a.computedB[i]
      result.hasComputedA = true
    elif a.hasComputedA:
      result.computedA = a.computedA
      result.hasComputedA = true
    else:
      result.hasComputedA = false

else:
  # OpenCL/SYCL backend: phantom operators for kernel codegen
  
  # Matrix/tensor multiplication: mat1[n] * mat2[n] or mat[n] * vec[n]
  proc `*`*[L, T](a, b: TensorSiteProxy[L, T]): MatMulResult[L, T] =
    raise newException(Defect, "TensorSiteProxy * phantom operator")

  # Matrix/tensor addition: mat1[n] + mat2[n]
  proc `+`*[L, T](a, b: TensorSiteProxy[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "TensorSiteProxy + phantom operator")

  # Matrix/tensor subtraction: mat1[n] - mat2[n]
  proc `-`*[L, T](a, b: TensorSiteProxy[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "TensorSiteProxy - phantom operator")

  # Scalar * tensor: 2.0 * mat[n]
  proc `*`*[L, T](scalar: T, proxy: TensorSiteProxy[L, T]): ScalarMulResult[L, T] =
    raise newException(Defect, "scalar * TensorSiteProxy phantom operator")

  # Tensor * scalar: mat[n] * 2.0
  proc `*`*[L, T](proxy: TensorSiteProxy[L, T], scalar: T): ScalarMulResult[L, T] =
    raise newException(Defect, "TensorSiteProxy * scalar phantom operator")

  # Scalar + tensor: 2.0 + mat[n]
  proc `+`*[L, T](scalar: T, proxy: TensorSiteProxy[L, T]): ScalarAddResult[L, T] =
    raise newException(Defect, "scalar + TensorSiteProxy phantom operator")

  # Tensor + scalar: mat[n] + 2.0
  proc `+`*[L, T](proxy: TensorSiteProxy[L, T], scalar: T): ScalarAddResult[L, T] =
    raise newException(Defect, "TensorSiteProxy + scalar phantom operator")

  #[ Operators for chaining result types - enables long expressions like A*B + C*D - E ]#
  # These are phantom operators for OpenCL/SYCL kernel codegen

  # MatMulResult + MatMulResult: A*B + C*D
  proc `+`*[L, T](a, b: MatMulResult[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "MatMulResult + MatMulResult phantom operator")

  # MatMulResult + TensorSiteProxy: A*B + C[n]
  proc `+`*[L, T](a: MatMulResult[L, T], b: TensorSiteProxy[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "MatMulResult + TensorSiteProxy phantom operator")

  # TensorSiteProxy + MatMulResult: A[n] + B*C
  proc `+`*[L, T](a: TensorSiteProxy[L, T], b: MatMulResult[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "TensorSiteProxy + MatMulResult phantom operator")

  # MatMulResult - MatMulResult: A*B - C*D
  proc `-`*[L, T](a, b: MatMulResult[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "MatMulResult - MatMulResult phantom operator")

  # MatMulResult - TensorSiteProxy: A*B - C[n]
  proc `-`*[L, T](a: MatMulResult[L, T], b: TensorSiteProxy[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "MatMulResult - TensorSiteProxy phantom operator")

  # TensorSiteProxy - MatMulResult: A[n] - B*C
  proc `-`*[L, T](a: TensorSiteProxy[L, T], b: MatMulResult[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "TensorSiteProxy - MatMulResult phantom operator")

  # MatAddResult + TensorSiteProxy: (A+B) + C[n]
  proc `+`*[L, T](a: MatAddResult[L, T], b: TensorSiteProxy[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "MatAddResult + TensorSiteProxy phantom operator")

  # TensorSiteProxy + MatAddResult: A[n] + (B+C)
  proc `+`*[L, T](a: TensorSiteProxy[L, T], b: MatAddResult[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "TensorSiteProxy + MatAddResult phantom operator")

  # MatAddResult - TensorSiteProxy: (A+B) - C[n]
  proc `-`*[L, T](a: MatAddResult[L, T], b: TensorSiteProxy[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "MatAddResult - TensorSiteProxy phantom operator")

  # TensorSiteProxy - MatAddResult: A[n] - (B+C)
  proc `-`*[L, T](a: TensorSiteProxy[L, T], b: MatAddResult[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "TensorSiteProxy - MatAddResult phantom operator")

  # MatAddResult + MatMulResult: (A+B) + C*D
  proc `+`*[L, T](a: MatAddResult[L, T], b: MatMulResult[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "MatAddResult + MatMulResult phantom operator")

  # MatMulResult + MatAddResult: A*B + (C+D)
  proc `+`*[L, T](a: MatMulResult[L, T], b: MatAddResult[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "MatMulResult + MatAddResult phantom operator")

  # MatAddResult - MatMulResult: (A+B) - C*D
  proc `-`*[L, T](a: MatAddResult[L, T], b: MatMulResult[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "MatAddResult - MatMulResult phantom operator")

  # MatMulResult - MatAddResult: A*B - (C+D)
  proc `-`*[L, T](a: MatMulResult[L, T], b: MatAddResult[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "MatMulResult - MatAddResult phantom operator")

  # MatAddResult + MatAddResult: (A+B) + (C+D)
  proc `+`*[L, T](a, b: MatAddResult[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "MatAddResult + MatAddResult phantom operator")

  # MatAddResult - MatAddResult: (A+B) - (C+D)
  proc `-`*[L, T](a, b: MatAddResult[L, T]): MatAddResult[L, T] =
    raise newException(Defect, "MatAddResult - MatAddResult phantom operator")

  # Scalar multiplication on result types
  proc `*`*[L, T](scalar: T, res: MatMulResult[L, T]): ScalarMulResult[L, T] =
    raise newException(Defect, "scalar * MatMulResult phantom operator")

  proc `*`*[L, T](scalar: T, res: MatAddResult[L, T]): ScalarMulResult[L, T] =
    raise newException(Defect, "scalar * MatAddResult phantom operator")

  proc `*`*[L, T](res: MatMulResult[L, T], scalar: T): ScalarMulResult[L, T] =
    raise newException(Defect, "MatMulResult * scalar phantom operator")

  proc `*`*[L, T](res: MatAddResult[L, T], scalar: T): ScalarMulResult[L, T] =
    raise newException(Defect, "MatAddResult * scalar phantom operator")

# String conversion for debugging: $view[n]
proc `$`*[L, T](proxy: TensorSiteProxy[L, T]): string =
  ## Convert site tensor to string for debugging.
  ## Works both in OpenCL kernels (via macro) and at runtime (via stored data).
  return $proxy.runtimeData

# Runtime formatting for CPU fallback
proc `$`*[T](data: RuntimeSiteData[T]): string =
  ## Format a site tensor nicely for printing
  ## Scalar: just the value
  ## Vector: column format [x, y, z]
  ## Matrix: grid format
  
  if data.rank == 0 or data.shape.len == 0:
    # Scalar
    if data.data.len > 0:
      return $data.data[0]
    return "[]"
  
  elif data.rank == 1:
    # Vector: [v0, v1, v2, ...]
    var parts: seq[string]
    for i in 0..<data.data.len:
      parts.add fmt"{data.data[i]:>10.4f}"
    return "[" & parts.join(", ") & "]"
  
  elif data.rank == 2:
    # Matrix: grid format
    let rows = data.shape[0]
    let cols = data.shape[1]
    var lines: seq[string]
    for r in 0..<rows:
      var rowParts: seq[string]
      for c in 0..<cols:
        let idx = r * cols + c
        rowParts.add fmt"{data.data[idx]:>10.4f}"
      if r == 0:
        lines.add "⎡" & rowParts.join("  ") & " ⎤"
      elif r == rows - 1:
        lines.add "⎣" & rowParts.join("  ") & " ⎦"
      else:
        lines.add "⎢" & rowParts.join("  ") & " ⎥"
    return lines.join("\n")
  
  else:
    # Higher rank: just show flat data with shape
    return "Tensor" & $data.shape & ": " & $data.data

#[ Tests ]#

when isMainModule:
  import std/[unittest]
  
  suite "Site Tensor Types":
    test "Vec element access":
      var v: Vec[3, float64]
      v[0] = 1.0
      v[1] = 2.0
      v[2] = 3.0
      check v[0] == 1.0
      check v[1] == 2.0
      check v[2] == 3.0
    
    test "Vec addition":
      var a, b: Vec[3, float64]
      for i in 0..<3:
        a[i] = float64(i + 1)
        b[i] = float64(i + 1) * 2.0
      let c = a + b
      check c[0] == 3.0
      check c[1] == 6.0
      check c[2] == 9.0
    
    test "Vec dot product":
      var a, b: Vec[3, float64]
      a[0] = 1.0; a[1] = 2.0; a[2] = 3.0
      b[0] = 4.0; b[1] = 5.0; b[2] = 6.0
      let d = dot(a, b)
      check d == 32.0  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    
    test "Mat element access":
      var m: Mat[2, 3, float64]
      m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0
      m[1, 0] = 4.0; m[1, 1] = 5.0; m[1, 2] = 6.0
      check m[0, 0] == 1.0
      check m[0, 2] == 3.0
      check m[1, 1] == 5.0
    
    test "Mat addition":
      var a, b: Mat[2, 2, float64]
      a[0, 0] = 1.0; a[0, 1] = 2.0
      a[1, 0] = 3.0; a[1, 1] = 4.0
      b[0, 0] = 5.0; b[0, 1] = 6.0
      b[1, 0] = 7.0; b[1, 1] = 8.0
      let c = a + b
      check c[0, 0] == 6.0
      check c[0, 1] == 8.0
      check c[1, 0] == 10.0
      check c[1, 1] == 12.0
    
    test "Mat multiplication (square)":
      var a, b: Mat[2, 2, float64]
      # A = [[1, 2], [3, 4]]
      a[0, 0] = 1.0; a[0, 1] = 2.0
      a[1, 0] = 3.0; a[1, 1] = 4.0
      # B = [[5, 6], [7, 8]]
      b[0, 0] = 5.0; b[0, 1] = 6.0
      b[1, 0] = 7.0; b[1, 1] = 8.0
      # C = A * B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
      #           = [[19, 22], [43, 50]]
      let c = a * b
      check c[0, 0] == 19.0
      check c[0, 1] == 22.0
      check c[1, 0] == 43.0
      check c[1, 1] == 50.0
    
    test "Mat-Vec multiplication":
      var m: Mat[2, 3, float64]
      var v: Vec[3, float64]
      # M = [[1, 2, 3], [4, 5, 6]]
      m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0
      m[1, 0] = 4.0; m[1, 1] = 5.0; m[1, 2] = 6.0
      # v = [1, 2, 3]
      v[0] = 1.0; v[1] = 2.0; v[2] = 3.0
      # result = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
      let result = m * v
      check result[0] == 14.0
      check result[1] == 32.0
    
    test "Mat transpose":
      var m: Mat[2, 3, float64]
      m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0
      m[1, 0] = 4.0; m[1, 1] = 5.0; m[1, 2] = 6.0
      let t = transpose(m)
      check t[0, 0] == 1.0
      check t[1, 0] == 2.0
      check t[2, 0] == 3.0
      check t[0, 1] == 4.0
      check t[1, 1] == 5.0
      check t[2, 1] == 6.0
    
    test "Mat trace":
      var m: Mat[3, 3, float64]
      m[0, 0] = 1.0; m[1, 1] = 5.0; m[2, 2] = 9.0
      check trace(m) == 15.0
    
    test "Identity matrix":
      let id = identity[3, float64]()
      check id[0, 0] == 1.0
      check id[1, 1] == 1.0
      check id[2, 2] == 1.0
      check id[0, 1] == 0.0
      check id[1, 0] == 0.0

#[ ============================================================================
   LocalSiteProxy Operators for CPU-only "for all" loops
   ============================================================================ ]#

# LocalSiteProxy element access
proc `[]`*[D: static[int], R: static[int], L, T](proxy: LocalSiteProxy[D, R, L, T], i: int): T {.inline.} =
  ## Vector element read: local[n][i]
  let data = cast[ptr UncheckedArray[T]](proxy.hostPtr)
  let base = proxy.site * proxy.elemsPerSite
  data[base + i]

proc `[]`*[D: static[int], R: static[int], L, T](proxy: LocalSiteProxy[D, R, L, T], i, j: int): T {.inline.} =
  ## Matrix element read: local[n][i, j]
  let data = cast[ptr UncheckedArray[T]](proxy.hostPtr)
  let base = proxy.site * proxy.elemsPerSite
  let cols = proxy.shape[1]
  data[base + i * cols + j]

proc `[]`*[D: static[int], R: static[int], L, T](proxy: LocalSiteProxy[D, R, L, T], i, j, k: int): T {.inline.} =
  ## 3D tensor element read: local[n][i, j, k]
  let data = cast[ptr UncheckedArray[T]](proxy.hostPtr)
  let base = proxy.site * proxy.elemsPerSite
  let dim1 = proxy.shape[1]
  let dim2 = proxy.shape[2]
  data[base + i * dim1 * dim2 + j * dim2 + k]

proc `[]=`*[D: static[int], R: static[int], L, T](proxy: LocalSiteProxy[D, R, L, T], i: int, value: T) {.inline.} =
  ## Vector element write: local[n][i] = value
  let data = cast[ptr UncheckedArray[T]](proxy.hostPtr)
  let base = proxy.site * proxy.elemsPerSite
  data[base + i] = value

proc `[]=`*[D: static[int], R: static[int], L, T](proxy: LocalSiteProxy[D, R, L, T], i, j: int, value: T) {.inline.} =
  ## Matrix element write: local[n][i, j] = value
  let data = cast[ptr UncheckedArray[T]](proxy.hostPtr)
  let base = proxy.site * proxy.elemsPerSite
  let cols = proxy.shape[1]
  data[base + i * cols + j] = value

proc `[]=`*[D: static[int], R: static[int], L, T](proxy: LocalSiteProxy[D, R, L, T], i, j, k: int, value: T) {.inline.} =
  ## 3D tensor element write: local[n][i, j, k] = value
  let data = cast[ptr UncheckedArray[T]](proxy.hostPtr)
  let base = proxy.site * proxy.elemsPerSite
  let dim1 = proxy.shape[1]
  let dim2 = proxy.shape[2]
  data[base + i * dim1 * dim2 + j * dim2 + k] = value

# LocalSiteProxy arithmetic operators
proc `+`*[D: static[int], R: static[int], L, T](a, b: LocalSiteProxy[D, R, L, T]): LocalAddResult[D, R, L, T] {.inline.} =
  result.proxyA = a
  result.proxyB = b
  result.isSubtraction = false

proc `-`*[D: static[int], R: static[int], L, T](a, b: LocalSiteProxy[D, R, L, T]): LocalAddResult[D, R, L, T] {.inline.} =
  result.proxyA = a
  result.proxyB = b
  result.isSubtraction = true

proc `*`*[D: static[int], R: static[int], L, T](a, b: LocalSiteProxy[D, R, L, T]): LocalMulResult[D, R, L, T] {.inline.} =
  result.proxyA = a
  result.proxyB = b

proc `*`*[D: static[int], R: static[int], L, T](scalar: T, proxy: LocalSiteProxy[D, R, L, T]): LocalScalarMulResult[D, R, L, T] {.inline.} =
  result.proxy = proxy
  result.scalar = scalar

proc `*`*[D: static[int], R: static[int], L, T](proxy: LocalSiteProxy[D, R, L, T], scalar: T): LocalScalarMulResult[D, R, L, T] {.inline.} =
  result.proxy = proxy
  result.scalar = scalar

proc `+`*[D: static[int], R: static[int], L, T](proxy: LocalSiteProxy[D, R, L, T], scalar: T): LocalScalarAddResult[D, R, L, T] {.inline.} =
  result.proxy = proxy
  result.scalar = scalar

proc `+`*[D: static[int], R: static[int], L, T](scalar: T, proxy: LocalSiteProxy[D, R, L, T]): LocalScalarAddResult[D, R, L, T] {.inline.} =
  result.proxy = proxy
  result.scalar = scalar
# LocalSiteProxy print support
proc `$`*[D: static[int], R: static[int], L, T](proxy: LocalSiteProxy[D, R, L, T]): string =
  ## Format LocalSiteProxy tensor for printing
  ## Scalar: just the value
  ## Vector: row format [x, y, z]
  ## Matrix: grid format with Unicode box drawing
  
  let data = cast[ptr UncheckedArray[T]](proxy.hostPtr)
  let base = proxy.site * proxy.elemsPerSite
  
  when R == 0:
    # Scalar (0-rank tensor)
    return $data[base]
  
  elif R == 1:
    # Vector: [v0, v1, v2, ...]
    var parts: seq[string]
    for i in 0..<proxy.shape[0]:
      when T is SomeFloat:
        parts.add fmt"{data[base + i]:>10.4f}"
      else:
        parts.add $data[base + i]
    return "[" & parts.join(", ") & "]"
  
  elif R == 2:
    # Matrix: grid format with Unicode box drawing
    let rows = proxy.shape[0]
    let cols = proxy.shape[1]
    var lines: seq[string]
    for r in 0..<rows:
      var rowParts: seq[string]
      for c in 0..<cols:
        let idx = r * cols + c
        when T is SomeFloat:
          rowParts.add fmt"{data[base + idx]:>10.4f}"
        else:
          rowParts.add $data[base + idx]
      if r == 0:
        lines.add "⎡" & rowParts.join("  ") & " ⎤"
      elif r == rows - 1:
        lines.add "⎣" & rowParts.join("  ") & " ⎦"
      else:
        lines.add "⎢" & rowParts.join("  ") & " ⎥"
    return lines.join("\n")
  
  else:
    # Higher rank: show flat data with shape info
    var vals: seq[string]
    for i in 0..<proxy.elemsPerSite:
      when T is SomeFloat:
        vals.add fmt"{data[base + i]:>10.4f}"
      else:
        vals.add $data[base + i]
    var shapeStr: seq[string]
    for i in 0..<R:
      shapeStr.add $proxy.shape[i]
    return "Tensor[" & shapeStr.join(", ") & "]: [" & vals.join(", ") & "]"
