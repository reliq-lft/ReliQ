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
    view*: pointer  # Pointer to view (not used at runtime)
    site*: int

  TensorElementProxy*[L, T] = object
    ## Proxy type returned by TensorSiteProxy[] - represents single element at site
    ## Used for element-level read/write: mView[n][i,j] = value
    view*: pointer
    site*: int
    indices*: array[4, int]  # Up to 4D tensor element indices
    numIndices*: int

  # Operation result marker types
  MatMulResult*[L, T] = object
    ## Result of matrix-matrix multiplication: mat1[n] * mat2[n]
    proxyA*, proxyB*: TensorSiteProxy[L, T]

  MatAddResult*[L, T] = object
    ## Result of matrix addition: mat1[n] + mat2[n]
    proxyA*, proxyB*: TensorSiteProxy[L, T]

  VecAddResult*[L, T] = object
    ## Result of vector addition: vec1[n] + vec2[n]
    proxyA*, proxyB*: TensorSiteProxy[L, T]

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

#[ TensorSiteProxy operators for element access ]#

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
  ## Convert site tensor to string for debugging in OpenCL kernels.
  ## This is a phantom operator - the `each` macro converts it to printf.
  raise newException(Defect, "TensorSiteProxy $ phantom operator")

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
