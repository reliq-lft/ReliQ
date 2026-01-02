#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/field/tensorfield.nim
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

import std/[macros]
import std/[tables]
import std/[math]

import reliq
import simplecubicscalar

import utils/[complex]

type Group = enum
  gkNone,
  gkOrthogonal,
  gkSpecialOrthogonal,
  gkUnitary,
  gkSpecialUnitary

type Representation* = enum
  rkNone,
  rkFundamental,
  rkAdjoint

type SimpleCubicTensorContext* = object
  group*: Group
  representation*: Representation

let defaultCtx = SimpleCubicTensorContext(group: gkNone, representation: rkNone)

type SimpleCubicTensor*[D: static[int], T] = object
  ## Simple cubic tensor implementation
  ##
  ## Represents a tensor field defined on a simple cubic lattice.
  ## Each component is stored as a separate SimpleCubicField.
  ctx*: SimpleCubicTensorContext
  lattice*: SimpleCubicLattice[D]
  shape*: seq[int]
  components*: seq[SimpleCubicField[D, T]]

type GaugeSimpleCubicTensor*[D: static[int], T] = array[D, SimpleCubicTensor[D, T]]

#[ helpers ]#

proc product(shape: seq[int]): int =
  ## Calculate total number of tensor components
  result = 1
  for dim in shape: result *= dim

proc flatIndex(indices: openArray[int], shape: seq[int]): int =
  ## Convert multi-dimensional tensor indices to flat index (row-major)
  result = 0
  var stride = 1
  for i in countdown(indices.len - 1, 0):
    result += indices[i] * stride
    stride *= shape[i]

#[ constructor ]#

proc newSimpleCubicTensor*[D: static[int]](
  lattice: SimpleCubicLattice[D],
  shape: seq[int],
  ctx: SimpleCubicTensorContext,
  T: typedesc
): SimpleCubicTensor[D, T] =
  ## Create a new SimpleCubicTensor
  ##
  ## Parameters:
  ## - `lattice`: SimpleCubicLattice on which the tensor is defined
  ## - `shape`: Shape of the tensor (e.g., @[3, 3] for 3x3 matrix)
  ##
  ## Returns:
  ## A new SimpleCubicTensor instance
  ##
  ## Example:
  ## ```nim
  ## let lattice = newSimpleCubicLattice([8, 8, 8, 16])
  ## let tensor = newSimpleCubicTensor(lattice, @[3, 3], float)
  ## ```
  var tensor = newSeq[SimpleCubicField[D, T]](product(shape))
  for i in 0..<tensor.len: tensor[i] = newSimpleCubicField(lattice, T)
  return SimpleCubicTensor[D, T](ctx: ctx, lattice: lattice, shape: shape, components: tensor)

proc newSimpleCubicTensor*[D: static[int]](
  lattice: SimpleCubicLattice[D],
  shape: seq[int],
  T: typedesc
): SimpleCubicTensor[D, T] = lattice.newSimpleCubicTensor(shape, defaultCtx, T)

proc newSimpleCubicTensor*[R: static[int], D: static[int]](
  lattice: SimpleCubicLattice[D],
  shape: array[R, int],
  ctx: SimpleCubicTensorContext,
  T: typedesc
): SimpleCubicTensor[D, T] =
  var s = newSeq[int](R)
  for i in 0..<R: s[i] = shape[i]
  return newSimpleCubicTensor(lattice, s, ctx, T)

proc newSimpleCubicTensor*[R: static[int], D: static[int]](
  lattice: SimpleCubicLattice[D],
  shape: array[R, int],
  T: typedesc
): SimpleCubicTensor[D, T] = lattice.newSimpleCubicTensor(shape, defaultCtx, T)

proc newSimpleCubicTensor*[D: static[int]](
  lattice: SimpleCubicLattice[D],
  ctx: SimpleCubicTensorContext,
  T: typedesc
): SimpleCubicTensor[D, T] = newSimpleCubicTensor(lattice, @[1], ctx, T)

proc newSimpleCubicTensor*[D: static[int]](
  lattice: SimpleCubicLattice[D],
  T: typedesc
): SimpleCubicTensor[D, T] = newSimpleCubicTensor(lattice, @[1], defaultCtx, T)

proc newGaugeSimpleCubicTensor*[D: static[int]](
  lattice: SimpleCubicLattice[D],
  shape: seq[int],
  ctx: SimpleCubicTensorContext,
  T: typedesc
): GaugeSimpleCubicTensor[D, T] =
  ## Create a new GaugeSimpleCubicTensor
  ##
  ## Parameters:
  ## - `lattice`: SimpleCubicLattice on which the gauge tensor is defined
  ## - `shape`: Shape of the tensor components
  ##
  ## Returns:
  ## A new GaugeSimpleCubicTensor instance
  ##
  ## Example:
  ## ```nim
  ## let lattice = newSimpleCubicLattice([8, 8, 8, 16])
  ## let gaugeSimpleCubicTensor = newGaugeSimpleCubicTensor(lattice, @[3, 3], float)
  ## ```
  for i in 0..<D: result[i] = newSimpleCubicTensor(lattice, shape, ctx, T)

#[ accessors ]#

proc rank*[D: static[int], T](tensor: SimpleCubicTensor[D, T]): int =
  ## Get the rank of the tensor
  tensor.shape.len

proc numComponents*[D: static[int], T](tensor: SimpleCubicTensor[D, T]): int =
  ## Get total number of tensor components
  tensor.components.len

proc numSites*[D: static[int], T](tensor: SimpleCubicTensor[D, T]): int =
  ## Get number of local sites
  tensor.components[0].numSites()

proc `[]`*[D: static[int], T](
  tensor: SimpleCubicTensor[D, T], 
  indices: openArray[int]
): SimpleCubicField[D, T] =
  ## Access tensor component by indices
  ##
  ## Example:
  ## ```nim
  ## let comp = tensor[[1, 2]]  # Access (1,2) component of rank-2 tensor
  ## ```
  let idx = flatIndex(indices, tensor.shape)
  tensor.components[idx]

proc `[]`*[D: static[int], T](tensor: var SimpleCubicTensor[D, T], indices: openArray[int]): var SimpleCubicField[D, T] =
  ## Access tensor component by indices (mutable)
  let idx = flatIndex(indices, tensor.shape)
  tensor.components[idx]

template `[]=`*[D: static[int], T](tensor: var SimpleCubicTensor[D, T], indices: openArray[int], value: SimpleCubicField[D, T]) {.dirty.} =
  ## Set tensor component by indices
  block:
    let idx = flatIndex(indices, tensor.shape)
    tensor.components[idx] := value

template `[]=`*[D: static[int], T](tensor: var SimpleCubicTensor[D, T], indices: openArray[int], value: T) {.dirty.} =
  ## Set tensor component to a scalar value (all sites)
  block:
    let idx = flatIndex(indices, tensor.shape)
    tensor.components[idx] := value

#[ unit tests ]#

template `:=`*[D: static[int], T](tensor: var SimpleCubicTensor[D, T], value: T) {.dirty.} =
  ## Set all tensor components to a scalar value
  for i in 0..<tensor.components.len:
    tensor.components[i] := value

template `:=`*[D: static[int], T](dest: var SimpleCubicTensor[D, T], src: SimpleCubicTensor[D, T]) {.dirty.} =
  ## Copy tensor values
  assert dest.shape == src.shape, "SimpleCubicTensor shapes must match"
  for i in 0..<dest.components.len:
    dest.components[i] := src.components[i]

template `+`*[D: static[int], T](a, b: SimpleCubicTensor[D, T]): SimpleCubicTensor[D, T] =
  ## SimpleCubicTensor addition
  assert a.shape == b.shape, "SimpleCubicTensor shapes must match"
  var r = newSimpleCubicTensor(a.lattice, a.shape): T
  for i in 0..<a.components.len:
    r.components[i] := a.components[i] + b.components[i]
  r

template `-`*[D: static[int], T](a, b: SimpleCubicTensor[D, T]): SimpleCubicTensor[D, T] =
  ## SimpleCubicTensor subtraction
  assert a.shape == b.shape, "SimpleCubicTensor shapes must match"
  var r = newSimpleCubicTensor(a.lattice, a.shape): T
  for i in 0..<a.components.len:
    r.components[i] := a.components[i] - b.components[i]
  r

template `*`*[D: static[int], T](tensor: SimpleCubicTensor[D, T], scalar: T): SimpleCubicTensor[D, T] =
  ## SimpleCubicTensor-scalar multiplication
  var r = newSimpleCubicTensor(tensor.lattice, tensor.shape): T
  for i in 0..<tensor.components.len:
    r.components[i] := tensor.components[i] * scalar
  r

template `*`*[D: static[int], T](scalar: T, tensor: SimpleCubicTensor[D, T]): SimpleCubicTensor[D, T] =
  ## Scalar-tensor multiplication
  tensor * scalar

template `/`*[D: static[int], T](tensor: SimpleCubicTensor[D, T], scalar: T): SimpleCubicTensor[D, T] =
  ## SimpleCubicTensor-scalar division
  var r = newSimpleCubicTensor(tensor.lattice, tensor.shape): T
  for i in 0..<tensor.components.len:
    r.components[i] := tensor.components[i] / scalar
  r

template `+=`*[D: static[int], T](a: var SimpleCubicTensor[D, T], b: SimpleCubicTensor[D, T]) =
  ## In-place tensor addition
  assert a.shape == b.shape, "SimpleCubicTensor shapes must match"
  for i in 0..<a.components.len:
    a.components[i] += b.components[i]

template `-=`*[D: static[int], T](a: var SimpleCubicTensor[D, T], b: SimpleCubicTensor[D, T]) =
  ## In-place tensor subtraction
  assert a.shape == b.shape, "SimpleCubicTensor shapes must match"
  for i in 0..<a.components.len:
    a.components[i] -= b.components[i]

template `*=`*[D: static[int], T](tensor: var SimpleCubicTensor[D, T], scalar: T) =
  ## In-place tensor-scalar multiplication
  for i in 0..<tensor.components.len:
    tensor.components[i] *= scalar

template `/=`*[D: static[int], T](tensor: var SimpleCubicTensor[D, T], scalar: T) =
  ## In-place tensor-scalar division
  for i in 0..<tensor.components.len:
    tensor.components[i] /= scalar

#[ matrix operations ]#

template transpose*[D: static[int], T](tensor: SimpleCubicTensor[D, T]): SimpleCubicTensor[D, T] =
  ## Transpose a rank-2 tensor (matrix)
  ##
  ## Example:
  ## ```nim
  ## let A = newSimpleCubicTensor(lattice, @[3, 3], float)
  ## let At = A.transpose()
  ## ```
  assert tensor.rank == 2, "Transpose only defined for rank-2 tensors"
  let rows = tensor.shape[0]
  let cols = tensor.shape[1]
  
  var r = newSimpleCubicTensor(tensor.lattice, @[cols, rows]): T
  for i in 0..<rows:
    for j in 0..<cols: r[[j, i]] := tensor[[i, j]]
  r

template complexTranspose*[D: static[int], T](tensor: SimpleCubicTensor[D, T]): SimpleCubicTensor[D, T] =
  ## Complex conjugate transpose (Hermitian) of a rank-2 tensor (matrix)
  ##
  ## Example:
  ## ```nim
  ## let A = newSimpleCubicTensor(lattice, @[3, 3], Complex64)
  ## let At = A.complexTranspose()
  ## ```
  assert tensor.rank == 2, "Complex transpose only defined for rank-2 tensors"
  let rows = tensor.shape[0]
  let cols = tensor.shape[1]
  
  var r = tensor.transpose()
  for i in 0..<rows:
    for j in 0..<cols: 
      let conj = tensor[[i, j]].adj
      r[[j, i]] := conj
  r

template matmul*[D: static[int], T](a, b: SimpleCubicTensor[D, T]): auto =
  ## Generalized matrix multiplication
  ## - Matrix × Matrix: (m×k) × (k×n) -> (m×n) tensor
  ## - Matrix × Vector: (m×k) × (k×1) -> (m×1) tensor
  ## - Vector × Matrix: (1×k) × (k×n) -> (1×n) tensor
  ##
  ## Example:
  ## ```nim
  ## let C = matmul(A, B)  # Matrix-matrix
  ## let v = matmul(A, x)  # Matrix-vector
  ## ```
  assert a.rank == 2 and b.rank == 2, "Matrix multiplication requires rank-2 tensors"
  assert a.shape[1] == b.shape[0], "Inner dimensions must match"
  
  let m = a.shape[0]
  let n = b.shape[1]
  let k = a.shape[1]
  
  var result = newSimpleCubicTensor(a.lattice, @[m, n]): T
  
  for i in 0..<m:
    for j in 0..<n:
      result[[i, j]] := 0.0
      for p in 0..<k: result[[i, j]] += a[[i, p]] * b[[p, j]]

  result

template outerProduct*[D: static[int], T](a, b: SimpleCubicTensor[D, T]): SimpleCubicTensor[D, T] =
  ## Outer product (tensor product) of two vectors
  ## Takes two rank-1 tensors (vectors) and produces a rank-2 tensor (matrix)
  ## For vectors u (m×1) and v (n×1), returns matrix M (m×n) where M[i,j] = u[i] * v[j]
  ##
  ## Example:
  ## ```nim
  ## let u = newSimpleCubicTensor(lattice, @[3, 1]): float  # 3D vector
  ## let v = newSimpleCubicTensor(lattice, @[4, 1]): float  # 4D vector
  ## let M = outerProduct(u, v)  # 3×4 matrix
  ## ```
  assert a.rank == 2 and b.rank == 2, "Outer product requires rank-2 tensors (vectors as nx1)"
  assert a.shape[1] == 1 and b.shape[1] == 1, "Outer product requires column vectors (shape [n,1])"
  
  let m = a.shape[0]
  let n = b.shape[0]
  
  var result = newSimpleCubicTensor(a.lattice, @[m, n]): T
  
  for i in 0..<m:
    for j in 0..<n: result[[i, j]] := a[[i, 0]] * b[[j, 0]]
  
  result

template `*`*[D: static[int], T](a, b: SimpleCubicTensor[D, T]): SimpleCubicTensor[D, T] =
  ## Element-wise tensor multiplication
  ## 
  ## Example:
  ## ```nim
  ## let C = A * B  # Element-wise multiplication
  ## ```
  matmul(a, b)

template trace*[D: static[int], T](tensor: SimpleCubicTensor[D, T]): SimpleCubicField[D, T] =
  ## Compute trace of a rank-2 tensor (matrix)
  ##
  ## Example:
  ## ```nim
  ## let tr = tensor.trace()
  ## ```
  assert tensor.rank == 2, "Trace only defined for rank-2 tensors"
  assert tensor.shape[0] == tensor.shape[1], "Trace requires square matrix"
  
  var r = newSimpleCubicField(tensor.lattice): T
  r := 0.0
  for i in 0..<tensor.shape[0]: r += tensor[[i, i]]
  r

template determinant*[D: static[int], T](tensor: SimpleCubicTensor[D, T]): SimpleCubicField[D, T] =
  ## Compute determinant of a rank-2 tensor (matrix)
  ## Currently implemented for 1x1, 2x2, 3x3, and 4x4 matrices
  ##
  ## Example:
  ## ```nim
  ## let det = tensor.determinant()
  ## ```
  assert tensor.rank == 2, "Determinant only defined for rank-2 tensors"
  assert tensor.shape[0] == tensor.shape[1], "Determinant requires square matrix"
  
  let n = tensor.shape[0]
  var r = newSimpleCubicField(tensor.lattice): T
  
  if n == 1: r := tensor[[0,0]]
  elif n == 2: r := tensor[[0,0]]*tensor[[1,1]] - tensor[[0,1]]*tensor[[1,0]]
  elif n == 3:
    # det(A) = a11(a22*a33 - a23*a32) - a12(a21*a33 - a23*a31) + a13(a21*a32 - a22*a31)
    r := tensor[[0,0]]*(tensor[[1,1]]*tensor[[2,2]] - tensor[[1,2]]*tensor[[2,1]]) -
         tensor[[0,1]]*(tensor[[1,0]]*tensor[[2,2]] - tensor[[1,2]]*tensor[[2,0]]) +
         tensor[[0,2]]*(tensor[[1,0]]*tensor[[2,1]] - tensor[[1,1]]*tensor[[2,0]])
  elif n == 4:
    # det(A) using cofactor expansion along first row
    # det(A) = a11*C11 - a12*C12 + a13*C13 - a14*C14
    # where Cij are the cofactors (minors with alternating signs)
    r := tensor[[0,0]]*(tensor[[1,1]]*(tensor[[2,2]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,2]]) -
                        tensor[[1,2]]*(tensor[[2,1]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,1]]) +
                        tensor[[1,3]]*(tensor[[2,1]]*tensor[[3,2]] - tensor[[2,2]]*tensor[[3,1]])) -
         tensor[[0,1]]*(tensor[[1,0]]*(tensor[[2,2]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,2]]) -
                        tensor[[1,2]]*(tensor[[2,0]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,0]]) +
                        tensor[[1,3]]*(tensor[[2,0]]*tensor[[3,2]] - tensor[[2,2]]*tensor[[3,0]])) +
         tensor[[0,2]]*(tensor[[1,0]]*(tensor[[2,1]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,1]]) -
                        tensor[[1,1]]*(tensor[[2,0]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,0]]) +
                        tensor[[1,3]]*(tensor[[2,0]]*tensor[[3,1]] - tensor[[2,1]]*tensor[[3,0]])) -
         tensor[[0,3]]*(tensor[[1,0]]*(tensor[[2,1]]*tensor[[3,2]] - tensor[[2,2]]*tensor[[3,1]]) -
                        tensor[[1,1]]*(tensor[[2,0]]*tensor[[3,2]] - tensor[[2,2]]*tensor[[3,0]]) +
                        tensor[[1,2]]*(tensor[[2,0]]*tensor[[3,1]] - tensor[[2,1]]*tensor[[3,0]]))
  else:
    raise newException(ValueError, "Determinant only implemented for 1x1, 2x2, 3x3, and 4x4 matrices")
  r

template adjugate*[D: static[int], T](tensor: SimpleCubicTensor[D, T]): SimpleCubicTensor[D, T] =
  ## Compute adjugate (adjoint) matrix of a rank-2 tensor
  ## The adjugate is the transpose of the cofactor matrix
  ## Currently implemented for 2x2, 3x3, and 4x4 matrices
  ##
  ## Example:
  ## ```nim
  ## let adj = tensor.adjugate()
  ## ```
  assert tensor.rank == 2, "Adjugate only defined for rank-2 tensors"
  assert tensor.shape[0] == tensor.shape[1], "Adjugate requires square matrix"
  
  let n = tensor.shape[0]
  var adj = newSimpleCubicTensor(tensor.lattice, tensor.shape): T
  
  if n == 1:
    adj[[0,0]] := 1.0
  elif n == 2:
    # For 2x2 matrix [[a,b],[c,d]], adjugate is [[d,-b],[-c,a]]
    adj[[0,0]] := tensor[[1,1]]
    adj[[0,1]] := -tensor[[0,1]]
    adj[[1,0]] := -tensor[[1,0]]
    adj[[1,1]] := tensor[[0,0]]
  elif n == 3:
    # Cofactor matrix elements (with checkerboard sign pattern)
    adj[[0,0]] := tensor[[1,1]]*tensor[[2,2]] - tensor[[1,2]]*tensor[[2,1]]
    adj[[0,1]] := -(tensor[[1,0]]*tensor[[2,2]] - tensor[[1,2]]*tensor[[2,0]])
    adj[[0,2]] := tensor[[1,0]]*tensor[[2,1]] - tensor[[1,1]]*tensor[[2,0]]
    
    adj[[1,0]] := -(tensor[[0,1]]*tensor[[2,2]] - tensor[[0,2]]*tensor[[2,1]])
    adj[[1,1]] := tensor[[0,0]]*tensor[[2,2]] - tensor[[0,2]]*tensor[[2,0]]
    adj[[1,2]] := -(tensor[[0,0]]*tensor[[2,1]] - tensor[[0,1]]*tensor[[2,0]])
    
    adj[[2,0]] := tensor[[0,1]]*tensor[[1,2]] - tensor[[0,2]]*tensor[[1,1]]
    adj[[2,1]] := -(tensor[[0,0]]*tensor[[1,2]] - tensor[[0,2]]*tensor[[1,0]])
    adj[[2,2]] := tensor[[0,0]]*tensor[[1,1]] - tensor[[0,1]]*tensor[[1,0]]
    
    # Transpose to get adjugate
    adj = adj.transpose()
  elif n == 4:
    # Compute cofactor matrix using 3x3 determinants
    # Cofactor C[i,j] = (-1)^(i+j) * det(M[i,j]) where M[i,j] is minor
    
    # Row 0
    adj[[0,0]] := tensor[[1,1]]*(tensor[[2,2]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,2]]) -
                  tensor[[1,2]]*(tensor[[2,1]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,1]]) +
                  tensor[[1,3]]*(tensor[[2,1]]*tensor[[3,2]] - tensor[[2,2]]*tensor[[3,1]])
    adj[[0,1]] := -(tensor[[1,0]]*(tensor[[2,2]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,2]]) -
                    tensor[[1,2]]*(tensor[[2,0]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,0]]) +
                    tensor[[1,3]]*(tensor[[2,0]]*tensor[[3,2]] - tensor[[2,2]]*tensor[[3,0]]))
    adj[[0,2]] := tensor[[1,0]]*(tensor[[2,1]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,1]]) -
                  tensor[[1,1]]*(tensor[[2,0]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,0]]) +
                  tensor[[1,3]]*(tensor[[2,0]]*tensor[[3,1]] - tensor[[2,1]]*tensor[[3,0]])
    adj[[0,3]] := -(tensor[[1,0]]*(tensor[[2,1]]*tensor[[3,2]] - tensor[[2,2]]*tensor[[3,1]]) -
                    tensor[[1,1]]*(tensor[[2,0]]*tensor[[3,2]] - tensor[[2,2]]*tensor[[3,0]]) +
                    tensor[[1,2]]*(tensor[[2,0]]*tensor[[3,1]] - tensor[[2,1]]*tensor[[3,0]]))
    
    # Row 1
    adj[[1,0]] := -(tensor[[0,1]]*(tensor[[2,2]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,2]]) -
                    tensor[[0,2]]*(tensor[[2,1]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,1]]) +
                    tensor[[0,3]]*(tensor[[2,1]]*tensor[[3,2]] - tensor[[2,2]]*tensor[[3,1]]))
    adj[[1,1]] := tensor[[0,0]]*(tensor[[2,2]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,2]]) -
                  tensor[[0,2]]*(tensor[[2,0]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,0]]) +
                  tensor[[0,3]]*(tensor[[2,0]]*tensor[[3,2]] - tensor[[2,2]]*tensor[[3,0]])
    adj[[1,2]] := -(tensor[[0,0]]*(tensor[[2,1]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,1]]) -
                    tensor[[0,1]]*(tensor[[2,0]]*tensor[[3,3]] - tensor[[2,3]]*tensor[[3,0]]) +
                    tensor[[0,3]]*(tensor[[2,0]]*tensor[[3,1]] - tensor[[2,1]]*tensor[[3,0]]))
    adj[[1,3]] := tensor[[0,0]]*(tensor[[2,1]]*tensor[[3,2]] - tensor[[2,2]]*tensor[[3,1]]) -
                  tensor[[0,1]]*(tensor[[2,0]]*tensor[[3,2]] - tensor[[2,2]]*tensor[[3,0]]) +
                  tensor[[0,2]]*(tensor[[2,0]]*tensor[[3,1]] - tensor[[2,1]]*tensor[[3,0]])
    
    # Row 2
    adj[[2,0]] := tensor[[0,1]]*(tensor[[1,2]]*tensor[[3,3]] - tensor[[1,3]]*tensor[[3,2]]) -
                  tensor[[0,2]]*(tensor[[1,1]]*tensor[[3,3]] - tensor[[1,3]]*tensor[[3,1]]) +
                  tensor[[0,3]]*(tensor[[1,1]]*tensor[[3,2]] - tensor[[1,2]]*tensor[[3,1]])
    adj[[2,1]] := -(tensor[[0,0]]*(tensor[[1,2]]*tensor[[3,3]] - tensor[[1,3]]*tensor[[3,2]]) -
                    tensor[[0,2]]*(tensor[[1,0]]*tensor[[3,3]] - tensor[[1,3]]*tensor[[3,0]]) +
                    tensor[[0,3]]*(tensor[[1,0]]*tensor[[3,2]] - tensor[[1,2]]*tensor[[3,0]]))
    adj[[2,2]] := tensor[[0,0]]*(tensor[[1,1]]*tensor[[3,3]] - tensor[[1,3]]*tensor[[3,1]]) -
                  tensor[[0,1]]*(tensor[[1,0]]*tensor[[3,3]] - tensor[[1,3]]*tensor[[3,0]]) +
                  tensor[[0,3]]*(tensor[[1,0]]*tensor[[3,1]] - tensor[[1,1]]*tensor[[3,0]])
    adj[[2,3]] := -(tensor[[0,0]]*(tensor[[1,1]]*tensor[[3,2]] - tensor[[1,2]]*tensor[[3,1]]) -
                    tensor[[0,1]]*(tensor[[1,0]]*tensor[[3,2]] - tensor[[1,2]]*tensor[[3,0]]) +
                    tensor[[0,2]]*(tensor[[1,0]]*tensor[[3,1]] - tensor[[1,1]]*tensor[[3,0]]))
    
    # Row 3
    adj[[3,0]] := -(tensor[[0,1]]*(tensor[[1,2]]*tensor[[2,3]] - tensor[[1,3]]*tensor[[2,2]]) -
                    tensor[[0,2]]*(tensor[[1,1]]*tensor[[2,3]] - tensor[[1,3]]*tensor[[2,1]]) +
                    tensor[[0,3]]*(tensor[[1,1]]*tensor[[2,2]] - tensor[[1,2]]*tensor[[2,1]]))
    adj[[3,1]] := tensor[[0,0]]*(tensor[[1,2]]*tensor[[2,3]] - tensor[[1,3]]*tensor[[2,2]]) -
                  tensor[[0,2]]*(tensor[[1,0]]*tensor[[2,3]] - tensor[[1,3]]*tensor[[2,0]]) +
                  tensor[[0,3]]*(tensor[[1,0]]*tensor[[2,2]] - tensor[[1,2]]*tensor[[2,0]])
    adj[[3,2]] := -(tensor[[0,0]]*(tensor[[1,1]]*tensor[[2,3]] - tensor[[1,3]]*tensor[[2,1]]) -
                    tensor[[0,1]]*(tensor[[1,0]]*tensor[[2,3]] - tensor[[1,3]]*tensor[[2,0]]) +
                    tensor[[0,3]]*(tensor[[1,0]]*tensor[[2,1]] - tensor[[1,1]]*tensor[[2,0]]))
    adj[[3,3]] := tensor[[0,0]]*(tensor[[1,1]]*tensor[[2,2]] - tensor[[1,2]]*tensor[[2,1]]) -
                  tensor[[0,1]]*(tensor[[1,0]]*tensor[[2,2]] - tensor[[1,2]]*tensor[[2,0]]) +
                  tensor[[0,2]]*(tensor[[1,0]]*tensor[[2,1]] - tensor[[1,1]]*tensor[[2,0]])
    
    # Transpose to get adjugate
    adj = adj.transpose()
  else:
    raise newException(ValueError, "Adjugate only implemented for 1x1, 2x2, 3x3, and 4x4 matrices")
  adj

template inverse*[D: static[int], T](tensor: SimpleCubicTensor[D, T]): SimpleCubicTensor[D, T] =
  ## Compute inverse of a rank-2 tensor (matrix)
  ## Uses the formula: A^(-1) = adj(A) / det(A)
  ## Currently implemented for 2x2, 3x3, and 4x4 matrices
  ##
  ## Example:
  ## ```nim
  ## let invA = tensor.inverse()
  ## ```
  assert tensor.rank == 2, "Inverse only defined for rank-2 tensors"
  assert tensor.shape[0] == tensor.shape[1], "Inverse requires square matrix"
  
  var inv = newSimpleCubicTensor(tensor.lattice, tensor.shape): T

  case tensor.ctx.group:
  of gkOrthogonal, gkSpecialOrthogonal:
    inv = tensor.transpose()
  of gkUnitary, gkSpecialUnitary:
    when isComplex(T): 
      inv = tensor.complexTranspose()
    else: 
      raise newException(ValueError, "complex transpose only defined for complex types")
  of gkNone:
    let det = tensor.determinant()
    let adj = tensor.adjugate()
    for i in 0..<tensor.numComponents(): 
      inv.components[i] := adj.components[i] / det
  
  inv

template frobeniusNorm2*[D: static[int]](tensor: SimpleCubicTensor[D, float]): SimpleCubicField[D, float] =
  ## Compute Frobenius norm of a real tensor field
  ## For a matrix A, ||A||_F = sqrt(sum_{i,j} |A_{ij}|²)
  ##
  ## Example:
  ## ```nim
  ## let norm = tensor.frobeniusNorm()
  ## ```
  var sumSq = newSimpleCubicField(tensor.lattice): float
  sumSq := 0.0
  for i in 0..<tensor.numComponents():
    sumSq += tensor.components[i] * tensor.components[i]
  sumSq

template frobeniusNorm2*[D: static[int]](tensor: SimpleCubicTensor[D, Complex64]): SimpleCubicField[D, float] =
  ## Compute Frobenius norm of a complex tensor field
  ## For a matrix A, ||A||_F = sqrt(sum_{i,j} |A_{ij}|²)
  ##
  ## Example:
  ## ```nim
  ## let norm = tensor.frobeniusNorm()
  ## ```
  var sumSq = newSimpleCubicField(tensor.lattice): float
  sumSq := 0.0
  for i in 0..<tensor.numComponents(): sumSq += tensor.components[i].norm2()
  sumSq

template traceNorm2*[D: static[int]](tensor: SimpleCubicTensor[D, float]): SimpleCubicField[D, float] =
  ## Compute trace norm (nuclear norm) of a real rank-2 tensor
  ## For a matrix A, ||A||_tr = sum of singular values
  ## Currently approximated as sqrt(tr(A† A)) for real matrices
  ##
  ## Example:
  ## ```nim
  ## let norm = tensor.traceNorm()
  ## ```
  assert tensor.rank == 2, "Trace norm only defined for rank-2 tensors"
  let At = tensor.transpose()
  let AtA = matmul(At, tensor)
  AtA.trace()

template traceNorm2*[D: static[int]](tensor: SimpleCubicTensor[D, Complex64]): SimpleCubicField[D, float] =
  ## Compute trace norm (nuclear norm) of a complex rank-2 tensor
  ## For a matrix A, ||A||_tr = sum of singular values
  ## Currently approximated as |tr(A† A)| for complex matrices
  ##
  ## Example:
  ## ```nim
  ## let norm = tensor.traceNorm()
  ## ```
  assert tensor.rank == 2, "Trace norm only defined for rank-2 tensors"
  let At = tensor.complexTranspose()
  let AtA = matmul(At, tensor)
  let tr = AtA.trace()
  tr.abs()

#[ misc ]#

proc tensorRemove*[D: static[int], T](tensor: SimpleCubicTensor[D, T]): SimpleCubicField[D, T] =
  ## Remove a specific index along a given axis
  ##
  ## Example:
  ## ```nim
  ## let reduced = tensor.tensorRemove(0, 1)  # Remove index 1 along axis 0
  ## ```
  assert tensor.shape == @[1], "tensorRemove only implemented for rank-1 tensors"
  return tensor.components[0]

template toPaddedSimpleCubicTensor*[D: static[int], T](
  tightSimpleCubicTensor: SimpleCubicTensor[D, T], 
  ghostGrid: array[D, int]
): SimpleCubicTensor[D, T] =
  ## Convert a tensor field to a padded tensor field with ghost zones
  ##
  ## Example:
  ## ```nim
  ## let paddedSimpleCubicTensor = tensor.toPaddedSimpleCubicTensor(@[1,1,1,1])  # Add 1 layer of ghost zones in all directions
  ## ```
  let paddedLattice = newSimpleCubicLattice(
    tightSimpleCubicTensor.lattice.dimensions,
    tightSimpleCubicTensor.lattice.mpiGrid,
    ghostGrid
  )
  var paddedSimpleCubicTensor = paddedLattice.newSimpleCubicTensor(tightSimpleCubicTensor.shape): T
  
  # Use field-level assignment for each component instead of component assignment
  for ijk in 0..<tightSimpleCubicTensor.numComponents():
    let paddedSimpleCubicField = tightSimpleCubicTensor.components[ijk].toPaddedSimpleCubicField(ghostGrid)
    paddedSimpleCubicTensor.components[ijk] := paddedSimpleCubicField
  
  paddedSimpleCubicTensor

template toTightSimpleCubicTensor*[D: static[int], T](
  paddedSimpleCubicTensor: SimpleCubicTensor[D, T]
): SimpleCubicTensor[D, T] =
  ## Convert a padded tensor field with ghost zones to a tight tensor field
  ##
  ## Example:
  ## ```nim
  ## let tightSimpleCubicTensor = paddedSimpleCubicTensor.toTightSimpleCubicTensor()
  ## ```
  let tightLattice = newSimpleCubicLattice(
    paddedSimpleCubicTensor.lattice.dimensions,
    paddedSimpleCubicTensor.lattice.mpiGrid
  )
  var tightSimpleCubicTensor = tightLattice.newSimpleCubicTensor(paddedSimpleCubicTensor.shape): T
  
  # Use field-level conversion for each component instead of direct assignment
  for ijk in 0..<paddedSimpleCubicTensor.numComponents():
    let tightSimpleCubicField = paddedSimpleCubicTensor.components[ijk].toTightSimpleCubicField()
    tightSimpleCubicTensor.components[ijk] := tightSimpleCubicField
  
  tightSimpleCubicTensor

template exchange*[D: static[int], T](tensor: SimpleCubicTensor[D, T]) =
  ## Perform halo exchange for all components of the tensor field
  ##
  ## Example:
  ## ```nim
  ## tensor.exchange()
  ## ```
  for i in 0..<tensor.numComponents():
    when isComplex(T):
      tensor.components[i].fieldRe.updateGhosts()
      tensor.components[i].fieldIm.updateGhosts()
    else: tensor.components[i].field.updateGhosts()

#[ unit tests ]#

test:
  let lattice = newSimpleCubicLattice([8, 8, 8, 8*numRanks()])
  
  # Test rank-2 tensor (matrix)
  var A = newSimpleCubicTensor(lattice, @[3, 3]): float
  var B = newSimpleCubicTensor(lattice, @[3, 3]): float
  
  assert(A.rank == 2, "SimpleCubicTensor rank mismatch")
  assert(A.numComponents() == 9, "SimpleCubicTensor should have 9 components")
  assert(A.shape == @[3, 3], "SimpleCubicTensor shape mismatch")
  
  for i in 0..<A.numComponents():
    A.components[i] := 2.0

  # Initialize tensors
  A := 2.0
  B := 3.0
  
  # Test component access
  A[[0, 0]] := 1.0
  A[[1, 1]] := 2.0
  A[[2, 2]] := 3.0

  # Test scalar operations
  var C = A * 2.0
  var D = A + B

  # matrix-matrix multiplication
  discard C*D
  
  # Test transpose
  var At = A.transpose()
  assert(At.shape == @[3, 3], "Transpose shape mismatch")

  # Test trace
  let tr = A.trace()
  let trView = tr.localSimpleCubicField()
  for i in 0..<tr.numSites():
    assert(abs(trView[i] - 6.0) < 1e-10, "Trace should be 1+2+3=6")

  # Test 2x2 determinant
  var M2 = newSimpleCubicTensor(lattice, @[2, 2]): float
  M2[[0, 0]] := 1.0
  M2[[0, 1]] := 2.0
  M2[[1, 0]] := 3.0
  M2[[1, 1]] := 4.0
  
  let det2 = M2.determinant()
  let det2View = det2.localSimpleCubicField()
  for i in 0..<det2.numSites():
    # det = 1*4 - 2*3 = -2
    assert(abs(det2View[i] - (-2.0)) < 1e-10, "2x2 determinant failed")
  
  # Test 4x4 determinant
  var M4 = newSimpleCubicTensor(lattice, @[4, 4]): float
  M4[[0, 0]] := 1.0
  M4[[0, 1]] := 0.0
  M4[[0, 2]] := 2.0
  M4[[0, 3]] := -1.0
  M4[[1, 0]] := 3.0
  M4[[1, 1]] := 0.0
  M4[[1, 2]] := 0.0
  M4[[1, 3]] := 5.0
  M4[[2, 0]] := 2.0
  M4[[2, 1]] := 1.0
  M4[[2, 2]] := 4.0
  M4[[2, 3]] := -3.0
  M4[[3, 0]] := 1.0
  M4[[3, 1]] := 0.0
  M4[[3, 2]] := 5.0
  M4[[3, 3]] := 0.0
  
  let det4 = M4.determinant()
  let det4View = det4.localSimpleCubicField()
  for i in 0..<det4.numSites():
    # det(M4) = 30
    assert(abs(det4View[i] - 30.0) < 1e-10, "4x4 determinant failed")
  
  # Test adjugate for 2x2 matrix
  var Madj2 = newSimpleCubicTensor(lattice, @[2, 2]): float
  Madj2[[0, 0]] := 4.0
  Madj2[[0, 1]] := 7.0
  Madj2[[1, 0]] := 2.0
  Madj2[[1, 1]] := 6.0
  
  let adj2 = Madj2.adjugate()
  let adj2_00_view = adj2[[0,0]].localSimpleCubicField()
  let adj2_01_view = adj2[[0,1]].localSimpleCubicField()
  let adj2_10_view = adj2[[1,0]].localSimpleCubicField()
  let adj2_11_view = adj2[[1,1]].localSimpleCubicField()
  for i in 0..<adj2[[0,0]].numSites():
    # adj([[4,7],[2,6]]) = [[6,-7],[-2,4]]
    assert(abs(adj2_00_view[i] - 6.0) < 1e-10, "Adjugate 2x2 [0,0] failed")
    assert(abs(adj2_01_view[i] - (-7.0)) < 1e-10, "Adjugate 2x2 [0,1] failed")
    assert(abs(adj2_10_view[i] - (-2.0)) < 1e-10, "Adjugate 2x2 [1,0] failed")
    assert(abs(adj2_11_view[i] - 4.0) < 1e-10, "Adjugate 2x2 [1,1] failed")
  
  # Test adjugate for 3x3 matrix
  var Madj3 = newSimpleCubicTensor(lattice, @[3, 3]): float
  Madj3[[0, 0]] := 1.0
  Madj3[[0, 1]] := 2.0
  Madj3[[0, 2]] := 3.0
  Madj3[[1, 0]] := 0.0
  Madj3[[1, 1]] := 1.0
  Madj3[[1, 2]] := 4.0
  Madj3[[2, 0]] := 5.0
  Madj3[[2, 1]] := 6.0
  Madj3[[2, 2]] := 0.0
  
  let adj3 = Madj3.adjugate()
  let adj3_00_view = adj3[[0,0]].localSimpleCubicField()
  let adj3_02_view = adj3[[0,2]].localSimpleCubicField()
  for i in 0..<adj3[[0,0]].numSites():
    # For matrix [[1,2,3],[0,1,4],[5,6,0]]:
    # Cofactor C[0,0] = +(1*0 - 4*6) = -24
    # Cofactor C[0,1] = -(0*0 - 4*5) = 20
    # Cofactor C[0,2] = +(0*6 - 1*5) = -5
    # After transpose: adj[0,0] = C[0,0] = -24, adj[0,2] = C[2,0]
    # C[2,0] = +(2*4 - 3*1) = 5
    assert(abs(adj3_00_view[i] - (-24.0)) < 1e-10, "Adjugate 3x3 [0,0] failed")
    assert(abs(adj3_02_view[i] - 5.0) < 1e-10, "Adjugate 3x3 [0,2] failed")
  
  # Test inverse for 2x2 matrix
  var Minv2 = newSimpleCubicTensor(lattice, @[2, 2]): float
  Minv2[[0, 0]] := 4.0
  Minv2[[0, 1]] := 7.0
  Minv2[[1, 0]] := 2.0
  Minv2[[1, 1]] := 6.0
  
  let inv2 = Minv2.inverse()
  let identity2 = matmul(Minv2, inv2)
  let id2_00_view = identity2[[0,0]].localSimpleCubicField()
  let id2_01_view = identity2[[0,1]].localSimpleCubicField()
  let id2_10_view = identity2[[1,0]].localSimpleCubicField()
  let id2_11_view = identity2[[1,1]].localSimpleCubicField()
  for i in 0..<identity2[[0,0]].numSites():
    # M * M^(-1) should equal identity
    assert(abs(id2_00_view[i] - 1.0) < 1e-10, "Inverse 2x2: M*M^(-1) [0,0] != 1")
    assert(abs(id2_01_view[i]) < 1e-10, "Inverse 2x2: M*M^(-1) [0,1] != 0")
    assert(abs(id2_10_view[i]) < 1e-10, "Inverse 2x2: M*M^(-1) [1,0] != 0")
    assert(abs(id2_11_view[i] - 1.0) < 1e-10, "Inverse 2x2: M*M^(-1) [1,1] != 1")
  
  # Test inverse for 3x3 matrix
  var Minv3 = newSimpleCubicTensor(lattice, @[3, 3]): float
  Minv3[[0, 0]] := 1.0
  Minv3[[0, 1]] := 2.0
  Minv3[[0, 2]] := 3.0
  Minv3[[1, 0]] := 0.0
  Minv3[[1, 1]] := 1.0
  Minv3[[1, 2]] := 4.0
  Minv3[[2, 0]] := 5.0
  Minv3[[2, 1]] := 6.0
  Minv3[[2, 2]] := 0.0
  
  let inv3 = Minv3.inverse()
  let identity3 = matmul(Minv3, inv3)
  let id3_00_view = identity3[[0,0]].localSimpleCubicField()
  let id3_01_view = identity3[[0,1]].localSimpleCubicField()
  let id3_11_view = identity3[[1,1]].localSimpleCubicField()
  let id3_22_view = identity3[[2,2]].localSimpleCubicField()
  for i in 0..<identity3[[0,0]].numSites():
    # M * M^(-1) should equal identity (check diagonal)
    assert(abs(id3_00_view[i] - 1.0) < 1e-9, "Inverse 3x3: M*M^(-1) [0,0] != 1")
    assert(abs(id3_11_view[i] - 1.0) < 1e-9, "Inverse 3x3: M*M^(-1) [1,1] != 1")
    assert(abs(id3_22_view[i] - 1.0) < 1e-9, "Inverse 3x3: M*M^(-1) [2,2] != 1")
    # Check one off-diagonal element
    assert(abs(id3_01_view[i]) < 1e-9, "Inverse 3x3: M*M^(-1) [0,1] != 0")
  
  # Test adjugate for 4x4 matrix
  var Madj4 = newSimpleCubicTensor(lattice, @[4, 4]): float
  Madj4[[0, 0]] := 1.0
  Madj4[[0, 1]] := 0.0
  Madj4[[0, 2]] := 2.0
  Madj4[[0, 3]] := -1.0
  Madj4[[1, 0]] := 3.0
  Madj4[[1, 1]] := 0.0
  Madj4[[1, 2]] := 0.0
  Madj4[[1, 3]] := 5.0
  Madj4[[2, 0]] := 2.0
  Madj4[[2, 1]] := 1.0
  Madj4[[2, 2]] := 4.0
  Madj4[[2, 3]] := -3.0
  Madj4[[3, 0]] := 1.0
  Madj4[[3, 1]] := 0.0
  Madj4[[3, 2]] := 5.0
  Madj4[[3, 3]] := 0.0
  
  let adj4 = Madj4.adjugate()
  let det_madj4 = Madj4.determinant()
  # Verify property: adj(A) * A = det(A) * I
  let verify4 = matmul(adj4, Madj4)
  let v4_00_view = verify4[[0,0]].localSimpleCubicField()
  let v4_11_view = verify4[[1,1]].localSimpleCubicField()
  let v4_01_view = verify4[[0,1]].localSimpleCubicField()
  let det_madj4_view = det_madj4.localSimpleCubicField()
  for i in 0..<verify4[[0,0]].numSites():
    # adj(A) * A = det(A) * I
    assert(abs(v4_00_view[i] - det_madj4_view[i]) < 1e-9, "Adjugate 4x4: adj*A [0,0] != det")
    assert(abs(v4_11_view[i] - det_madj4_view[i]) < 1e-9, "Adjugate 4x4: adj*A [1,1] != det")
    assert(abs(v4_01_view[i]) < 1e-9, "Adjugate 4x4: adj*A [0,1] != 0")
  
  # Test inverse for 4x4 matrix (use same matrix as determinant test)
  let inv4 = M4.inverse()
  let identity4 = matmul(M4, inv4)
  let id4_00_view = identity4[[0,0]].localSimpleCubicField()
  let id4_11_view = identity4[[1,1]].localSimpleCubicField()
  let id4_22_view = identity4[[2,2]].localSimpleCubicField()
  let id4_33_view = identity4[[3,3]].localSimpleCubicField()
  let id4_01_view = identity4[[0,1]].localSimpleCubicField()
  let id4_12_view = identity4[[1,2]].localSimpleCubicField()
  for i in 0..<identity4[[0,0]].numSites():
    # M * M^(-1) should equal identity (check diagonal)
    assert(abs(id4_00_view[i] - 1.0) < 1e-9, "Inverse 4x4: M*M^(-1) [0,0] != 1")
    assert(abs(id4_11_view[i] - 1.0) < 1e-9, "Inverse 4x4: M*M^(-1) [1,1] != 1")
    assert(abs(id4_22_view[i] - 1.0) < 1e-9, "Inverse 4x4: M*M^(-1) [2,2] != 1")
    assert(abs(id4_33_view[i] - 1.0) < 1e-9, "Inverse 4x4: M*M^(-1) [3,3] != 1")
    # Check off-diagonal elements
    assert(abs(id4_01_view[i]) < 1e-9, "Inverse 4x4: M*M^(-1) [0,1] != 0")
    assert(abs(id4_12_view[i]) < 1e-9, "Inverse 4x4: M*M^(-1) [1,2] != 0")
  
  print "Process ", myRank(), "/", numRanks(), ": All adjugate and inverse tests passed!"
  
  # Test matrix-vector multiplication
  var Mmv = newSimpleCubicTensor(lattice, @[3, 3]): float
  Mmv[[0, 0]] := 1.0
  Mmv[[0, 1]] := 2.0
  Mmv[[0, 2]] := 3.0
  Mmv[[1, 0]] := 4.0
  Mmv[[1, 1]] := 5.0
  Mmv[[1, 2]] := 6.0
  Mmv[[2, 0]] := 7.0
  Mmv[[2, 1]] := 8.0
  Mmv[[2, 2]] := 9.0
  
  var vmv = newSimpleCubicTensor(lattice, @[3, 1]): float
  vmv[[0, 0]] := 1.0
  vmv[[1, 0]] := 2.0
  vmv[[2, 0]] := 3.0
  
  let result_mv = matmul(Mmv, vmv)
  assert(result_mv.shape == @[3, 1], "Matrix-vector result shape should be [3,1]")
  let mv0_view = result_mv[[0,0]].localSimpleCubicField()
  let mv1_view = result_mv[[1,0]].localSimpleCubicField()
  let mv2_view = result_mv[[2,0]].localSimpleCubicField()
  for i in 0..<result_mv[[0,0]].numSites():
    # [1,2,3] · [1,2,3] = 1*1 + 2*2 + 3*3 = 14
    # [4,5,6] · [1,2,3] = 4*1 + 5*2 + 6*3 = 32
    # [7,8,9] · [1,2,3] = 7*1 + 8*2 + 9*3 = 50
    assert(abs(mv0_view[i] - 14.0) < 1e-10, "Matrix-vector [0] failed")
    assert(abs(mv1_view[i] - 32.0) < 1e-10, "Matrix-vector [1] failed")
    assert(abs(mv2_view[i] - 50.0) < 1e-10, "Matrix-vector [2] failed")
  
  # Test outer product
  var u = newSimpleCubicTensor(lattice, @[3, 1]): float
  u[[0, 0]] := 1.0
  u[[1, 0]] := 2.0
  u[[2, 0]] := 3.0
  
  var v = newSimpleCubicTensor(lattice, @[2, 1]): float
  v[[0, 0]] := 4.0
  v[[1, 0]] := 5.0
  
  let outer = outerProduct(u, v)
  assert(outer.shape == @[3, 2], "Outer product shape should be [3,2]")
  let o00_view = outer[[0,0]].localSimpleCubicField()
  let o01_view = outer[[0,1]].localSimpleCubicField()
  let o10_view = outer[[1,0]].localSimpleCubicField()
  let o21_view = outer[[2,1]].localSimpleCubicField()
  for i in 0..<outer[[0,0]].numSites():
    # u ⊗ v = [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]]
    assert(abs(o00_view[i] - 4.0) < 1e-10, "Outer product [0,0] failed")
    assert(abs(o01_view[i] - 5.0) < 1e-10, "Outer product [0,1] failed")
    assert(abs(o10_view[i] - 8.0) < 1e-10, "Outer product [1,0] failed")
    assert(abs(o21_view[i] - 15.0) < 1e-10, "Outer product [2,1] failed")
  
  print "Process ", myRank(), "/", numRanks(), ": All matrix-vector and outer product tests passed!"
  
  # Test compound operations
  A += B
  A *= 2.0
  
  print "Process ", myRank(), "/", numRanks(), ": All SimpleCubicTensor tests passed!"
  
  # Test complex tensor
  var CT = newSimpleCubicTensor(lattice, @[2, 2]): Complex64
  CT[[0, 0]] := complex(1.0, 0.0)
  CT[[0, 1]] := complex(0.0, 1.0)
  CT[[1, 0]] := complex(0.0, -1.0)
  CT[[1, 1]] := complex(1.0, 0.0)
  
  # Test trace of complex matrix
  let ctr = CT.trace()
  let ctrView = ctr.localSimpleCubicField()
  for i in 0..<ctr.numSites():
    let val = ctrView[i]
    assert(abs(val.re - 2.0) < 1e-10, "Complex trace real part should be 2.0")
    assert(abs(val.im) < 1e-10, "Complex trace imag part should be 0.0")
  
  print "Process ", myRank(), "/", numRanks(), ": All complex tensor tests passed!"
  
  # Test Frobenius norm for real tensor
  var FN = newSimpleCubicTensor(lattice, @[2, 2]): float
  FN[[0, 0]] := 1.0
  FN[[0, 1]] := 2.0
  FN[[1, 0]] := 3.0
  FN[[1, 1]] := 4.0
  
  let fnorm = FN.frobeniusNorm2()
  let fnormView = fnorm.localSimpleCubicField()
  for i in 0..<fnorm.numSites():
    # ||A||_F = 1² + 2² + 3² + 4² = 30
    let expected = 30.0
    assert(abs(fnormView[i] - expected) < 1e-10, "Real Frobenius norm failed")
  
  # Test Frobenius norm for complex tensor
  var CFN = newSimpleCubicTensor(lattice, @[2, 2]): Complex64
  CFN[[0, 0]] := complex(1.0, 1.0)
  CFN[[0, 1]] := complex(2.0, 0.0)
  CFN[[1, 0]] := complex(0.0, 3.0)
  CFN[[1, 1]] := complex(1.0, 2.0)
  
  let cfnorm = CFN.frobeniusNorm2()
  let cfnormView = cfnorm.localSimpleCubicField()
  for i in 0..<cfnorm.numSites():
    # ||A||_F = |1+i|² + |2|² + |3i|² + |1+2i|²
    #         = 2 + 4 + 9 + 5 = 20
    let expected = 20.0
    assert(abs(cfnormView[i] - expected) < 1e-10, "Complex Frobenius norm failed")
  
  
  # Test trace norm for real tensor
  let tnorm = M2.traceNorm2()
  let tnormView = tnorm.localSimpleCubicField()
  for i in 0..<tnorm.numSites():
    # For M2 = [[1, 2], [3, 4]], M2ᵀ M2 = [[10, 14], [14, 20]]
    # tr(M2ᵀ M2) = 30
    let expected = 30.0
    assert(abs(tnormView[i] - expected) < 1e-10, "Real trace norm failed")
  
  # Test trace norm for complex tensor
  let ctnorm = CT.traceNorm2()
  let ctnormView = ctnorm.localSimpleCubicField()
  for i in 0..<ctnorm.numSites():
    # For CT = [[1, i], [-i, 1]], CT† CT = [[1+1, i-i], [-i+i, 1+1]] = [[2, 0], [0, 2]]
    # tr(CT† CT) = 4, |4| = 4
    let expected = 4.0
    assert(abs(ctnormView[i] - expected) < 1e-10, "Complex trace norm failed")
  print "Process ", myRank(), "/", numRanks(), ": All norm tests passed!"

  # Test tensor padding conversion with a fresh tensor
  var testSimpleCubicTensor = newSimpleCubicTensor(lattice, @[2, 2]): float
  testSimpleCubicTensor[[0, 0]] := 1.0
  testSimpleCubicTensor[[0, 1]] := 2.0
  testSimpleCubicTensor[[1, 0]] := 3.0
  testSimpleCubicTensor[[1, 1]] := 4.0
  
  let ghostGrid = [1, 1, 1, 1]
  let paddedLattice = newSimpleCubicLattice(lattice.dimensions, lattice.mpiGrid, ghostGrid)
  var paddedSimpleCubicTensor = newSimpleCubicTensor(paddedLattice, testSimpleCubicTensor.shape): float
  let tightLattice = newSimpleCubicLattice(lattice.dimensions, lattice.mpiGrid)
  var tightSimpleCubicTensor = newSimpleCubicTensor(tightLattice, testSimpleCubicTensor.shape): float
  
  assert paddedSimpleCubicTensor.lattice.ghostGrid == ghostGrid, "Padded tensor ghost grid mismatch"
  
  # Test tensor assignments across different lattice configurations
  # This mirrors the scalar field test approach
  for i in 0..<testSimpleCubicTensor.numComponents():
    tightSimpleCubicTensor.components[i] := testSimpleCubicTensor.components[i]
    paddedSimpleCubicTensor.components[i] := tightSimpleCubicTensor.components[i]
  
  # Verify that assignments worked correctly
  for i in 0..<testSimpleCubicTensor.numComponents():
    let originalView = testSimpleCubicTensor.components[i].localSimpleCubicField()
    let tightView = tightSimpleCubicTensor.components[i].localSimpleCubicField()
    let paddedView = paddedSimpleCubicTensor.components[i].localSimpleCubicField()
    
    for j in 0..<originalView.numSites():
      assert abs(originalView[j] - tightView[j]) < 1e-10, "Tight tensor assignment mismatch"
      assert abs(originalView[j] - paddedView[j]) < 1e-10, "Padded tensor assignment mismatch"
      assert abs(tightView[j] - paddedView[j]) < 1e-10, "Tight-padded tensor assignment mismatch"
      
  print "Process ", myRank(), "/", numRanks(), ": SimpleCubicTensor padding conversion tests passed!"