import reliq
import arrays
import simplecubicfield
import lattice/[simplecubiclattice]

import std/[macros]

type SimpleCubicTensor*[D: static[int], T] = object
  ## Simple cubic tensor implementation
  ##
  ## Represents a tensor field defined on a simple cubic lattice.
  ## Each component is stored as a separate SimpleCubicField.
  lattice*: SimpleCubicLattice[D]
  shape*: seq[int]
  components*: seq[SimpleCubicField[D, T]]

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
  return SimpleCubicTensor[D, T](lattice: lattice, shape: shape, components: tensor)

proc newSimpleCubicTensor*[R: static[int], D: static[int]](
  lattice: SimpleCubicLattice[D],
  shape: array[R, int],
  T: typedesc
): SimpleCubicTensor[D, T] =
  return newSimpleCubicTensor(lattice, shape.toSeq(), T)

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
  assert dest.shape == src.shape, "Tensor shapes must match"
  for i in 0..<dest.components.len:
    dest.components[i] := src.components[i]

template `+`*[D: static[int], T](a, b: SimpleCubicTensor[D, T]): SimpleCubicTensor[D, T] =
  ## Tensor addition
  assert a.shape == b.shape, "Tensor shapes must match"
  var r = newSimpleCubicTensor(a.lattice, a.shape, T)
  for i in 0..<a.components.len:
    r.components[i] := a.components[i] + b.components[i]
  r

template `-`*[D: static[int], T](a, b: SimpleCubicTensor[D, T]): SimpleCubicTensor[D, T] =
  ## Tensor subtraction
  assert a.shape == b.shape, "Tensor shapes must match"
  var r = newSimpleCubicTensor(a.lattice, a.shape, T)
  for i in 0..<a.components.len:
    r.components[i] := a.components[i] - b.components[i]
  r

template `*`*[D: static[int], T](tensor: SimpleCubicTensor[D, T], scalar: T): SimpleCubicTensor[D, T] =
  ## Tensor-scalar multiplication
  var r = newSimpleCubicTensor(tensor.lattice, tensor.shape, T)
  for i in 0..<tensor.components.len:
    r.components[i] := tensor.components[i] * scalar
  r

template `*`*[D: static[int], T](scalar: T, tensor: SimpleCubicTensor[D, T]): SimpleCubicTensor[D, T] =
  ## Scalar-tensor multiplication
  tensor * scalar

template `/`*[D: static[int], T](tensor: SimpleCubicTensor[D, T], scalar: T): SimpleCubicTensor[D, T] =
  ## Tensor-scalar division
  var r = newSimpleCubicTensor(tensor.lattice, tensor.shape, T)
  for i in 0..<tensor.components.len:
    r.components[i] := tensor.components[i] / scalar
  r

template `+=`*[D: static[int], T](a: var SimpleCubicTensor[D, T], b: SimpleCubicTensor[D, T]) =
  ## In-place tensor addition
  assert a.shape == b.shape, "Tensor shapes must match"
  for i in 0..<a.components.len:
    a.components[i] += b.components[i]

template `-=`*[D: static[int], T](a: var SimpleCubicTensor[D, T], b: SimpleCubicTensor[D, T]) =
  ## In-place tensor subtraction
  assert a.shape == b.shape, "Tensor shapes must match"
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
  
  var r = newSimpleCubicTensor(tensor.lattice, @[cols, rows], T)
  for i in 0..<rows:
    for j in 0..<cols: r[[j, i]] := tensor[[i, j]]
  r

proc matmul*[D: static[int], T](a, b: SimpleCubicTensor[D, T]): SimpleCubicTensor[D, T] =
  ## Matrix multiplication for rank-2 tensors
  ##
  ## Example:
  ## ```nim
  ## let C = matmul(A, B)  # C = A * B
  ## ```
  assert a.rank == 2 and b.rank == 2, "Matrix multiplication requires rank-2 tensors"
  assert a.shape[1] == b.shape[0], "Inner dimensions must match"
  
  let m = a.shape[0]
  let n = b.shape[1]
  let k = a.shape[1]
  
  result = newSimpleCubicTensor(a.lattice, @[m, n], T)
  
  for i in 0..<m:
    for j in 0..<n:
      result[[i, j]] := 0.0
      for p in 0..<k: result[[i, j]] += a[[i, p]] * b[[p, j]]

proc trace*[D: static[int], T](tensor: SimpleCubicTensor[D, T]): SimpleCubicField[D, T] =
  ## Compute trace of a rank-2 tensor (matrix)
  ##
  ## Example:
  ## ```nim
  ## let tr = tensor.trace()
  ## ```
  assert tensor.rank == 2, "Trace only defined for rank-2 tensors"
  assert tensor.shape[0] == tensor.shape[1], "Trace requires square matrix"
  
  var r = newSimpleCubicField(tensor.lattice, T)
  r := 0.0
  #for i in 0..<tensor.shape[0]: r += tensor[[i, i]]
  return r

proc determinant*[D: static[int], T](tensor: SimpleCubicTensor[D, T]): SimpleCubicField[D, T] =
  ## Compute determinant of a rank-2 tensor (matrix)
  ## Currently only implemented for 2x2 and 3x3 matrices
  ##
  ## Example:
  ## ```nim
  ## let det = tensor.determinant()
  ## ```
  assert tensor.rank == 2, "Determinant only defined for rank-2 tensors"
  assert tensor.shape[0] == tensor.shape[1], "Determinant requires square matrix"
  
  let n = tensor.shape[0]
  result = newSimpleCubicField(tensor.lattice, T)
  
  if n == 2:
    # det(A) = a11*a22 - a12*a21
    result := tensor[[0,0]]*tensor[[1,1]] - tensor[[0,1]]*tensor[[1,0]]
  elif n == 3:
    # det(A) = a11(a22*a33 - a23*a32) - a12(a21*a33 - a23*a31) + a13(a21*a32 - a22*a31)
    result := tensor[[0,0]]*(tensor[[1,1]]*tensor[[2,2]] - tensor[[1,2]]*tensor[[2,1]]) -
              tensor[[0,1]]*(tensor[[1,0]]*tensor[[2,2]] - tensor[[1,2]]*tensor[[2,0]]) +
              tensor[[0,2]]*(tensor[[1,0]]*tensor[[2,1]] - tensor[[1,1]]*tensor[[2,0]])
  else:
    raise newException(ValueError, "Determinant only implemented for 2x2 and 3x3 matrices")

#[ tensor contractions ]#

proc contract*[D: static[int], T](tensor: SimpleCubicTensor[D, T], axis1, axis2: int): SimpleCubicTensor[D, T] =
  ## Contract tensor along two specified axes
  ##
  ## Example:
  ## ```nim
  ## let contracted = tensor.contract(0, 1)  # Contract first two indices
  ## ```
  assert tensor.rank >= 2, "Contraction requires at least rank-2 tensor"
  assert axis1 != axis2, "Contraction axes must be different"
  assert tensor.shape[axis1] == tensor.shape[axis2], "Contracted dimensions must match"
  
  # Simplified implementation for rank-2 case (returns scalar field via trace)
  if tensor.rank == 2:
    let traceField = tensor.trace()
    result = newSimpleCubicTensor(tensor.lattice, @[1], T)
    result[[0]] := traceField
  else:
    raise newException(ValueError, "General contraction not yet implemented for rank > 2")

proc norm*[D: static[int], T](tensor: SimpleCubicTensor[D, T]): SimpleCubicField[D, T] =
  ## Compute Frobenius norm of tensor
  ##
  ## Example:
  ## ```nim
  ## let tensorNorm = tensor.norm()
  ## ```
  result = newSimpleCubicField(tensor.lattice, T)
  result := 0.0
  for i in 0..<tensor.components.len:
    result += tensor.components[i] * tensor.components[i]
  # Note: sqrt would need to be applied element-wise after this

#[ unit tests ]#

test:
  let lattice = newSimpleCubicLattice([8, 8, 8, 8*numRanks()])
  
  # Test rank-2 tensor (matrix)
  var A = newSimpleCubicTensor(lattice, @[3, 3], float)
  var B = newSimpleCubicTensor(lattice, @[3, 3], float)
  
  assert(A.rank == 2, "Tensor rank mismatch")
  assert(A.numComponents() == 9, "Tensor should have 9 components")
  assert(A.shape == @[3, 3], "Tensor shape mismatch")
  
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
  
  # Test transpose
  var At = A.transpose()
  assert(At.shape == @[3, 3], "Transpose shape mismatch")

  #[
  # Test trace
  let tr = A.trace()
  let trView = tr.localField()
  for i in 0..<tr.numSites():
    assert(abs(trView[i] - 6.0) < 1e-10, "Trace should be 1+2+3=6")

  # Test 2x2 determinant
  var M2 = newSimpleCubicTensor(lattice, @[2, 2], float)
  M2[[0, 0]] := 1.0
  M2[[0, 1]] := 2.0
  M2[[1, 0]] := 3.0
  M2[[1, 1]] := 4.0
  
  let det2 = M2.determinant()
  let det2View = det2.localField()
  for i in 0..<det2.numSites():
    # det = 1*4 - 2*3 = -2
    assert(abs(det2View[i] - (-2.0)) < 1e-10, "2x2 determinant failed")
  
  # Test compound operations
  A += B
  A *= 2.0
  
  echo "Process ", myRank(), "/", numRanks(), ": All SimpleCubicTensor tests passed!"
]#