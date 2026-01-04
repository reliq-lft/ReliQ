import x86wrap
import math  # For abs function

when defined(sse):
  import ssewrap
when defined(avx2):
  import avx2wrap
when defined(avx512):
  import avx512wrap

# Receive vectorWidth from compile-time define (passed via --define:"vectorWidth=N")
const vectorWidth* {.intdefine.} = 0  # Must be set via Makefile define

type SIMD[T] = object
  ## A generic SIMD type that can hold various SIMD types based on the architecture.
  ## The type parameter `T` determines the specific SIMD type used.
  x: T

#[ constructor ]#

proc newSIMD*[T](x: ptr T): auto =
  when vectorWidth == 1: SIMD[T](x: x[])
  elif (vectorWidth == 2) and defined(sse):
    when T is int32: SIMD[m128i](x: mm_loadu_si128(cast[ptr m128i](x)))
    elif T is int64: SIMD[m128i](x: mm_loadu_si128(cast[ptr m128i](x)))
    elif T is float32: SIMD[m128](x: mm_loadu_ps(cast[ptr cfloat](x)))
    elif T is float64: SIMD[m128d](x: mm_loadu_pd(cast[ptr cdouble](x)))
    else: SIMD[T](x: x[])
  elif vectorWidth == 4:
    when T is int32: 
      when defined(sse): SIMD[m128i](x: mm_loadu_si128(cast[ptr m128i](x)))
      else: SIMD[T](x: x[])
    elif T is int64: 
      when defined(avx2): SIMD[m256i](x: mm256_loadu_si256(cast[ptr m256i](x)))
      else: SIMD[T](x: x[])
    elif T is float32: 
      when defined(sse): SIMD[m128](x: mm_loadu_ps(cast[ptr cfloat](x)))
      else: SIMD[T](x: x[])
    elif T is float64: 
      when defined(avx2): SIMD[m256d](x: mm256_loadu_pd(cast[ptr cdouble](x)))
      else: SIMD[T](x: x[])
    else: SIMD[T](x: x[])
  elif vectorWidth == 8:
    when T is int32: 
      when defined(avx2): SIMD[m256i](x: mm256_loadu_si256(cast[ptr m256i](x)))
      else: SIMD[T](x: x[])
    elif T is int64: 
      when defined(avx512): SIMD[m512i](x: mm512_loadu_si512(cast[ptr m512i](x)))
      else: SIMD[T](x: x[])
    elif T is float32: 
      when defined(avx2): SIMD[m256](x: mm256_loadu_ps(cast[ptr cfloat](x)))
      else: SIMD[T](x: x[])
    elif T is float64: 
      when defined(avx512): SIMD[m512d](x: mm512_loadu_pd(cast[ptr cdouble](x)))
      elif defined(avx2): SIMD[m256d](x: mm256_loadu_pd(cast[ptr cdouble](x)))
      else: SIMD[T](x: x[])
    else: SIMD[T](x: x[])
  elif (vectorWidth == 16) and defined(avx512):
    when T is int32: SIMD[m512i](x: mm512_loadu_si512(cast[ptr m512i](x)))
    elif T is float32: SIMD[m512](x: mm512_loadu_ps(cast[ptr cfloat](x)))
    else: SIMD[T](x: x[])
  else: SIMD[T](x: x[])

#[ array extractor ]#

proc toArray*[T](simd: SIMD, dest: ptr T): void =
  ## Extract SIMD register contents to an array.
  ## The destination array must be properly sized for the vector width and data type.
  when vectorWidth == 1:
    dest[] = simd.x
  elif (vectorWidth == 2) and defined(sse):
    when T is int32: mm_storeu_si128(cast[ptr m128i](dest), simd.x)
    elif T is int64: mm_storeu_si128(cast[ptr m128i](dest), simd.x)
    elif T is float32: mm_storeu_ps(cast[ptr cfloat](dest), simd.x)
    elif T is float64: mm_storeu_pd(cast[ptr cdouble](dest), simd.x)
    else: dest[] = simd.x
  elif vectorWidth == 4:
    when T is int32: 
      when defined(sse): mm_storeu_si128(cast[ptr m128i](dest), simd.x)
      else: dest[] = simd.x
    elif T is int64: 
      when defined(avx2): mm256_storeu_si256(cast[ptr m256i](dest), simd.x)
      else: dest[] = simd.x
    elif T is float32: 
      when defined(sse): mm_storeu_ps(cast[ptr cfloat](dest), simd.x)
      else: dest[] = simd.x
    elif T is float64: 
      when defined(avx2): mm256_storeu_pd(cast[ptr cdouble](dest), simd.x)
      else: dest[] = simd.x
    else: dest[] = simd.x
  elif vectorWidth == 8:
    when T is int32: 
      when defined(avx2): mm256_storeu_si256(cast[ptr m256i](dest), simd.x)
      else: dest[] = simd.x
    elif T is int64: 
      when defined(avx512): mm512_storeu_si512(cast[pointer](dest), simd.x)
      else: dest[] = simd.x
    elif T is float32: 
      when defined(avx2): mm256_storeu_ps(cast[ptr cfloat](dest), simd.x)
      else: dest[] = simd.x
    elif T is float64: 
      when defined(avx512): mm512_storeu_pd(cast[pointer](dest), simd.x)
      elif defined(avx2): mm256_storeu_pd(cast[ptr cdouble](dest), simd.x)
      else: dest[] = simd.x
    else: dest[] = simd.x
  elif (vectorWidth == 16) and defined(avx512):
    when T is int32: mm512_storeu_si512(cast[pointer](dest), simd.x)
    elif T is float32: mm512_storeu_ps(cast[pointer](dest), simd.x)
    else: dest[] = simd.x
  else: dest[] = simd.x

proc toArray*[T](simd: SIMD[T]): auto =
  ## Extract SIMD register contents and return as an array.
  ## Array size is automatically determined by vectorWidth.
  when vectorWidth == 1:
    result = [simd.x]
  elif (vectorWidth == 2) and defined(sse):
    when T is m128i:
      var temp: array[2, int32]  # Assume int32 for m128i with vectorWidth=2
      mm_storeu_si128(cast[ptr m128i](addr temp[0]), simd.x)
      result = temp
    elif T is m128:
      var temp: array[4, float32]
      mm_storeu_ps(cast[ptr cfloat](addr temp[0]), simd.x)
      result = temp
    elif T is m128d:
      var temp: array[2, float64]
      mm_storeu_pd(cast[ptr cdouble](addr temp[0]), simd.x)
      result = temp
  elif vectorWidth == 4:
    when T is m128i:
      var temp: array[4, int32]
      mm_storeu_si128(cast[ptr m128i](addr temp[0]), simd.x)
      result = temp
    elif T is m256i:
      var temp: array[4, int64]  # Assume int64 for m256i with vectorWidth=4
      mm256_storeu_si256(cast[ptr m256i](addr temp[0]), simd.x)
      result = temp
    elif T is m128:
      var temp: array[4, float32]
      mm_storeu_ps(cast[ptr cfloat](addr temp[0]), simd.x)
      result = temp
    elif T is m256d:
      var temp: array[4, float64]
      mm256_storeu_pd(cast[ptr cdouble](addr temp[0]), simd.x)
      result = temp
  elif vectorWidth == 8:
    when T is m256i:
      var temp: array[8, int32]  # Assume int32 for m256i with vectorWidth=8
      mm256_storeu_si256(cast[ptr m256i](addr temp[0]), simd.x)
      result = temp
    elif T is m512i:
      var temp: array[8, int64]  # Assume int64 for m512i with vectorWidth=8
      mm512_storeu_si512(cast[pointer](addr temp[0]), simd.x)
      result = temp
    elif T is m256:
      var temp: array[8, float32]
      mm256_storeu_ps(cast[ptr cfloat](addr temp[0]), simd.x)
      result = temp
    elif T is m512d:
      var temp: array[8, float64]
      mm512_storeu_pd(cast[pointer](addr temp[0]), simd.x)
      result = temp
    elif T is m256d:
      var temp: array[4, float64]  # m256d holds 4 doubles
      mm256_storeu_pd(cast[ptr cdouble](addr temp[0]), simd.x)
      result = temp
  elif (vectorWidth == 16) and defined(avx512):
    when T is m512i:
      var temp: array[16, int32]
      mm512_storeu_si512(cast[pointer](addr temp[0]), simd.x)
      result = temp
    elif T is m512:
      var temp: array[16, float32]
      mm512_storeu_ps(cast[pointer](addr temp[0]), simd.x)
      result = temp

#[ scatter store ]#

proc store*[T, S](arr: ptr UncheckedArray[T], index: SomeInteger, simd: SIMD[S]): void =
  ## Store SIMD vector at arbitrary index in unchecked array.
  ## Vector elements will fill arr[index], arr[index+1], ..., arr[index+vectorWidth-1]
  when vectorWidth == 1:
    arr[index] = simd.x
  elif (vectorWidth == 2) and defined(sse):
    when T is int32 and S is m128i: mm_storeu_si128(cast[ptr m128i](addr arr[index]), simd.x)
    elif T is int64 and S is m128i: mm_storeu_si128(cast[ptr m128i](addr arr[index]), simd.x)
    elif T is float32 and S is m128: mm_storeu_ps(cast[ptr cfloat](addr arr[index]), simd.x)
    elif T is float64 and S is m128d: mm_storeu_pd(cast[ptr cdouble](addr arr[index]), simd.x)
    else: arr[index] = simd.x
  elif vectorWidth == 4:
    when T is int32 and S is m128i: 
      when defined(sse): mm_storeu_si128(cast[ptr m128i](addr arr[index]), simd.x)
      else: arr[index] = simd.x
    elif T is int64 and S is m256i: 
      when defined(avx2): mm256_storeu_si256(cast[ptr m256i](addr arr[index]), simd.x)
      else: arr[index] = simd.x
    elif T is float32 and S is m128: 
      when defined(sse): mm_storeu_ps(cast[ptr cfloat](addr arr[index]), simd.x)
      else: arr[index] = simd.x
    elif T is float64 and S is m256d: 
      when defined(avx2): mm256_storeu_pd(cast[ptr cdouble](addr arr[index]), simd.x)
      else: arr[index] = simd.x
    else: arr[index] = simd.x
  elif vectorWidth == 8:
    when T is int32 and S is m256i: 
      when defined(avx2): mm256_storeu_si256(cast[ptr m256i](addr arr[index]), simd.x)
      else: arr[index] = simd.x
    elif T is int64 and S is m512i: 
      when defined(avx512): mm512_storeu_si512(cast[pointer](addr arr[index]), simd.x)
      else: arr[index] = simd.x
    elif T is float32 and S is m256: 
      when defined(avx2): mm256_storeu_ps(cast[ptr cfloat](addr arr[index]), simd.x)
      else: arr[index] = simd.x
    elif T is float64 and S is m512d: 
      when defined(avx512): mm512_storeu_pd(cast[pointer](addr arr[index]), simd.x)
      elif defined(avx2): 
        # m512d with vectorWidth=8 stores 8 doubles, but m256d only holds 4
        # Store in two chunks
        var temp: array[8, float64]
        mm512_storeu_pd(cast[pointer](addr temp[0]), simd.x)
        for i in 0..<8: arr[index + i] = T(temp[i])
      else: arr[index] = simd.x
    elif T is float64 and S is m256d:
      when defined(avx2): mm256_storeu_pd(cast[ptr cdouble](addr arr[index]), simd.x)
      else: arr[index] = simd.x
    else: arr[index] = simd.x
  elif (vectorWidth == 16) and defined(avx512):
    when T is int32 and S is m512i: mm512_storeu_si512(cast[pointer](addr arr[index]), simd.x)
    elif T is float32 and S is m512: mm512_storeu_ps(cast[pointer](addr arr[index]), simd.x)
    else: arr[index] = simd.x
  else: arr[index] = simd.x

#[ arithmetic ]#

proc `+`*[T](a, b: SIMD[T]): SIMD[T] =
  when vectorWidth == 1: SIMD[T](x: a.x + b.x)
  else:
    when T is m128i: SIMD[T](x: mm_add_epi32(a.x, b.x))  # Assume int32 for m128i
    elif T is m256i: 
      when vectorWidth == 4: SIMD[T](x: mm256_add_epi64(a.x, b.x))  # int64 for vectorWidth=4
      else: SIMD[T](x: mm256_add_epi32(a.x, b.x))  # int32 for vectorWidth=8
    elif T is m512i:
      when vectorWidth == 8: SIMD[T](x: mm512_add_epi64(a.x, b.x))  # int64 for vectorWidth=8
      else: SIMD[T](x: mm512_add_epi32(a.x, b.x))  # int32 for vectorWidth=16
    elif T is m128: SIMD[T](x: mm_add_ps(a.x, b.x))
    elif T is m256: SIMD[T](x: mm256_add_ps(a.x, b.x))
    elif T is m512: SIMD[T](x: mm512_add_ps(a.x, b.x))
    elif T is m128d: SIMD[T](x: mm_add_pd(a.x, b.x))
    elif T is m256d: SIMD[T](x: mm256_add_pd(a.x, b.x))
    elif T is m512d: SIMD[T](x: mm512_add_pd(a.x, b.x))
    else: SIMD[T](x: a.x)  # Fallback

proc `-`*[T](a, b: SIMD[T]): SIMD[T] =
  when vectorWidth == 1: SIMD[T](x: a.x - b.x)
  else:
    when T is m128i: SIMD[T](x: mm_sub_epi32(a.x, b.x))
    elif T is m256i:
      when vectorWidth == 4: SIMD[T](x: mm256_sub_epi64(a.x, b.x))
      else: SIMD[T](x: mm256_sub_epi32(a.x, b.x))
    elif T is m512i:
      when vectorWidth == 8: SIMD[T](x: mm512_sub_epi64(a.x, b.x))
      else: SIMD[T](x: mm512_sub_epi32(a.x, b.x))
    elif T is m128: SIMD[T](x: mm_sub_ps(a.x, b.x))
    elif T is m256: SIMD[T](x: mm256_sub_ps(a.x, b.x))
    elif T is m512: SIMD[T](x: mm512_sub_ps(a.x, b.x))
    elif T is m128d: SIMD[T](x: mm_sub_pd(a.x, b.x))
    elif T is m256d: SIMD[T](x: mm256_sub_pd(a.x, b.x))
    elif T is m512d: SIMD[T](x: mm512_sub_pd(a.x, b.x))
    else: SIMD[T](x: a.x)

proc `*`*[T](a, b: SIMD[T]): SIMD[T] =
  when vectorWidth == 1: SIMD[T](x: a.x * b.x)
  else:
    when T is m128i: SIMD[T](x: mm_mullo_epi32(a.x, b.x))
    elif T is m256i:
      when vectorWidth == 4: SIMD[T](x: mm256_mullo_epi64(a.x, b.x))
      else: SIMD[T](x: mm256_mullo_epi32(a.x, b.x))
    elif T is m512i:
      when vectorWidth == 8: SIMD[T](x: mm512_mullo_epi64(a.x, b.x))
      else: SIMD[T](x: mm512_mullo_epi32(a.x, b.x))
    elif T is m128: SIMD[T](x: mm_mul_ps(a.x, b.x))
    elif T is m256: SIMD[T](x: mm256_mul_ps(a.x, b.x))
    elif T is m512: SIMD[T](x: mm512_mul_ps(a.x, b.x))
    elif T is m128d: SIMD[T](x: mm_mul_pd(a.x, b.x))
    elif T is m256d: SIMD[T](x: mm256_mul_pd(a.x, b.x))
    elif T is m512d: SIMD[T](x: mm512_mul_pd(a.x, b.x))
    else: SIMD[T](x: a.x)

proc `/`*[T](a, b: SIMD[T]): SIMD[T] =
  when vectorWidth == 1: SIMD[T](x: a.x div b.x)
  else:
    when T is m128: SIMD[T](x: mm_div_ps(a.x, b.x))
    elif T is m256: SIMD[T](x: mm256_div_ps(a.x, b.x))
    elif T is m512: SIMD[T](x: mm512_div_ps(a.x, b.x))
    elif T is m128d: SIMD[T](x: mm_div_pd(a.x, b.x))
    elif T is m256d: SIMD[T](x: mm256_div_pd(a.x, b.x))
    elif T is m512d: SIMD[T](x: mm512_div_pd(a.x, b.x))
    else: SIMD[T](x: a.x)  # Integer division not supported in SIMD

#[ tests ]#

when isMainModule:
  # Ensure arrays are large enough for the SIMD operations and properly aligned
  var testInt32: array[vectorWidth, int32]    # 16 elements to cover all SIMD sizes
  var testInt64: array[vectorWidth, int64]
  var testFloat32: array[vectorWidth, float32]
  var testFloat64: array[vectorWidth, float64]

  # Initialize arrays to prevent accessing uninitialized memory
  for i in 0..<vectorWidth:
    testInt32[i] = int32(i)
    testInt64[i] = int64(i)
    testFloat32[i] = float32(i)
    testFloat64[i] = float64(i)

  var testSIMDInt32 = newSIMD(addr testInt32[0])
  var testSIMDInt64 = newSIMD(addr testInt64[0])
  var testSIMDFloat32 = newSIMD(addr testFloat32[0])
  var testSIMDFloat64 = newSIMD(addr testFloat64[0])

  # Comprehensive arithmetic tests
  echo "Testing SIMD arithmetic operations..."
  
  # Test addition
  echo "Testing addition..."
  var addResultInt32 = testSIMDInt32 + testSIMDInt32
  var addResultInt64 = testSIMDInt64 + testSIMDInt64  
  var addResultFloat32 = testSIMDFloat32 + testSIMDFloat32
  var addResultFloat64 = testSIMDFloat64 + testSIMDFloat64
  
  var addArrayInt32 = addResultInt32.toArray()
  var addArrayInt64 = addResultInt64.toArray()
  var addArrayFloat32 = addResultFloat32.toArray()
  var addArrayFloat64 = addResultFloat64.toArray()
  
  for i in 0..<vectorWidth:
    assert addArrayInt32[i] == testInt32[i] + testInt32[i], "Addition failed for int32 at index " & $i
    assert addArrayInt64[i] == testInt64[i] + testInt64[i], "Addition failed for int64 at index " & $i
    assert abs(addArrayFloat32[i] - (testFloat32[i] + testFloat32[i])) < 1e-6, "Addition failed for float32 at index " & $i
    assert abs(addArrayFloat64[i] - (testFloat64[i] + testFloat64[i])) < 1e-12, "Addition failed for float64 at index " & $i

  # Test subtraction  
  echo "Testing subtraction..."
  var subResultInt32 = testSIMDInt32 - testSIMDInt32
  var subResultInt64 = testSIMDInt64 - testSIMDInt64
  var subResultFloat32 = testSIMDFloat32 - testSIMDFloat32
  var subResultFloat64 = testSIMDFloat64 - testSIMDFloat64
  
  var subArrayInt32 = subResultInt32.toArray()
  var subArrayInt64 = subResultInt64.toArray()
  var subArrayFloat32 = subResultFloat32.toArray()
  var subArrayFloat64 = subResultFloat64.toArray()
  
  for i in 0..<vectorWidth:
    assert subArrayInt32[i] == 0, "Subtraction failed for int32 at index " & $i & " (expected 0, got " & $subArrayInt32[i] & ")"
    assert subArrayInt64[i] == 0, "Subtraction failed for int64 at index " & $i & " (expected 0, got " & $subArrayInt64[i] & ")"
    assert abs(subArrayFloat32[i]) < 1e-6, "Subtraction failed for float32 at index " & $i & " (expected ~0, got " & $subArrayFloat32[i] & ")"
    assert abs(subArrayFloat64[i]) < 1e-12, "Subtraction failed for float64 at index " & $i & " (expected ~0, got " & $subArrayFloat64[i] & ")"

  # Test multiplication
  echo "Testing multiplication..."
  var mulResultInt32 = testSIMDInt32 * testSIMDInt32
  var mulResultInt64 = testSIMDInt64 * testSIMDInt64
  var mulResultFloat32 = testSIMDFloat32 * testSIMDFloat32
  var mulResultFloat64 = testSIMDFloat64 * testSIMDFloat64
  
  var mulArrayInt32 = mulResultInt32.toArray()
  var mulArrayInt64 = mulResultInt64.toArray()
  var mulArrayFloat32 = mulResultFloat32.toArray()
  var mulArrayFloat64 = mulResultFloat64.toArray()
  
  for i in 0..<vectorWidth:
    assert mulArrayInt32[i] == testInt32[i] * testInt32[i], "Multiplication failed for int32 at index " & $i
    assert mulArrayInt64[i] == testInt64[i] * testInt64[i], "Multiplication failed for int64 at index " & $i
    assert abs(mulArrayFloat32[i] - (testFloat32[i] * testFloat32[i])) < 1e-6, "Multiplication failed for float32 at index " & $i
    assert abs(mulArrayFloat64[i] - (testFloat64[i] * testFloat64[i])) < 1e-12, "Multiplication failed for float64 at index " & $i

  # Test division (floating point only)
  echo "Testing division..."
  var divResultFloat32 = testSIMDFloat32 / testSIMDFloat32
  var divResultFloat64 = testSIMDFloat64 / testSIMDFloat64
  
  var divArrayFloat32 = divResultFloat32.toArray()
  var divArrayFloat64 = divResultFloat64.toArray()
  
  for i in 0..<vectorWidth:
    if testFloat32[i] != 0.0:
      assert abs(divArrayFloat32[i] - 1.0) < 1e-6, "Division failed for float32 at index " & $i & " (expected ~1, got " & $divArrayFloat32[i] & ")"
    if testFloat64[i] != 0.0:
      assert abs(divArrayFloat64[i] - 1.0) < 1e-12, "Division failed for float64 at index " & $i & " (expected ~1, got " & $divArrayFloat64[i] & ")"

  # Test mixed operations and complex expressions
  echo "Testing complex expressions..."
  var test = testSIMDFloat32 + testSIMDFloat32*testSIMDFloat32 - testSIMDFloat32/testSIMDFloat32
  var back = test.toArray()
  for i in 0..<vectorWidth:
    if testFloat32[i] != 0.0:
      let expected = testFloat32[i] + testFloat32[i]*testFloat32[i] - testFloat32[i]/testFloat32[i]
      assert abs(back[i] - expected) < 1e-5, "Complex expression failed at index " & $i & " (expected " & $expected & ", got " & $back[i] & ")"

  # Test with different value patterns
  echo "Testing with special values..."
  
  # Test with zeros
  var zerosFloat32: array[vectorWidth, float32]
  var zerosInt32: array[vectorWidth, int32]
  for i in 0..<vectorWidth:
    zerosFloat32[i] = 0.0
    zerosInt32[i] = 0
  
  var simdZerosFloat32 = newSIMD(addr zerosFloat32[0])
  var simdZerosInt32 = newSIMD(addr zerosInt32[0])
  
  var zeroAddResult = simdZerosFloat32 + testSIMDFloat32
  var zeroAddArray = zeroAddResult.toArray()
  for i in 0..<vectorWidth:
    assert abs(zeroAddArray[i] - testFloat32[i]) < 1e-6, "Zero addition failed at index " & $i
  
  # Test with ones
  var onesFloat32: array[vectorWidth, float32]
  var onesInt32: array[vectorWidth, int32]
  for i in 0..<vectorWidth:
    onesFloat32[i] = 1.0
    onesInt32[i] = 1
    
  var simdOnesFloat32 = newSIMD(addr onesFloat32[0])
  var simdOnesInt32 = newSIMD(addr onesInt32[0])
  
  var onesMulResult = simdOnesFloat32 * testSIMDFloat32
  var onesMulArray = onesMulResult.toArray()
  for i in 0..<vectorWidth:
    assert abs(onesMulArray[i] - testFloat32[i]) < 1e-6, "Ones multiplication failed at index " & $i
  
  # Test with negative values
  var negativeFloat32: array[vectorWidth, float32]
  var negativeInt32: array[vectorWidth, int32]
  for i in 0..<vectorWidth:
    negativeFloat32[i] = -float32(i + 1)
    negativeInt32[i] = -(int32(i) + 1)
    
  var simdNegativeFloat32 = newSIMD(addr negativeFloat32[0])
  var simdNegativeInt32 = newSIMD(addr negativeInt32[0])
  
  var negAddResult = simdNegativeFloat32 + simdNegativeFloat32
  var negAddArray = negAddResult.toArray()
  for i in 0..<vectorWidth:
    assert abs(negAddArray[i] - (negativeFloat32[i] + negativeFloat32[i])) < 1e-6, "Negative addition failed at index " & $i

  # Test commutativity: a + b == b + a
  echo "Testing commutativity..."
  var commutAdd1 = testSIMDFloat32 + simdOnesFloat32
  var commutAdd2 = simdOnesFloat32 + testSIMDFloat32
  var commutArray1 = commutAdd1.toArray()
  var commutArray2 = commutAdd2.toArray()
  
  for i in 0..<vectorWidth:
    assert abs(commutArray1[i] - commutArray2[i]) < 1e-6, "Commutativity failed for addition at index " & $i
  
  var commutMul1 = testSIMDFloat32 * simdOnesFloat32
  var commutMul2 = simdOnesFloat32 * testSIMDFloat32
  var commutMulArray1 = commutMul1.toArray()
  var commutMulArray2 = commutMul2.toArray()
  
  for i in 0..<vectorWidth:
    assert abs(commutMulArray1[i] - commutMulArray2[i]) < 1e-6, "Commutativity failed for multiplication at index " & $i

  # Test associativity: (a + b) + c == a + (b + c)
  echo "Testing associativity..."
  var assocLeft = (testSIMDFloat32 + simdOnesFloat32) + simdNegativeFloat32
  var assocRight = testSIMDFloat32 + (simdOnesFloat32 + simdNegativeFloat32)
  var assocLeftArray = assocLeft.toArray()
  var assocRightArray = assocRight.toArray()
  
  for i in 0..<vectorWidth:
    assert abs(assocLeftArray[i] - assocRightArray[i]) < 1e-5, "Associativity failed for addition at index " & $i

  echo "All arithmetic tests passed!"

  # Test scatter store functionality
  echo "Testing scatter store operations..."
  
  # Create larger unchecked arrays for testing
  const testSize = 32
  var largeFloat32: ptr UncheckedArray[float32] = cast[ptr UncheckedArray[float32]](allocShared(testSize * sizeof(float32)))
  var largeInt32: ptr UncheckedArray[int32] = cast[ptr UncheckedArray[int32]](allocShared(testSize * sizeof(int32)))
  var largeFloat64: ptr UncheckedArray[float64] = cast[ptr UncheckedArray[float64]](allocShared(testSize * sizeof(float64)))
  var largeInt64: ptr UncheckedArray[int64] = cast[ptr UncheckedArray[int64]](allocShared(testSize * sizeof(int64)))

  # Initialize arrays with known values
  for i in 0..<testSize:
    largeFloat32[i] = 99.0
    largeInt32[i] = 88
    largeFloat64[i] = 77.0
    largeInt64[i] = 66

  # Test store with different starting indices
  echo "Testing store with float32..."
  store(largeFloat32, 0, testSIMDFloat32)
  store(largeFloat32, 8, testSIMDFloat32)
  store(largeFloat32, 16, testSIMDFloat32)
  
  # Verify the stored values
  for i in 0..<vectorWidth:
    assert largeFloat32[i] == testFloat32[i], "Failed at index " & $i & " (position 0)"
    assert largeFloat32[8 + i] == testFloat32[i], "Failed at index " & $(8 + i) & " (position 8)"
    assert largeFloat32[16 + i] == testFloat32[i], "Failed at index " & $(16 + i) & " (position 16)"
  
  # Verify unchanged values
  for i in vectorWidth..<8:
    assert largeFloat32[i] == 99.0, "Unexpected change at index " & $i
  for i in (8 + vectorWidth)..<16:
    assert largeFloat32[i] == 99.0, "Unexpected change at index " & $i
  for i in (16 + vectorWidth)..<testSize:
    assert largeFloat32[i] == 99.0, "Unexpected change at index " & $i

  echo "Testing store with int32..."
  store(largeInt32, 2, testSIMDInt32)
  store(largeInt32, 10, testSIMDInt32)
  
  # Verify the stored values
  for i in 0..<vectorWidth:
    assert largeInt32[2 + i] == testInt32[i], "Failed at int32 index " & $(2 + i) & " (position 2)"
    assert largeInt32[10 + i] == testInt32[i], "Failed at int32 index " & $(10 + i) & " (position 10)"
    
  # Verify unchanged values  
  assert largeInt32[0] == 88 and largeInt32[1] == 88, "Unexpected change before position 2"
  for i in (2 + vectorWidth)..<10:
    assert largeInt32[i] == 88, "Unexpected change at int32 index " & $i

  echo "Testing newSIMD functionality..."
  # Test newSIMD - should load the same values we just stored
  var loadedFloat32_0 = newSIMD(addr largeFloat32[0])
  var loadedFloat32_8 = newSIMD(addr largeFloat32[8])
  var loadedInt32_2 = newSIMD(addr largeInt32[2])
  
  # Convert back to arrays for comparison
  var loadedArray0 = loadedFloat32_0.toArray()
  var loadedArray8 = loadedFloat32_8.toArray()
  var loadedArrayInt2 = loadedInt32_2.toArray()
  
  for i in 0..<vectorWidth:
    assert loadedArray0[i] == testFloat32[i], "newSIMD failed at float32 index " & $i & " (position 0)"
    assert loadedArray8[i] == testFloat32[i], "newSIMD failed at float32 index " & $i & " (position 8)"
    assert loadedArrayInt2[i] == testInt32[i], "newSIMD failed at int32 index " & $i & " (position 2)"

  echo "Testing round-trip operations..."
  # Test round-trip: store->load->store should preserve data
  var original = testSIMDFloat32.toArray()
  store(largeFloat32, 5, testSIMDFloat32)
  var roundTrip = newSIMD(addr largeFloat32[5])
  var roundTripArray = roundTrip.toArray()
  
  for i in 0..<vectorWidth:
    assert abs(original[i] - roundTripArray[i]) < 1e-6, "Round-trip failed at index " & $i

  echo "Testing overlapping stores..."
  # Store at overlapping positions to test edge cases
  for startPos in 0..<8:
    store(largeFloat32, startPos, testSIMDFloat32)
    var loaded = newSIMD(addr largeFloat32[startPos])
    var loadedArr = loaded.toArray()
    for i in 0..<vectorWidth:
      assert loadedArr[i] == testFloat32[i], "Overlapping store/load failed at pos " & $startPos & " idx " & $i

  # Clean up allocated memory
  deallocShared(largeFloat32)
  deallocShared(largeInt32)
  deallocShared(largeFloat64)
  deallocShared(largeInt64)
  
  echo "All scatter store tests passed!"