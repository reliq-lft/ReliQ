import x86wrap

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

when isMainModule:
  # Ensure arrays are large enough for the SIMD operations and properly aligned
  var testInt32: array[16, int32]    # 16 elements to cover all SIMD sizes
  var testInt64: array[16, int64]
  var testFloat32: array[16, float32]
  var testFloat64: array[16, float64]

  # Initialize arrays to prevent accessing uninitialized memory
  for i in 0..<16:
    testInt32[i] = int32(i)
    testInt64[i] = int64(i)
    testFloat32[i] = float32(i)
    testFloat64[i] = float64(i)

  var testSIMDInt32 = newSIMD(addr testInt32[0])
  var testSIMDInt64 = newSIMD(addr testInt64[0])
  var testSIMDFloat32 = newSIMD(addr testFloat32[0])
  var testSIMDFloat64 = newSIMD(addr testFloat64[0])