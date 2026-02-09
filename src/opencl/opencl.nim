#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/opencl/clir.nim
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

import clbase
import cldisp
import clreduce

export cldisp
export clreduce

template clParallel*(body: untyped): untyped =
  initCL()
  block:
    body
    # Views declared in body are destroyed here (end of block),
    # while OpenCL queues and context are still alive.
  finalizeCL()

when isMainModule:
  import std/[unittest, random, math, times]

  suite "OpenCL clParallel Dispatch":
    
    test "Vector addition - small (1K elements)":
      clParallel:
        const size = 1_000
        var
          a = newSeq[float](size)
          b = newSeq[float](size)
          c = newSeq[float](size)

        for i in 0 ..< size:
          a[i] = float(i)
          b[i] = float(2 * i)

        var
          gpuA = clContext.bufferLike(a)
          gpuB = clContext.bufferLike(b)
          gpuC = clContext.bufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        for i in each 0..<size:
          gpuC[i] = gpuA[i] + gpuB[i]
        
        clQueues[0].read(c, gpuC)

        for i in 0 ..< size:
          check c[i] == a[i] + b[i]

    test "Vector addition - medium (100K elements)":
      clParallel:
        const size = 100_000
        var
          a = newSeq[float](size)
          b = newSeq[float](size)
          c = newSeq[float](size)

        for i in 0 ..< size:
          a[i] = float(i) * 0.5
          b[i] = float(size - i) * 0.25

        var
          gpuA = clContext.bufferLike(a)
          gpuB = clContext.bufferLike(b)
          gpuC = clContext.bufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        for i in each 0..<size:
          gpuC[i] = gpuA[i] + gpuB[i]
        
        clQueues[0].read(c, gpuC)

        for i in 0 ..< size:
          check c[i] == a[i] + b[i]

    test "Vector addition - large (1M elements)":
      clParallel:
        const size = 1_000_000
        var
          a = newSeq[float](size)
          b = newSeq[float](size)
          c = newSeq[float](size)

        for i in 0 ..< size:
          a[i] = float(i)
          b[i] = float(2 * i)

        var
          gpuA = clContext.bufferLike(a)
          gpuB = clContext.bufferLike(b)
          gpuC = clContext.bufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        for i in each 0..<size:
          gpuC[i] = gpuA[i] + gpuB[i]
        
        clQueues[0].read(c, gpuC)

        for i in 0 ..< size:
          check c[i] == a[i] + b[i]

    test "Vector addition - stress test (10M elements)":
      clParallel:
        const size = 10_000_000
        var
          a = newSeq[float](size)
          b = newSeq[float](size)
          c = newSeq[float](size)

        for i in 0 ..< size:
          a[i] = float(i mod 1000)
          b[i] = float((i * 7) mod 1000)

        var
          gpuA = clContext.bufferLike(a)
          gpuB = clContext.bufferLike(b)
          gpuC = clContext.bufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        let start = cpuTime()
        for i in each 0..<size:
          gpuC[i] = gpuA[i] + gpuB[i]
        let elapsed = cpuTime() - start
        echo "  10M elements kernel time: ", elapsed * 1000, " ms"
        
        clQueues[0].read(c, gpuC)

        # Sample check (checking all 10M would be slow)
        for i in countup(0, size - 1, size div 1000):
          check c[i] == a[i] + b[i]

    test "Random data verification":
      clParallel:
        const size = 500_000
        var rng = initRand(42)
        var
          a = newSeq[float](size)
          b = newSeq[float](size)
          c = newSeq[float](size)

        for i in 0 ..< size:
          a[i] = rng.rand(1000.0)
          b[i] = rng.rand(1000.0)

        var
          gpuA = clContext.bufferLike(a)
          gpuB = clContext.bufferLike(b)
          gpuC = clContext.bufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        for i in each 0..<size:
          gpuC[i] = gpuA[i] + gpuB[i]
        
        clQueues[0].read(c, gpuC)

        for i in 0 ..< size:
          check abs(c[i] - (a[i] + b[i])) < 1e-10

    test "Edge case - single element":
      clParallel:
        const size = 1
        var
          a = @[42.0]
          b = @[58.0]
          c = @[0.0]

        var
          gpuA = clContext.bufferLike(a)
          gpuB = clContext.bufferLike(b)
          gpuC = clContext.bufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        for i in each 0..<size:
          gpuC[i] = gpuA[i] + gpuB[i]
        
        clQueues[0].read(c, gpuC)

        check c[0] == 100.0

    test "Edge case - power of 2 sizes":
      for power in [8, 10, 12, 14, 16]:
        clParallel:
          let size = 1 shl power
          var
            a = newSeq[float](size)
            b = newSeq[float](size)
            c = newSeq[float](size)

          for i in 0 ..< size:
            a[i] = float(i)
            b[i] = float(i * 2)

          var
            gpuA = clContext.bufferLike(a)
            gpuB = clContext.bufferLike(b)
            gpuC = clContext.bufferLike(c)

          clQueues[0].write(a, gpuA)
          clQueues[0].write(b, gpuB)

          for i in each 0..<size:
            gpuC[i] = gpuA[i] + gpuB[i]
          
          clQueues[0].read(c, gpuC)

          for i in 0 ..< size:
            check c[i] == a[i] + b[i]

    test "Edge case - non-power of 2 sizes":
      for size in [17, 127, 1023, 4097, 65537]:
        clParallel:
          var
            a = newSeq[float](size)
            b = newSeq[float](size)
            c = newSeq[float](size)

          for i in 0 ..< size:
            a[i] = float(i)
            b[i] = float(i * 3)

          var
            gpuA = clContext.bufferLike(a)
            gpuB = clContext.bufferLike(b)
            gpuC = clContext.bufferLike(c)

          clQueues[0].write(a, gpuA)
          clQueues[0].write(b, gpuB)

          for i in each 0..<size:
            gpuC[i] = gpuA[i] + gpuB[i]
          
          clQueues[0].read(c, gpuC)

          for i in 0 ..< size:
            check c[i] == a[i] + b[i]

    test "Negative and zero values":
      clParallel:
        const size = 10_000
        var
          a = newSeq[float](size)
          b = newSeq[float](size)
          c = newSeq[float](size)

        for i in 0 ..< size:
          a[i] = float(i) - float(size div 2)  # -5000 to 4999
          b[i] = -a[i]  # Should sum to 0

        var
          gpuA = clContext.bufferLike(a)
          gpuB = clContext.bufferLike(b)
          gpuC = clContext.bufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        for i in each 0..<size:
          gpuC[i] = gpuA[i] + gpuB[i]
        
        clQueues[0].read(c, gpuC)

        for i in 0 ..< size:
          check c[i] == 0.0

    test "Large values near float limits":
      clParallel:
        const size = 1_000
        var
          a = newSeq[float](size)
          b = newSeq[float](size)
          c = newSeq[float](size)

        for i in 0 ..< size:
          a[i] = 1e30 + float(i)
          b[i] = 1e30 + float(i * 2)

        var
          gpuA = clContext.bufferLike(a)
          gpuB = clContext.bufferLike(b)
          gpuC = clContext.bufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        for i in each 0..<size:
          gpuC[i] = gpuA[i] + gpuB[i]
        
        clQueues[0].read(c, gpuC)

        for i in 0 ..< size:
          check abs(c[i] - (a[i] + b[i])) < 1e20  # Relative tolerance for large numbers

    test "Repeated kernel execution":
      clParallel:
        const size = 10_000
        const iterations = 10
        var
          a = newSeq[float](size)
          b = newSeq[float](size)
          c = newSeq[float](size)

        for i in 0 ..< size:
          a[i] = float(i)
          b[i] = 1.0

        var
          gpuA = clContext.bufferLike(a)
          gpuB = clContext.bufferLike(b)
          gpuC = clContext.bufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        # Run the same kernel multiple times
        for iter in 0 ..< iterations:
          for i in each 0..<size:
            gpuC[i] = gpuA[i] + gpuB[i]
        
        clQueues[0].read(c, gpuC)

        # Result should still be a[i] + b[i] (last iteration)
        for i in 0 ..< size:
          check c[i] == a[i] + b[i]

    test "Typed GpuBuffer - float64":
      clParallel:
        const size = 1_000
        var
          a = newSeq[float64](size)
          b = newSeq[float64](size)
          c = newSeq[float64](size)

        for i in 0 ..< size:
          a[i] = float64(i)
          b[i] = float64(2 * i)

        var
          gpuA = clContext.gpuBufferLike(a)
          gpuB = clContext.gpuBufferLike(b)
          gpuC = clContext.gpuBufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        for i in each 0..<size:
          gpuC[i] = gpuA[i] + gpuB[i]
        
        clQueues[0].read(c, gpuC)

        for i in 0 ..< size:
          check c[i] == a[i] + b[i]

    test "Typed GpuBuffer - float32":
      clParallel:
        const size = 1_000
        var
          a = newSeq[float32](size)
          b = newSeq[float32](size)
          c = newSeq[float32](size)

        for i in 0 ..< size:
          a[i] = float32(i)
          b[i] = float32(2 * i)

        var
          gpuA = clContext.gpuBufferLike(a)
          gpuB = clContext.gpuBufferLike(b)
          gpuC = clContext.gpuBufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        for i in each 0..<size:
          gpuC[i] = gpuA[i] + gpuB[i]
        
        clQueues[0].read(c, gpuC)

        for i in 0 ..< size:
          check c[i] == a[i] + b[i]
    test "Typed GpuBuffer - int32":
      clParallel:
        const size = 1_000
        var
          a = newSeq[int32](size)
          b = newSeq[int32](size)
          c = newSeq[int32](size)

        for i in 0 ..< size:
          a[i] = int32(i)
          b[i] = int32(2 * i)

        var
          gpuA = clContext.gpuBufferLike(a)
          gpuB = clContext.gpuBufferLike(b)
          gpuC = clContext.gpuBufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        for i in each 0..<size:
          gpuC[i] = gpuA[i] + gpuB[i]
        
        clQueues[0].read(c, gpuC)

        for i in 0 ..< size:
          check c[i] == a[i] + b[i]

    # ===== AXPY and scalar operations =====
    
    test "AXPY - y = a*x + y":
      clParallel:
        const size = 10_000
        const alpha = 2.5
        var
          x = newSeq[float64](size)
          y = newSeq[float64](size)
          expected = newSeq[float64](size)

        for i in 0 ..< size:
          x[i] = float64(i)
          y[i] = float64(size - i)
          expected[i] = alpha * x[i] + y[i]

        var
          gpuX = clContext.gpuBufferLike(x)
          gpuY = clContext.gpuBufferLike(y)

        clQueues[0].write(x, gpuX)
        clQueues[0].write(y, gpuY)

        for i in each 0..<size:
          gpuY[i] = alpha * gpuX[i] + gpuY[i]
        
        clQueues[0].read(y, gpuY)

        for i in 0 ..< size:
          check abs(y[i] - expected[i]) < 1e-10

    test "Scalar multiply - y = a*x":
      clParallel:
        const size = 5_000
        const alpha = 3.14159
        var
          x = newSeq[float64](size)
          y = newSeq[float64](size)

        for i in 0 ..< size:
          x[i] = float64(i) * 0.1

        var
          gpuX = clContext.gpuBufferLike(x)
          gpuY = clContext.gpuBufferLike(y)

        clQueues[0].write(x, gpuX)

        for i in each 0..<size:
          gpuY[i] = alpha * gpuX[i]
        
        clQueues[0].read(y, gpuY)

        for i in 0 ..< size:
          check abs(y[i] - alpha * x[i]) < 1e-10

    test "Vector subtraction":
      clParallel:
        const size = 5_000
        var
          a = newSeq[float64](size)
          b = newSeq[float64](size)
          c = newSeq[float64](size)

        for i in 0 ..< size:
          a[i] = float64(i * 2)
          b[i] = float64(i)

        var
          gpuA = clContext.gpuBufferLike(a)
          gpuB = clContext.gpuBufferLike(b)
          gpuC = clContext.gpuBufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        for i in each 0..<size:
          gpuC[i] = gpuA[i] - gpuB[i]
        
        clQueues[0].read(c, gpuC)

        for i in 0 ..< size:
          check c[i] == a[i] - b[i]

    test "Element-wise multiplication":
      clParallel:
        const size = 5_000
        var
          a = newSeq[float64](size)
          b = newSeq[float64](size)
          c = newSeq[float64](size)

        for i in 0 ..< size:
          a[i] = float64(i) * 0.1
          b[i] = float64(size - i) * 0.1

        var
          gpuA = clContext.gpuBufferLike(a)
          gpuB = clContext.gpuBufferLike(b)
          gpuC = clContext.gpuBufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        for i in each 0..<size:
          gpuC[i] = gpuA[i] * gpuB[i]
        
        clQueues[0].read(c, gpuC)

        for i in 0 ..< size:
          check abs(c[i] - a[i] * b[i]) < 1e-10

    test "Element-wise division":
      clParallel:
        const size = 5_000
        var
          a = newSeq[float64](size)
          b = newSeq[float64](size)
          c = newSeq[float64](size)

        for i in 0 ..< size:
          a[i] = float64(i + 1) * 10.0
          b[i] = float64(i + 1)

        var
          gpuA = clContext.gpuBufferLike(a)
          gpuB = clContext.gpuBufferLike(b)
          gpuC = clContext.gpuBufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        for i in each 0..<size:
          gpuC[i] = gpuA[i] / gpuB[i]
        
        clQueues[0].read(c, gpuC)

        for i in 0 ..< size:
          check abs(c[i] - 10.0) < 1e-10

    # ===== Conditional operations =====

    test "If statement - conditional assignment":
      clParallel:
        const size = 10_000
        var
          a = newSeq[float64](size)
          b = newSeq[float64](size)

        for i in 0 ..< size:
          a[i] = float64(i) - float64(size div 2)

        var
          gpuA = clContext.gpuBufferLike(a)
          gpuB = clContext.gpuBufferLike(b)

        clQueues[0].write(a, gpuA)

        for i in each 0..<size:
          if gpuA[i] > 0.0:
            gpuB[i] = gpuA[i]
          else:
            gpuB[i] = -gpuA[i]
        
        clQueues[0].read(b, gpuB)

        for i in 0 ..< size:
          check b[i] == abs(a[i])

    test "If-elif-else chain":
      clParallel:
        const size = 9_000
        var
          a = newSeq[int32](size)
          b = newSeq[int32](size)

        for i in 0 ..< size:
          a[i] = int32(i mod 3)  # 0, 1, 2, 0, 1, 2, ...

        var
          gpuA = clContext.gpuBufferLike(a)
          gpuB = clContext.gpuBufferLike(b)

        clQueues[0].write(a, gpuA)

        for i in each 0..<size:
          if gpuA[i] == 0:
            gpuB[i] = 100
          elif gpuA[i] == 1:
            gpuB[i] = 200
          else:
            gpuB[i] = 300
        
        clQueues[0].read(b, gpuB)

        for i in 0 ..< size:
          case a[i]
          of 0: check b[i] == 100
          of 1: check b[i] == 200
          else: check b[i] == 300

    test "Nested if statements":
      clParallel:
        const size = 10_000
        var
          x = newSeq[int32](size)
          y = newSeq[int32](size)
          result_arr = newSeq[int32](size)

        for i in 0 ..< size:
          x[i] = int32(i mod 4)
          y[i] = int32(i mod 3)

        var
          gpuX = clContext.gpuBufferLike(x)
          gpuY = clContext.gpuBufferLike(y)
          gpuR = clContext.gpuBufferLike(result_arr)

        clQueues[0].write(x, gpuX)
        clQueues[0].write(y, gpuY)

        for i in each 0..<size:
          if gpuX[i] > 1:
            if gpuY[i] > 0:
              gpuR[i] = 1
            else:
              gpuR[i] = 2
          else:
            gpuR[i] = 0
        
        clQueues[0].read(result_arr, gpuR)

        for i in 0 ..< size:
          if x[i] > 1:
            if y[i] > 0:
              check result_arr[i] == 1
            else:
              check result_arr[i] == 2
          else:
            check result_arr[i] == 0

    # ===== Complex arithmetic =====

    test "Polynomial evaluation - ax^2 + bx + c":
      clParallel:
        const size = 5_000
        const a_coef = 2.0
        const b_coef = -3.0
        const c_coef = 1.0
        var
          x = newSeq[float64](size)
          y = newSeq[float64](size)

        for i in 0 ..< size:
          x[i] = float64(i) * 0.01 - 25.0

        var
          gpuX = clContext.gpuBufferLike(x)
          gpuY = clContext.gpuBufferLike(y)

        clQueues[0].write(x, gpuX)

        for i in each 0..<size:
          gpuY[i] = a_coef * gpuX[i] * gpuX[i] + b_coef * gpuX[i] + c_coef
        
        clQueues[0].read(y, gpuY)

        for i in 0 ..< size:
          let expected = a_coef * x[i] * x[i] + b_coef * x[i] + c_coef
          check abs(y[i] - expected) < 1e-8

    test "Complex expression with multiple operators":
      clParallel:
        const size = 5_000
        var
          a = newSeq[float64](size)
          b = newSeq[float64](size)
          c = newSeq[float64](size)
          d = newSeq[float64](size)

        for i in 0 ..< size:
          a[i] = float64(i + 1)
          b[i] = float64(i + 2)  # Changed to avoid division by zero
          c[i] = float64((i mod 100) + 1)

        var
          gpuA = clContext.gpuBufferLike(a)
          gpuB = clContext.gpuBufferLike(b)
          gpuC = clContext.gpuBufferLike(c)
          gpuD = clContext.gpuBufferLike(d)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)
        clQueues[0].write(c, gpuC)

        for i in each 0..<size:
          gpuD[i] = (gpuA[i] + gpuB[i]) * gpuC[i] / (gpuA[i] + gpuB[i] + 1.0)
        
        clQueues[0].read(d, gpuD)

        for i in 0 ..< size:
          let expected = (a[i] + b[i]) * c[i] / (a[i] + b[i] + 1.0)
          check abs(d[i] - expected) < 1e-8

    # ===== Mixed types =====

    test "Mixed precision - float32 AXPY":
      clParallel:
        const size = 10_000
        const alpha: float32 = 2.5
        var
          x = newSeq[float32](size)
          y = newSeq[float32](size)
          expected = newSeq[float32](size)

        for i in 0 ..< size:
          x[i] = float32(i) * 0.1
          y[i] = float32(size - i) * 0.1
          expected[i] = alpha * x[i] + y[i]

        var
          gpuX = clContext.gpuBufferLike(x)
          gpuY = clContext.gpuBufferLike(y)

        clQueues[0].write(x, gpuX)
        clQueues[0].write(y, gpuY)

        for i in each 0..<size:
          gpuY[i] = alpha * gpuX[i] + gpuY[i]
        
        clQueues[0].read(y, gpuY)

        for i in 0 ..< size:
          check abs(y[i] - expected[i]) < 1e-3  # float32 has ~7 decimal digits of precision

    test "Integer arithmetic":
      clParallel:
        const size = 10_000
        var
          a = newSeq[int32](size)
          b = newSeq[int32](size)
          c = newSeq[int32](size)

        for i in 0 ..< size:
          a[i] = int32(i * 3 + 7)
          b[i] = int32(i * 2 + 5)

        var
          gpuA = clContext.gpuBufferLike(a)
          gpuB = clContext.gpuBufferLike(b)
          gpuC = clContext.gpuBufferLike(c)

        clQueues[0].write(a, gpuA)
        clQueues[0].write(b, gpuB)

        for i in each 0..<size:
          gpuC[i] = gpuA[i] * gpuB[i] + gpuA[i] - gpuB[i]
        
        clQueues[0].read(c, gpuC)

        for i in 0 ..< size:
          check c[i] == a[i] * b[i] + a[i] - b[i]