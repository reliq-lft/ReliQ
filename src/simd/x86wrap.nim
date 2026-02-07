#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/simd/x86wrap.nim
  Contact: reliq-lft@proton.me

  Author: James Osborn and Xiao-Yong Jin
  Modifications: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  Original License:

  MIT License
  
  Copyright (c) 2017 James Osborn

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

  ReliQ Modifications License:

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

{.pragma: imm, header:"immintrin.h".}

{.pragma: imms, header:"immintrin.h", incompletestruct.}

type
  m64*   {.importc: "__m64"  , imms.} = object
  m128*  {.importc: "__m128" , imms.} = object
  m128d* {.importc: "__m128d", imms.} = object
  m128i* {.importc: "__m128i", imms.} = object
  m128h* = distinct int64
  m256*  {.importc: "__m256" , imms.} = object
  m256d* {.importc: "__m256d", imms.} = object
  m256i* {.importc: "__m256i", imms.} = object
  m256h* = distinct m128i
  m512*  {.importc: "__m512" , imms.} = object
  m512d* {.importc: "__m512d", imms.} = object
  m512i* {.importc: "__m512i", imms.} = object
  m512h* = distinct m256i
  mmask8*  {.importc: "__mmask8" , imms.} = object
  mmask16* {.importc: "__mmask16", imms.} = object
  mmask32* {.importc: "__mmask32", imms.} = object
  mmask64* {.importc: "__mmask64", imms.} = object