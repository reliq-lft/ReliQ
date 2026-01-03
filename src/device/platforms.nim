#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/device/platform.nim
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

when defined(nvidia):
  const PLATFORM* = "nvidia"
elif defined(amd):
  const PLATFORM* = "amd"
elif defined(cpu):
  const PLATFORM* = "cpu"

template nvidia*(body: untyped): untyped =
  ## Decorator to indicate that a function or proc is specific to the NVIDIA platform.
  when PLATFORM == "nvidia": body
  else: discard

template amd*(body: untyped): untyped =
  ## Decorator to indicate that a function or proc is specific to the AMD platform.
  when PLATFORM == "amd": body
  else: discard

template gpu*(body: untyped): untyped =
  ## Decorator to indicate that a function or proc is specific to GPU platforms.
  nvidia: body
  amd: body
  else: discard

template cpu*(body: untyped): untyped =
  ## Decorator to indicate that a function or proc is specific to the CPU platform.
  when PLATFORM == "cpu": body
  else: discard