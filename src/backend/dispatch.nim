#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/backend/dispatch.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of chadge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, medge, publish, distribute, sublicense, and/or sell
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
import kokkos/[kokkosbase]

# import backend header files
kokkos: discard

#[ frontend: dispatch types ]#

type
  ParallelForContext* = ref object
    ## Context for parallel dispatch routines
    ##
    ## <in need of documentation>
    
    # user data pointers
    ptrs: seq[pointer]

    # metadata for dispatching and dispatched routines
    iters: int
    strs: seq[string]
    ints: seq[int]
    flts: seq[float]

type ParallelForFunctor* = proc(thread: int, context: ParallelForContext) {.cdecl.}

#[ frontend: parallel for constructor and methods ]#

proc newParallelForContext*(
  iters: int;
  ptrs: seq[pointer] = @[];
  strs: seq[string] = @[];
  ints: seq[int] = @[];
  flts: seq[float] = @[]
): ParallelForContext =
  ParallelForContext(ptrs: ptrs, iters: iters, strs: strs, ints: ints, flts: flts)

template pack*[T](ctx: ParallelForContext; items: varargs[ptr T]): untyped =
  for item in items: ctx.ptrs.add(cast[pointer](item))

macro pack*[T](ctx: ParallelForContext; items: varargs[var T]): untyped =
  result = newCall(bindSym"pack")
  result.add(ctx)
  for item in items: result.add(newCall(bindSym"addr", item))

#[ backend/frontend: local parallel dispatch ]#

proc parallelForRange(
  start, stop: SomeInteger; 
  body: ParallelForFunctor; 
  ctx: ParallelForContext
) {.importcpp: "parallel_for_range(@)", kokkos_wrapper.}

template each*(ctx: ParallelForContext; n, work: untyped): untyped =
  # thoughts:
  # * i'd like to have this take in variables, create a context, and cast
  #   them to pointers of the appropriate type. i'd also like the variable
  #   names to be available in the body, even though the user won't know that
  #   the names refer to pointers. something like:
  #   ctx.each(i, (x, float), (y, float)):
  #     x[] += 1.0
  #     y[] += 1.0
  #   ... sort of
  let body: ParallelForFunctor = proc(
      thread: int; 
      context: ParallelForContext
    ) {.cdecl.} =
    let 
      iters = context.iters div numThreads()
      remdr = context.iters mod numThreads()
    let
      start = thread * iters
      stop = case thread != numThreads() - 1:
        of true: start + iters
        of false: (thread + 1) * iters + remdr
    proc myThread(): int {.inject.} = thread 
    for n in start..<stop: work
  parallelForRange(0, numThreads(), body, ctx)

# lessons learned:
# * nested procedures that are to be passed to a call expecting a specific 
#   procedure kind must be declared with that procedure kind explicitly (see 
#   "body" declaration in "each" template above)
when isMainModule:
  import runtime
  import utils/[reliqutils]
  const verbosity = 1
  reliq:
    var 
      x = 10.0
      y = 20.0
    var ctx = newParallelForContext(100)
    
    ctx.pack(x, y)
    print "x =", x
    print "y =", y
    ctx.each(i):
      let x = cast[ptr float](ctx.ptrs[0])
      let y = cast[ptr float](ctx.ptrs[1])
      if myThread() == 0: 
        x[] += 1.0
        y[] += 1.0
      if verbosity > 1:
        print "i =", i, "thread =", myThread(), "x =", x[], "y =", y[]
    print "x =", x
    print "y =", y