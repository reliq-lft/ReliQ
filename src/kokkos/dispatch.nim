#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/kokkos/dispatch.nim
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

import kokkosbase

# shorten pragmas pointing to Kokkos headers and include local dispatch wrapper
kokkos: discard

#[ frontend: execution policy types ]#

#[ -- NOT READY YET --
type
  RangePolicy* {.importcpp: "Kokkos::RangePolicy", kokkos.} = object
  MDRangePolicy* {.importcpp: "Kokkos::MDRangePolicy", kokkos.} = object
  TeamPolicy* {.importcpp: "Kokkos::TeamPolicy", kokkos.} = object
  TeamThreadRange* {.importcpp: "Kokkos::TeamThreadRange", kokkos.} = object
  ThreadVectorRange* {.importcpp: "Kokkos::ThreadVectorRange", kokkos.} = object
-- NOT READY YET -- ]#

#[ frontend: execution policy constructors ]#

#[ -- NOT READY YET --
# range policy constructor
proc newRangePolicy(begin, stop: cint): RangePolicy
  {.importcpp: "Kokkos::RangePolicy<>(@)", constructor, kokkos.}

# MD range policy constructor
proc newMDRangePolicy*(start, stop: cint): MDRangePolicy
  {.importcpp: "Kokkos::MDRangePolicy<>(@)", constructor, kokkos.}

# team policy constructor
proc newTeamPolicy*(leagueSize: cint; teamSize: cint): TeamPolicy
  {.importcpp: "Kokkos::TeamPolicy<>(@)", constructor, kokkos.}

# team thread range constructor
proc newTeamThreadRange*(teamPolicy: TeamPolicy; start, stop: cint): TeamThreadRange
  {.importcpp: "Kokkos::TeamThreadRange(@)", constructor, kokkos.}

# thread vector range constructor
proc newThreadVectorRange*(
  teamPolicy: TeamPolicy; 
  start, stop: cint
): ThreadVectorRange {.importcpp: "Kokkos::ThreadVectorRange(@)", constructor, kokkos.}
-- NOT READY YET -- ]# 

#[ wrapper for parallel for ]#

# parallel for wrapper
proc forall(start, stop: cint; body: proc(i: int) {.cdecl.}) 
  {.importcpp: "parallel_for_range(@)", kokkos_wrapper.}
proc forall*(start, stop: SomeInteger; body: proc(i: int) {.cdecl.}) {.inline.} =
  forall(cint(start), cint(stop), body)

when isMainModule:
  import runtime
  import utils
  const verbosity = 1
  reliq:
    let (start, stop) = (0, 1024)
    #let 
    #  execPolicyA = newRangePolicy(start, stop)
    #  execPolicyB = newMDRangePolicy(0, 1024)
    #  execPolicyC = newTeamPolicy(16, 16)
    #  execPolicyD = newTeamThreadRange(execPolicyC, 0, 16)
    #  execPolicyE = newThreadVectorRange(execPolicyC, 0, 16)

    proc body(n: int) {.cdecl.} =
      if verbosity > 1: print $n + ": hello, fellow traveler!"
    forall(start, stop, body)

#[
# include kokkos headers
kokkos: {.pragma: iostream, header: "<iostream>".}

# test implementation of a parallel for
proc parallelFor(tag: cstring; start, stop: cint) 
  {.importcpp: "Kokkos::parallel_for(#, Kokkos::RangePolicy<>(#, #), KOKKOS_LAMBDA (const int i) { std::cout << i << \": hello, fellow traveler!\" << std::endl; })", inline, kokkos, iostream.}

when isMainModule:
  import runtime
  reliq:
    parallelFor("test", 0, 1024)
]#