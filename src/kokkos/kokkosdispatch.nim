#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/kokkos/kokkosdispatch.nim
  Contact: reliq-lft@proton.me

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

import kokkosbase
import utils

# import backend header files
kokkos: discard

# backend: constants for Kokkos parallel_for wrapping
const LAMBDA = "KOKKOS_LAMBDA" 
const 
  MEMBERTYPE = "Kokkos::TeamPolicy<>::member_type&"
  PARFOR = "Kokkos::parallel_for"
  TEAMFORALL = PARFOR & "(*#," + LAMBDA + "(const" + MEMBERTYPE +  "team) { #(team); })"
const FORALL = PARFOR & "(*#," + LAMBDA + "(const int idx) { #(idx); })"
const 
  THREADVECTORRANGE = "Kokkos::ThreadVectorRange(#, #, #)"
  FOREACH = PARFOR & "(" & THREADVECTORRANGE & ", [&] (int idx) { #(idx); })"

# frontend: team member type
type ThreadTeam* {.importcpp: "Kokkos::TeamPolicy<>::member_type", kokkos.} = object

type # backend: range/team policy types
  RangePolicy {.importcpp: "Kokkos::RangePolicy<>", kokkos.} = object
  TeamPolicy {.importcpp: "Kokkos::TeamPolicy<>", kokkos.} = object
  ThreadVectorPolicy {.importcpp: "Kokkos::ThreadVectorRange", kokkos.} = object

type # backend: Nim --> C++ procedure kernels
  NimRangeKernel = proc(idx: cint) {.cdecl.}
  NimTeamKernel = proc(team: ThreadTeam) {.cdecl.}

#[ backend: parallel dispatch policy wrappers ]#

# C++ new/delete bindings for RangePolicy
proc newRangePolicy(lower, upper: int): ptr RangePolicy 
  {.importcpp: "new Kokkos::RangePolicy<>(#, #)", constructor, kokkos.}
proc deleteRangePolicy(policy: ptr RangePolicy) {.importcpp: "delete #", kokkos.}

# C++ new/delete bindings for TeamPolicy
proc newTeamPolicy(leagueSize, teamSize: int): ptr TeamPolicy 
  {.importcpp: "new Kokkos::TeamPolicy<>(#, #)", constructor, kokkos.}
proc deleteTeamPolicy(policy: ptr TeamPolicy) {.importcpp: "delete #", kokkos.}

#[ backend: parallel dispatch wrappers ]#

# parallel for wrappers
proc forall(policy: ptr TeamPolicy; body: NimTeamKernel) 
  {.importcpp: TEAMFORALL, kokkos.}
proc forall(policy: ptr RangePolicy; body: NimRangeKernel) 
  {.importcpp: FORALL, kokkos.}
proc foreach(team: ThreadTeam; lower, upper: int; body: NimRangeKernel)
  {.importcpp: FOREACH, kokkos.}

# range for all
proc rangeForAll(lower, upper: int; body: NimRangeKernel) {.inline.} =
  let policy = newRangePolicy(lower, upper)
  policy.forall: body
  deleteRangePolicy(policy)

# team "parallel for" wrapper
proc teamForAll(leagueSize, teamSize: int; body: NimTeamKernel) {.inline.} =
  let policy = newTeamPolicy(leagueSize, teamSize)
  policy.forall: body
  deleteTeamPolicy(policy)

# range "hierarchical parallel for" wrapper
proc threadForEach(
  team: ThreadTeam; 
  lower, upper: int; 
  body: NimRangeKernel
) {.inline.} = team.foreach(lower, upper): body

#[ frontend: thread team methods ]#

## frontend: gets team size
proc teamSize*(team: ThreadTeam): int {.importcpp: "#.team_size()", inline, kokkos.}

## frontend: gets league rank of team member
proc leagueRank*(team: ThreadTeam): int 
  {.importcpp: "#.league_rank()", inline, kokkos.}

## frontend: gets league rank of team member
proc teamRank*(team: ThreadTeam): int {.importcpp: "#.team_rank()", inline, kokkos.}

## frontend: gets league rank of team member
proc myRank*(team: ThreadTeam): int = 
  team.teamRank() + team.leagueRank() * team.teamSize()

## frontend: setups up team barrier
proc wait*(team: ThreadTeam) {.importcpp: "#.team_barrier()", kokkos.}

## frontend: ensures that only main thread executes
template threadMain*(thread: ThreadTeam; body: untyped): untyped =
  if thread.myRank() == 0: body

#[ frontend: threaded dispatch ]#

template threadTeams*(leagueSize, teamSize: SomeInteger; body: untyped): untyped =
  ## thread teams construct
  ## 
  ## <in need of documentation>
  teamForAll(leagueSize, teamSize): 
    proc(localTeamHandle: ThreadTeam) {.cdecl.} = 
      let team {.inject.} = localTeamHandle
      body
  localBarrier()

template threads*(body: untyped): untyped =
  ## `QEX`-like `threads` construct
  ##
  ## <in need of documentation>
  teamForAll(numThreads(), 1):
    proc(localTeamHandle: ThreadTeam) {.cdecl.} = 
      let thread {.inject.} = localTeamHandle
      body
  localBarrier()

template forall*(lower, upper: SomeInteger; n, body: untyped): untyped =
  ## `forall` construct
  ## 
  ## <in need of documentation>
  rangeForAll(lower, upper):
    proc(idx: cint) {.cdecl.} = 
      let n = idx
      body
  localBarrier()

template forall*(numIters: SomeInteger; n, body: untyped): untyped = 
  ## `forall` construct
  ## 
  ## Equivalent to `forall` with lower = 0 and upper = numIters
  forall(0, numIters, n, body)

template foreach*(lower, upper: SomeInteger; n, body: untyped): untyped =
  ## vectorized `foreach` construct
  ## 
  ## <in need of documentation>
  team.threadForEach(lower, upper):
    proc(idx: cint) {.cdecl.} =
      let n = idx
      body

template foreach*(numIters: SomeInteger; n, body: untyped): untyped =
  ## vectorized `foreach` construct
  ##
  ## Equivalent to `foreach` with lower = 0 and upper = numIters
  foreach(0, numIters, n, body)

template forevery*(lower, upper: SomeInteger; n, body: untyped): untyped =
  ## `forall` + `foreach` = `forevery` construct
  ## 
  ## Don't think about it too hard.
  ## <in need of documentation>
  let segment = (upper - lower) div numThreads()
  teamForAll(numThreads(), 1):
    proc(localTeamHandle: ThreadTeam) {.cdecl.} = 
      let
        tlower = localTeamHandle.myRank()*segment
        tupper = tlower + segment
      localTeamHandle.threadForEach(tlower, tupper):
        proc(idx: cint) {.cdecl.} =
          let n = idx
          body
  localBarrier()

template forevery*(numIters: SomeInteger; n, body: untyped): untyped =
  ## `forall` + `foreach` = `forevery` construct
  ## 
  ## Equivalent to `forevery` with lower = 0 and upper = numIters
  forevery(0, numIters, n, body)

# lessons learned:
# * nested procedures that are to be passed to a call expecting a specific 
#   procedure kind must be declared with that procedure kind explicitly
# needed? vvvvv
# template capture(body: untyped): untyped = 
#   setupForeignThreadGc()
#   body
#   tearDownForeignThreadGc()
when isMainModule:
  import runtime
  const verbosity = 1
  reliq:
    var testVar = 2.0
    print "testVar before threads:", testVar
    threadTeams(4, 4):
      let rank = team.leagueRank()
      if verbosity > 1: print "Hello from team member with league rank:", rank
      team.wait()
      if verbosity > 1: print "Goodbye from team member with league rank:", rank
      team.threadMain:
        foreach(0, 4, i):
          testVar += 1
    threads:
      let rank = thread.myRank()
      if verbosity > 1: print "Hello from team member with my rank:", rank
      thread.wait()
      if verbosity > 1: print "Goodbye from team member with my rank:", rank
      thread.wait()
      thread.threadMain:
        testVar += 1.0
        if verbosity > 1: print "main thread:", rank
    print "testVar after threads:", testVar
    forall(0, 10, i): 
      if verbosity > 1: print "Hello from forall with i =", i
    assert(testVar == 7.0)
    let (lo, hi) = (0, numThreads()*10)
    var testSeq = newSeq[float](hi)
    print "seq before:", testSeq
    forevery(lo, hi, i):
      testSeq[i] += 1
    print "seq after:", testSeq
    for el in testSeq: assert(el == 1.0)