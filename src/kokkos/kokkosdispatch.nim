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

import std/[macros]

import kokkosbase
import utils/[nimutils]

# import backend header files
Kokkos: discard

# constants for Kokkos parallel_for wrapping
const LAMBDA = "KOKKOS_LAMBDA" 
const 
  MEMBERTYPE = "Kokkos::TeamPolicy<>::member_type&"
  PARFOR = "Kokkos::parallel_for"
  TEAMFORALL = PARFOR & "(*#," + LAMBDA + "(const" + MEMBERTYPE +  "team) { #(team); })"
const FORALL = PARFOR & "(*#," + LAMBDA + "(const int idx) { #(idx); })"
const 
  THREADVECTORRANGE = "Kokkos::ThreadVectorRange(#, #, #)"
  FOREACH = PARFOR & "(" & THREADVECTORRANGE & ", [&] (int idx) { #(idx); })"

# team member type
type ThreadTeam* {.importcpp: "Kokkos::TeamPolicy<>::member_type", kokkos.} = object

type # range/team policy types
  RangePolicy {.importcpp: "Kokkos::RangePolicy<>", kokkos.} = object
  TeamPolicy {.importcpp: "Kokkos::TeamPolicy<>", kokkos.} = object
  ThreadVectorPolicy {.importcpp: "Kokkos::ThreadVectorRange", kokkos.} = object

type # Nim --> C++ procedure kernels
  NimRangeKernel = proc(idx: cint) {.cdecl.}
  NimTeamKernel = proc(team: ThreadTeam) {.cdecl.}

#[ parallel dispatch policy wrappers ]#

# C++ new/delete bindings for RangePolicy
proc newRangePolicy(lower, upper: int): ptr RangePolicy 
  {.importcpp: "new Kokkos::RangePolicy<>(#, #)", constructor, kokkos.}
proc deleteRangePolicy(policy: ptr RangePolicy) {.importcpp: "delete #", kokkos.}

# C++ new/delete bindings for TeamPolicy
proc newTeamPolicy(leagueSize, teamSize: int): ptr TeamPolicy 
  {.importcpp: "new Kokkos::TeamPolicy<>(#, #)", constructor, kokkos.}
proc deleteTeamPolicy(policy: ptr TeamPolicy) {.importcpp: "delete #", kokkos.}

#[ parallel dispatch wrappers ]#

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

#[ thread team methods ]#

## gets team size
proc teamSize*(team: ThreadTeam): int {.importcpp: "#.team_size()", inline, kokkos.}

## gets league rank of team member
proc leagueRank*(team: ThreadTeam): int 
  {.importcpp: "#.league_rank()", inline, kokkos.}

## gets league rank of team member
proc teamRank*(team: ThreadTeam): int {.importcpp: "#.team_rank()", inline, kokkos.}

## gets league rank of team member
proc myRank*(team: ThreadTeam): int = 
  team.teamRank() + team.leagueRank() * team.teamSize()

## setups up team barrier
proc wait*(team: ThreadTeam) {.importcpp: "#.team_barrier()", kokkos.}

## ensures that only main thread executes
template threadMain*(thread: ThreadTeam; body: untyped): untyped =
  if thread.myRank() == 0: body

#[ thread/vector dispatch ]#

template forall*(lower, upper: SomeInteger; n, body: untyped): untyped =
  # threaded `forall` construct
  rangeForAll(lower, upper):
    proc(idx: cint) {.cdecl.} = 
      let n = idx
      body
  localBarrier()

template foreach*(lower, upper: SomeInteger; n, body: untyped): untyped =
  # vectorized `foreach` construct
  team.threadForEach(lower, upper):
    proc(idx: cint) {.cdecl.} =
      let n = idx
      body

template forevery*(lower, upper: SomeInteger; n, body: untyped): untyped =
  # `forall` + `foreach` = `forevery` construct. Don't think about it too hard.
  let segment = (upper - lower) div numThreads()
  teamForAll(numThreads(), 1):
    proc(localTeamHandle: ThreadTeam) {.cdecl.} = 
      let tlo = localTeamHandle.myRank()*segment
      let thi = tlo + segment
      localTeamHandle.threadForEach(tlo, thi):
        proc(idx: cint) {.cdecl.} =
          let n = idx
          body
  localBarrier()

#[ threaded dispatch ]#

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

#[ thread/vector dispatch macros ]#

macro all*(x: ForLoopStmt): untyped =
  ## Threaded for loop consturct
  ## 
  ## Turns a `for` loop of the form:
  ## ```
  ## for i in all 0..10: <body>
  ## ```
  ## into a threaded parallel loop.
  let (idnt, call, body) = (x[0], x[1], x[2])
  let (itr, rng) = (call[1], call[1][0])
  let (lo, hi) = (itr[1], itr[^1])
  result = case $rng:
    of "..": 
      quote do: forall(`lo`, `hi` + 1, `idnt`, `body`)
    of "..<":
      quote do: forall(`lo`, `hi`, `idnt`, `body`)
    else: 
      quote do: discard

macro each*(x: ForLoopStmt): untyped =
  ## Vectorized for loop consturct
  ## 
  ## Turns a `for` loop of the form:
  ## ```
  ## for i in each 0..10: <body>
  ## ```
  ## into a vectorized parallel loop.
  let (idnt, call, body) = (x[0], x[1], x[2])
  let (itr, rng) = (call[1], call[1][0])
  let (lo, hi) = (itr[1], itr[^1])
  result = case $rng:
    of "..": 
      quote do: foreach(`lo`, `hi` + 1, `idnt`, `body`)
    of "..<":
      quote do: foreach(`lo`, `hi`, `idnt`, `body`)
    else: 
      quote do: discard

macro every*(x: ForLoopStmt): untyped =
  ## Threaded + vectorized for loop consturct
  ## 
  ## Turns a `for` loop of the form:
  ## ```
  ## for i in every 0..10: <body>
  ## ```
  ## into a threaded + vectorized parallel loop. Behaves like `Grid`'s
  ## `accelerator_for` construct. 
  let (idnt, call, body) = (x[0], x[1], x[2])
  let (itr, rng) = (call[1], call[1][0])
  let (lo, hi) = (itr[1], itr[^1])
  result = case $rng:
    of "..": 
      quote do: forevery(`lo`, `hi` + 1, `idnt`, `body`)
    of "..<":
      quote do: forevery(`lo`, `hi`, `idnt`, `body`)
    else: 
      quote do: discard

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
    var 
      testSeqA, testSeqB: seq[float] = newSeq[float](16)
    proc funWork[T](team: T; seqA, seqB: var seq[float]) =
      for i in 0..<seqA.len div numThreads():
        seqA[team.myRank()] += 1.0
        seqB[team.myRank()] += 2.0
    var testVar = 2.0
    print "testVar before threads:", testVar
    threadTeams(4, 4):
      let rank = team.leagueRank()
      if verbosity > 1: print "Hello from team member with league rank:", rank
      team.wait()
      if verbosity > 1: print "Goodbye from team member with league rank:", rank
      team.funWork(testSeqA, testSeqB)
      team.wait()
      team.threadMain:
        if verbosity > 1: print "main thread:", rank
        for i in each 0..3:
          testVar += 1.0
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
    for i in all 0..<10:
      if verbosity > 1: print "Hello from forall with i =", i
    assert(testVar == 7.0)
    let (lo, hi) = (0, numThreads()*10)
    var testSeq = newSeq[float](hi)
    print "seq before:", testSeq
    for i in every lo..<hi:
      testSeq[i] += 1
    print "seq after:", testSeq
    for el in testSeq: assert(el == 1.0)
