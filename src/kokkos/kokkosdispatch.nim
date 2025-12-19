#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/kokkos/kokkosdispatch.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
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

# considerations for parallel reduction:
# - should be threaded & vectorized
# - need to handle different data types (int, float, complex)
# - need to handle different reduction operations (sum, max, min, etc.)

import std/[macros]

import kokkosbase

Kokkos: discard

const LAMBDA = "KOKKOS_LAMBDA" 
const 
  MEMBERTYPE = "Kokkos::TeamPolicy<>::member_type&"
  PARFOR = "Kokkos::parallel_for"
  TEAMFORALL = PARFOR & "(*#, " & LAMBDA & " (const " & MEMBERTYPE & " team) { #(team); })"
const 
  THREADVECTORRANGE = "Kokkos::ThreadVectorRange(#, #, #)"
  FOREACH = PARFOR & "(" & THREADVECTORRANGE & ", [&] (int idx) { #(idx); })"

type ThreadTeam {.importcpp: "Kokkos::TeamPolicy<>::member_type", kokkos.} = object

type TeamPolicy {.importcpp: "Kokkos::TeamPolicy<>", kokkos.} = object

type
  NimRangeKernel = proc(idx: cint) {.cdecl.}
  NimTeamKernel = proc(team: ThreadTeam) {.cdecl.}

#[ parallel dispatch policy wrappers ]#

proc newTeamPolicy(leagueSize, teamSize: int): ptr TeamPolicy 
  {.importcpp: "new Kokkos::TeamPolicy<>(#, #)", constructor, kokkos.}

proc deleteTeamPolicy(policy: ptr TeamPolicy) {.importcpp: "delete #", kokkos.}

#[ parallel dispatch wrappers ]#

proc forall(policy: ptr TeamPolicy; body: NimTeamKernel) 
  {.importcpp: TEAMFORALL, kokkos.}

proc foreach(team: ThreadTeam; lower, upper: int; body: NimRangeKernel)
  {.importcpp: FOREACH, kokkos.}

proc teamForAll(leagueSize, teamSize: int; body: NimTeamKernel) {.inline.} =
  let policy = newTeamPolicy(leagueSize, teamSize)
  policy.forall: body
  deleteTeamPolicy(policy)

proc threadForEach(
  team: ThreadTeam; 
  lower, upper: int; 
  body: NimRangeKernel
) {.inline.} = team.foreach(lower, upper): body

#[ thread team methods ]#

proc teamSize*(team: ThreadTeam): int {.importcpp: "#.team_size()", inline, kokkos.}
  ## gets team size

proc leagueRank*(team: ThreadTeam): int 
  {.importcpp: "#.league_rank()", inline, kokkos.}
  ## gets league rank of team member

proc teamRank*(team: ThreadTeam): int {.importcpp: "#.team_rank()", inline, kokkos.}
  ## gets league rank of team member

proc myRank*(team: ThreadTeam): int = 
  ## gets league rank of team member
  team.teamRank() + team.leagueRank() * team.teamSize()

proc wait*(team: ThreadTeam) {.importcpp: "#.team_barrier()", kokkos.}
  ## setups up team barrier

template threadMain*(thread: ThreadTeam; body: untyped): untyped =
  ## ensures that only main thread executes
  if thread.myRank() == 0: body

#[ parallel dispatch ]#

template forevery*(lower, upper: SomeInteger; n, body: untyped): untyped =
  ## Threaded + vectorized for loop consturct
  ## 
  ## Executes a `for` loop of the form:
  ## ```
  ## forevery lo, hi, n: <body>
  ## ```
  ## as a threaded + vectorized loop on CPU/GPU.
  teamForAll(numThreads(), 1):
    proc(localTeamHandle: ThreadTeam) {.cdecl.} = 
      let segment = (upper - lower) div numThreads()
      let tlo = localTeamHandle.myRank()*segment
      let thi = tlo + segment
      localTeamHandle.threadForEach(tlo, thi):
        proc(idx: cint) {.cdecl.} =
          let n = idx
          body
  localBarrier()

macro every*(x: ForLoopStmt): untyped =
  ## Threaded + vectorized for loop consturct
  ## 
  ## Turns a `for` loop of the form:
  ## ```
  ## for i in every 0..10: <body>
  ## ```
  ## into a threaded + vectorized loop on CPU/GPU. Behaves like `Grid`'s
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

when isMainModule:
  initKokkos()

  let (lo, hi) = (0, numThreads()*10)
  var testSeq = newSeq[float](hi)
  
  for i in every lo..<hi: testSeq[i] += 1
  
  for i in lo..<hi: 
    assert testSeq[i] == 1.0, "testSeq[" & $i & "] = " & $testSeq[i]

  echo "for every macro test passed."

  finalizeKokkos()

  echo "kokkosdispatch.nim tests passed."