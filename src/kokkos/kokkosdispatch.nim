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

import std/[macros]
import std/[math]

import kokkosbase

Kokkos: discard

const LAMBDA = "KOKKOS_LAMBDA" 
const 
  MEMBERTYPE = "Kokkos::TeamPolicy<>::member_type&"
  PARFOR = "Kokkos::parallel_for"
  REDFOR = "Kokkos::parallel_reduce"
  TEAMFORALL = PARFOR & "(*#, " & LAMBDA & " (const " & MEMBERTYPE & " team) { #(team); })"
  TEAMREDALL = REDFOR & "(*#, " & LAMBDA & " (const " & MEMBERTYPE & " team, auto& result) { #(team, &result); }, *#)"
const 
  THREADVECTORRANGE = "Kokkos::ThreadVectorRange(#, #, #)"
  FOREACH = PARFOR & "(" & THREADVECTORRANGE & ", [&] (int idx) { #(idx); })"
  REDEACH = REDFOR & "(" & THREADVECTORRANGE & ", [&] (int idx, auto& result) { #(idx, &result); }, *#)"

type ThreadTeam {.importcpp: "Kokkos::TeamPolicy<>::member_type", kokkos.} = object

type TeamPolicy {.importcpp: "Kokkos::TeamPolicy<>", kokkos.} = object

type
  NimRangeKernel = proc(idx: cint) {.cdecl.}
  NimTeamKernel = proc(team: ThreadTeam) {.cdecl.}

type
  NimRangeReducerInt32Kernel = proc(idx: cint, result: ptr cint) {.cdecl.}
  NimRangeReducerInt64Kernel = proc(idx: cint, result: ptr clonglong) {.cdecl.}
  NimRangeReducerFloat32Kernel = proc(idx: cint, result: ptr cfloat) {.cdecl.}
  NimRangeReducerFloat64Kernel = proc(idx: cint, result: ptr cdouble) {.cdecl.}

type
  NimTeamReducerInt32Kernel = proc(team: ThreadTeam, result: ptr cint) {.cdecl.}
  NimTeamReducerInt64Kernel = proc(team: ThreadTeam, result: ptr clonglong) {.cdecl.}
  NimTeamReducerFloat32Kernel = proc(team: ThreadTeam, result: ptr cfloat) {.cdecl.}
  NimTeamReducerFloat64Kernel = proc(team: ThreadTeam, result: ptr cdouble) {.cdecl.}

#[ parallel dispatch policy wrappers ]#

proc newTeamPolicy(leagueSize, teamSize: int): ptr TeamPolicy 
  {.importcpp: "new Kokkos::TeamPolicy<>(#, #)", constructor, kokkos.}

proc deleteTeamPolicy(policy: ptr TeamPolicy) {.importcpp: "delete #", kokkos.}

#[ parallel dispatch wrappers ]#

proc forall(policy: ptr TeamPolicy; body: NimTeamKernel) 
  {.importcpp: TEAMFORALL, kokkos.}

proc foreach(team: ThreadTeam; lower, upper: int; body: NimRangeKernel)
  {.importcpp: FOREACH, kokkos.}

proc redalli32(
  policy: ptr TeamPolicy; 
  body: NimTeamReducerInt32Kernel; 
  result: ptr cint
) {.importcpp: TEAMREDALL, kokkos.}

proc redalli64(
  policy: ptr TeamPolicy; 
  body: NimTeamReducerInt64Kernel; 
  result: ptr clonglong
) {.importcpp: TEAMREDALL, kokkos.}

proc redallf32(
  policy: ptr TeamPolicy; 
  body: NimTeamReducerFloat32Kernel; 
  result: ptr cfloat
) {.importcpp: TEAMREDALL, kokkos.}

proc redallf64(
  policy: ptr TeamPolicy; 
  body: NimTeamReducerFloat64Kernel; 
  result: ptr cdouble
) {.importcpp: TEAMREDALL, kokkos.}

proc redeachi32(
  team: ThreadTeam; 
  lower, upper: int; 
  body: NimRangeReducerInt32Kernel; 
  result: ptr cint
) {.importcpp: REDEACH, kokkos.}

proc redeachi64(
  team: ThreadTeam; 
  lower, upper: int; 
  body: NimRangeReducerInt64Kernel; 
  result: ptr clonglong
) {.importcpp: REDEACH, kokkos.}

proc redeachf32(
  team: ThreadTeam; 
  lower, upper: int; 
  body: NimRangeReducerFloat32Kernel; 
  result: ptr cfloat
) {.importcpp: REDEACH, kokkos.}

proc redeachf64(
  team: ThreadTeam; 
  lower, upper: int; 
  body: NimRangeReducerFloat64Kernel; 
  result: ptr cdouble
) {.importcpp: REDEACH, kokkos.}

proc teamForAll(leagueSize, teamSize: int; body: NimTeamKernel) {.inline.} =
  let policy = newTeamPolicy(leagueSize, teamSize)
  policy.forall: body
  deleteTeamPolicy(policy)

proc threadForEach(
  team: ThreadTeam; 
  lower, upper: int; 
  body: NimRangeKernel
) {.inline.} = team.foreach(lower, upper): body

proc teamRedAllInt32(
  leagueSize, teamSize: int; 
  body: NimTeamReducerInt32Kernel; 
  result: ptr cint
) {.inline.} =
  let policy = newTeamPolicy(leagueSize, teamSize)
  policy.redalli32(body, result)
  deleteTeamPolicy(policy)

proc teamRedAllInt64(
  leagueSize, teamSize: int; 
  body: NimTeamReducerInt64Kernel; 
  result: ptr clonglong
) {.inline.} =
  let policy = newTeamPolicy(leagueSize, teamSize)
  policy.redalli64(body, result)
  deleteTeamPolicy(policy)

proc teamRedAllFloat32(
  leagueSize, teamSize: int; 
  body: NimTeamReducerFloat32Kernel; 
  result: ptr cfloat
) {.inline.} =
  let policy = newTeamPolicy(leagueSize, teamSize)
  policy.redallf32(body, result)
  deleteTeamPolicy(policy)

proc teamRedAllFloat64(
  leagueSize, teamSize: int; 
  body: NimTeamReducerFloat64Kernel; 
  result: ptr cdouble
) {.inline.} =
  let policy = newTeamPolicy(leagueSize, teamSize)
  policy.redallf64(body, result)
  deleteTeamPolicy(policy)

proc teamRedAll[T]( 
  leagueSize, teamSize: int; 
  body: auto; 
  result: ptr T
) {.inline.} =
  when T is int32:
    teamRedAllInt32(leagueSize, teamSize, cast[NimTeamReducerInt32Kernel](body), cast[ptr cint](result))
  elif T is int64:
    teamRedAllInt64(leagueSize, teamSize, cast[NimTeamReducerInt64Kernel](body), cast[ptr clonglong](result))
  elif T is float32:
    teamRedAllFloat32(leagueSize, teamSize, cast[NimTeamReducerFloat32Kernel](body), cast[ptr cfloat](result))
  elif T is float64:
    teamRedAllFloat64(leagueSize, teamSize, cast[NimTeamReducerFloat64Kernel](body), cast[ptr cdouble](result))
  else: 
    {.error: "Unsupported type for teamRedAll".}

proc teamRedEachInt32(
  team: ThreadTeam; 
  lower, upper: int; 
  body: NimRangeReducerInt32Kernel; 
  result: ptr cint
) {.inline.} = team.redeachi32(lower, upper, body, result)

proc teamRedEachInt64(
  team: ThreadTeam; 
  lower, upper: int; 
  body: NimRangeReducerInt64Kernel; 
  result: ptr clonglong
) {.inline.} = team.redeachi64(lower, upper, body, result)

proc teamRedEachFloat32(
  team: ThreadTeam; 
  lower, upper: int; 
  body: NimRangeReducerFloat32Kernel; 
  result: ptr cfloat
) {.inline.} = team.redeachf32(lower, upper, body, result)

proc teamRedEachFloat64(
  team: ThreadTeam; 
  lower, upper: int; 
  body: NimRangeReducerFloat64Kernel; 
  result: ptr cdouble
) {.inline.} = team.redeachf64(lower, upper, body, result)

proc teamRedEach[T](
  team: ThreadTeam; 
  lower, upper: int; 
  body: auto; 
  result: ptr T
) {.inline.} =
  when T is int32:
    teamRedEachInt32(team, lower, upper, cast[NimRangeReducerInt32Kernel](body), cast[ptr cint](result))
  elif T is int64:
    teamRedEachInt64(team, lower, upper, cast[NimRangeReducerInt64Kernel](body), cast[ptr clonglong](result))
  elif T is float32:
    teamRedEachFloat32(team, lower, upper, cast[NimRangeReducerFloat32Kernel](body), cast[ptr cfloat](result))
  elif T is float64:
    teamRedEachFloat64(team, lower, upper, cast[NimRangeReducerFloat64Kernel](body), cast[ptr cdouble](result))
  else: 
    {.error: "Unsupported type for teamRedEach".}

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

template sum*[T](lower, upper: SomeInteger; n, body: untyped): untyped =
  ## Threaded + vectorized sum reduction construct
  ## 
  ## Executes a `for` loop of the form:
  ## ```
  ## sum lo, hi, n: <body>
  ## ```
  ## as a threaded + vectorized sum reduction on CPU/GPU.
  var result: T
  when T is SomeNumber:
    result = T(0)
  else:
    result = T()
  teamRedAll[T](
    numThreads(), 
    1, 
    proc(localTeamHandle: ThreadTeam, teamResult: ptr T) {.cdecl.} = 
      # Each thread gets a portion of the range
      let totalRange = int(upper - lower)
      let segment = totalRange div numThreads()
      let tlo = int(lower) + localTeamHandle.myRank() * segment
      let thi = if localTeamHandle.myRank() == numThreads() - 1: int(upper) 
                else: tlo + segment
      
      # Use vectorized reduction for this thread's range
      var localSum: T
      when T is SomeNumber: localSum = T(0)
      else: localSum = T()
      teamRedEach[T](localTeamHandle, tlo, thi, 
        proc(idx: int, contribution: ptr T) {.cdecl.} =
          let `n` = idx
          contribution[] += body, 
        addr localSum
      )
      
      # teamRedAll automatically accumulates localSum from all threads
      teamResult[] += localSum, 
    addr result
  )
  result

template sum*[T](iters: SomeInteger; n, body: untyped): untyped =
  sum[T](0, iters, n, body)

when isMainModule:
  initKokkos()

  let (lo, hi) = (0, numThreads()*10)
  var testSeq = newSeq[float](hi)
  
  for i in every lo..<hi: testSeq[i] += 1
  
  for i in lo..<hi: 
    assert testSeq[i] == 1.0, "testSeq[" & $i & "] = " & $testSeq[i]

  echo "for every macro test passed."

  # Test sum reduction
  let sumResult = sum[float](lo, hi, n): float(n)  # Use n to match template parameter
  let expectedSum = float(hi*(hi-1)/2 - lo*(lo-1)/2)  # Sum of integers from lo to hi-1
  assert abs(sumResult - expectedSum) < 1e-6, "Sum test failed: got " & $sumResult & ", expected " & $expectedSum

  echo "sum reduction test passed."

  finalizeKokkos()

  echo "kokkosdispatch.nim tests passed."