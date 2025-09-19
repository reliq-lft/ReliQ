#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/kokkos/kokkosthreads.nim
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

type
  TeamPolicy {.importcpp: "Kokkos::TeamPolicy<>", kokkos.} = object
  TeamMember {.importcpp: "Kokkos::TeamPolicy<>::member_type", kokkos.} = object
  TeamThreadRange {.importcpp: "Kokkos::TeamThreadRange", kokkos.} = object
  TeamVectorRange {.importcpp: "Kokkos::TeamVectorRange", kokkos.} = object

type NimKernel = proc(team: TeamMember) {.cdecl.}

# C++ new/delete bindings for TeamPolicy
proc newTeamPolicy*(leagueSize, teamSize: int): ptr TeamPolicy 
  {.importcpp: "new Kokkos::TeamPolicy<>(#, #)", constructor, kokkos.}
proc deleteTeamPolicy*(policy: ptr TeamPolicy) 
  {.importcpp: "delete #", kokkos.}

# gets league rank of team member
proc myRank*(team: TeamMember): int {.importcpp: "#.league_rank()", inline, kokkos.}

# setups up team barrier
proc wait*(team: TeamMember) {.importcpp: "#.team_barrier()", kokkos_wrapper.}

# backend: team parallel for wrapper 
const 
  MEMBERTYPE = "Kokkos::TeamPolicy<>::member_type&"
  FORALL = "Kokkos::parallel_for(*#, KOKKOS_LAMBDA (const" + MEMBERTYPE +  "m) { #(m); })"
proc forall(policy: ptr TeamPolicy; body: NimKernel) {.importcpp: FORALL, kokkos.}

# frontend: team parallel for wrapper
proc forall*(leagueSize, teamSize: int; body: NimKernel) {.inline.} =
  let policy = newTeamPolicy(leagueSize, teamSize)
  policy.forall(body)
  deleteTeamPolicy(policy)

# threads
template threads*(leagueSize, teamSize: int; body: untyped): untyped =
  let kernel: NimKernel = proc(thread: TeamMember) {.cdecl.} = 
    body(thread)
  forall(leagueSize, teamSize, kernel)

# lessons learned:
# * nested procedures that are to be passed to a call expecting a specific 
#   procedure kind must be declared with that procedure kind explicitly
when isMainModule:
  import runtime
  reliq:
    threads(numThreads(), 1) do (thread: TeamMember):
      let rank = thread.myRank()
      echo "Hello from team member with league rank: ", rank
      thread.wait()
      echo "Goodbye from team member with league rank: ", rank