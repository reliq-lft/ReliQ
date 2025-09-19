/**
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/kokkos/kokkos.hpp
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
 */

#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_DynRankView.hpp>
#include <Kokkos_SIMD.hpp>

#ifndef RELIQ_KOKKOS_HPP
#define RELIQ_KOKKOS_HPP

// view wrappers
namespace Views {
  // wrap Kokkos View; called "static" bc the rank is fixed at compile time
  template <typename T>
  using StaticView = Kokkos::View<T*>;

  // wrap Kokkos DynRankView; called "dynamic" bc the rank is determined at runtime
  template <typename T>
  using DynamicView = Kokkos::DynRankView<T>;
}

// wrappers for parallel_for constructs
namespace Distpatch {
  // some typedefs
  typedef Kokkos::TeamPolicy<>::member_type TeamMember;
  typedef void (*NimKernel)(const TeamMember&);

  // get team rank of team member
  extern "C" inline int rank(const TeamMember& member) { return member.league_rank(); }

  // team barrier
  extern "C" inline void barrier(const TeamMember& member) { member.team_barrier(); }

  // team parallel for wrapper; OpenMP analogue of parallel region
  extern "C" void team_parallel_for(Kokkos::TeamPolicy<> policy, NimKernel proc) {
    Kokkos::parallel_for(
      "team_parallel_for_wrapper", 
      policy,
      KOKKOS_LAMBDA (const TeamMember& member) { 
        // v--- maybe import this???
        //extern "C" int trank(const TeamMember& member) { return member.league_rank(); }
        std::cout << "team_parallel_for thread " << member.league_rank() << "\n";
        std::cout << "team_parallel_for thread: " << rank(member) << "\n";
        (*proc)(member); }
    );
  }
}

#endif // RELIQ_KOKKOS_HPP