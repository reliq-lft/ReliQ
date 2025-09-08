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

/* view wrappers */

// wrap Kokkos View; called "static" bc the rank is fixed at compile time
template <class T>
using StaticView = Kokkos::View<T*>;

// wrap Kokkos DynRankView; called "dynamic" bc the rank is determined at runtime
template <class T>
using DynamicView = Kokkos::DynRankView<T>;

/* dispatch wrappers */

// typedef for C function pointer to be passed to Kokkos parallel_for
typedef void (*NimProc)(int);

// wrapper for Kokkos parallel_for
inline void parallel_for_range(int start, int stop, NimProc proc) {
  auto execPolicy = Kokkos::RangePolicy<>(start, stop);
  Kokkos::parallel_for(execPolicy, KOKKOS_LAMBDA (const int n) { proc(n); });
}

/* simd wrappers */

#endif // RELIQ_KOKKOS_HPP