#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/types/localview.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
]#

import std/[complex]

import reliq

import globalarrays/[gabase]
import kokkos/[kokkosbase]

import globalarrays/gatypes

GlobalArrays: discard
Kokkos: discard

# Kokkos View wrapper - opaque handle
type KokkosHandle = pointer

type LocalView*[D: static[int], T] = object
  ## Wrapper around Kokkos View for local portion of GlobalArray
  ##
  ## Holds both the LocalData (which manages GA access) and a Kokkos View
  ## that provides efficient access for computations.
  handle: KokkosHandle
  rank: int
  dims: array[D, int]
  ghostGrid*: array[D, int]
  localData: LocalData[D, T]  # Keep LocalData alive for the lifetime of LocalView

type ComplexLocalView*[D: static[int], F] = object
  ## Wrapper for complex field views - stores real and imaginary parts separately
  re*: LocalView[D, F]
  im*: LocalView[D, F]

#[ C++ helpers for dealing with Kokkos semantics ]#

{.emit: """
#include <Kokkos_Core.hpp>

template<typename T>
void* create_kokkos_view(T* data, size_t rank, const size_t* dims) {
  using Layout = Kokkos::LayoutLeft;
  using Memory = Kokkos::HostSpace;
  using Traits = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  
  switch(rank) {
    case 1: {
      auto* view = new Kokkos::View<T*, Layout, Memory, Traits>(data, dims[0]);
      return static_cast<void*>(view);
    }
    case 2: {
      auto* view = new Kokkos::View<T**, Layout, Memory, Traits>(data, dims[0], dims[1]);
      return static_cast<void*>(view);
    }
    case 3: {
      auto* view = new Kokkos::View<T***, Layout, Memory, Traits>(data, dims[0], dims[1], dims[2]);
      return static_cast<void*>(view);
    }
    case 4: {
      auto* view = new Kokkos::View<T****, Layout, Memory, Traits>(data, dims[0], dims[1], dims[2], dims[3]);
      return static_cast<void*>(view);
    }
    case 5: {
      auto* view = new Kokkos::View<T*****, Layout, Memory, Traits>(data, dims[0], dims[1], dims[2], dims[3], dims[4]);
      return static_cast<void*>(view);
    }
    case 6: {
      auto* view = new Kokkos::View<T******, Layout, Memory, Traits>(data, dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
      return static_cast<void*>(view);
    }
    case 7: {
      auto* view = new Kokkos::View<T*******, Layout, Memory, Traits>(data, dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]);
      return static_cast<void*>(view);
    }
    default: return nullptr;
  }
}

template<typename T>
void destroy_kokkos_view(void* handle, size_t rank) {
  if (!handle) return;
  
  using Layout = Kokkos::LayoutLeft;
  using Memory = Kokkos::HostSpace;
  using Traits = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  
  switch(rank) {
    case 1: delete static_cast<Kokkos::View<T*, Layout, Memory, Traits>*>(handle); break;
    case 2: delete static_cast<Kokkos::View<T**, Layout, Memory, Traits>*>(handle); break;
    case 3: delete static_cast<Kokkos::View<T***, Layout, Memory, Traits>*>(handle); break;
    case 4: delete static_cast<Kokkos::View<T****, Layout, Memory, Traits>*>(handle); break;
    case 5: delete static_cast<Kokkos::View<T*****, Layout, Memory, Traits>*>(handle); break;
    case 6: delete static_cast<Kokkos::View<T******, Layout, Memory, Traits>*>(handle); break;
    case 7: delete static_cast<Kokkos::View<T*******, Layout, Memory, Traits>*>(handle); break;
  }
}

template<typename T>
T& access_kokkos_view_1d(void* handle, size_t i) {
  using Layout = Kokkos::LayoutLeft;
  using Memory = Kokkos::HostSpace;
  using Traits = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  auto& view = *static_cast<Kokkos::View<T*, Layout, Memory, Traits>*>(handle);
  return view(i);
}
""".}

proc createKokkosViewInt(data: pointer, rank: csize_t, dims: ptr csize_t): KokkosHandle 
  {.importcpp: "create_kokkos_view<int>(@)", header: "<Kokkos_Core.hpp>".}

proc createKokkosViewFloat(data: pointer, rank: csize_t, dims: ptr csize_t): KokkosHandle 
  {.importcpp: "create_kokkos_view<float>(@)", header: "<Kokkos_Core.hpp>".}

proc createKokkosViewDouble(data: pointer, rank: csize_t, dims: ptr csize_t): KokkosHandle 
  {.importcpp: "create_kokkos_view<double>(@)", header: "<Kokkos_Core.hpp>".}

proc destroyKokkosViewInt(handle: KokkosHandle, rank: csize_t) 
  {.importcpp: "destroy_kokkos_view<int>(@)", header: "<Kokkos_Core.hpp>".}

proc destroyKokkosViewFloat(handle: KokkosHandle, rank: csize_t) 
  {.importcpp: "destroy_kokkos_view<float>(@)", header: "<Kokkos_Core.hpp>".}

proc destroyKokkosViewDouble(handle: KokkosHandle, rank: csize_t) 
  {.importcpp: "destroy_kokkos_view<double>(@)", header: "<Kokkos_Core.hpp>".}

proc accessKokkosViewInt1D(handle: KokkosHandle, i: csize_t): ptr cint 
  {.importcpp: "&access_kokkos_view_1d<int>(@)", header: "<Kokkos_Core.hpp>".}

proc accessKokkosViewFloat1D(handle: KokkosHandle, i: csize_t): ptr cfloat 
  {.importcpp: "&access_kokkos_view_1d<float>(@)", header: "<Kokkos_Core.hpp>".}

proc accessKokkosViewDouble1D(handle: KokkosHandle, i: csize_t): ptr cdouble 
  {.importcpp: "&access_kokkos_view_1d<double>(@)", header: "<Kokkos_Core.hpp>".}

#[ LocalView constructors ]#

proc localView*[D: static[int], T](local: LocalData[D, T]): LocalView[D, T] =
  ## Create a LocalView from LocalData (acquired from GlobalArray downcast)
  ##
  ## This function creates a Kokkos View wrapper around the GlobalArray
  ## local data for efficient computation.
  ##
  ## Parameters:
  ## - `local`: LocalData obtained from GlobalArray downcast
  ##
  ## Returns:
  ## LocalView with Kokkos View for efficient computation
  ## 
  ## Example:
  ## ```nim
  ## let localData = downcast(globalArray)
  ## let localView = localView(localData)
  ## ```
  let ghostGrid = local.ghostGrid
  var dims: array[D, csize_t]
  var storedDims: array[D, int]
  
  for i in 0..<D:
    storedDims[i] = local.hi[i] - local.lo[i] + 1
    dims[i] = csize_t(storedDims[i] + 2*ghostGrid[i]) # ghost offset

  let pdata = cast[pointer](local.data)
  let pdims = addr dims[0]
  when T is int:
    let handle = createKokkosViewInt(pdata, csize_t(D), pdims)
  elif T is float32:
    let handle = createKokkosViewFloat(pdata, csize_t(D), pdims)
  elif T is float64:
    let handle = createKokkosViewDouble(pdata, csize_t(D), pdims)
  
  return LocalView[D, T](
    handle: handle, 
    rank: D, 
    dims: storedDims,
    ghostGrid: ghostGrid,
    localData: local  # Store LocalData to keep it alive
  )

proc localView*[D: static[int], T](ga: GlobalArray[D, T]): LocalView[D, T] =
  ## Create a LocalView directly from a GlobalArray
  ##
  ## This convenience function downcasts the GlobalArray to LocalData
  ## and then creates a Kokkos View wrapper.
  ##
  ## Parameters:
  ## - `ga`: GlobalArray to create view from
  ##
  ## Returns:
  ## LocalView with Kokkos View for efficient computation
  ## 
  ## Example:
  ## ```nim
  ## let localView = localView(globalArray)
  ## ```
  return localView(downcast(ga))

proc `=destroy`*[D: static[int], T](view: LocalView[D, T]) =
  ## Destructor - cleans up Kokkos View handle
  if view.handle != nil:
    when T is int:
      destroyKokkosViewInt(view.handle, csize_t(view.rank))
    elif T is float32:
      destroyKokkosViewFloat(view.handle, csize_t(view.rank))
    elif T is float64:
      destroyKokkosViewDouble(view.handle, csize_t(view.rank))

proc `=copy`*[D: static[int], T](dest: var LocalView[D, T], src: LocalView[D, T]) {.error.}
  ## Prevent copying of LocalView - it manages Kokkos View resources

#[ LocalView accessors ]#

proc `[]`*[D: static[int], T](view: LocalView[D, T], i: SomeInteger): T =
  ## 1D accessor for LocalView
  ##
  ## Parameters:
  ## - `view`: LocalView instance
  ## - `i`: Linear index
  ##
  ## Returns:
  ## Value at index i
  when T is int:
    return T(accessKokkosViewInt1D(view.handle, csize_t(i))[])
  elif T is float32:
    return T(accessKokkosViewFloat1D(view.handle, csize_t(i))[])
  elif T is float64:
    return T(accessKokkosViewDouble1D(view.handle, csize_t(i))[])

proc `[]=`*[D: static[int], T](view: var LocalView[D, T], i: SomeInteger, val: T) =
  ## 1D assignment for LocalView
  ##
  ## Parameters:
  ## - `view`: LocalView instance
  ## - `i`: Linear index
  ## - `val`: Value to assign
  when T is int:
    accessKokkosViewInt1D(view.handle, csize_t(i))[] = cint(val)
  elif T is float32:
    accessKokkosViewFloat1D(view.handle, csize_t(i))[] = cfloat(val)
  elif T is float64:
    accessKokkosViewDouble1D(view.handle, csize_t(i))[] = cdouble(val)

#[ ComplexLocalView accessors ]#

proc `[]`*[D: static[int], F](view: ComplexLocalView[D, F], i: SomeInteger): Complex[F] =
  ## 1D accessor for ComplexLocalView
  ##
  ## Parameters:
  ## - `view`: ComplexLocalView instance  
  ## - `i`: Linear index
  ##
  ## Returns:
  ## Complex value at index i
  return complex(view.re[i], view.im[i])

proc `[]=`*[D: static[int], F](view: var ComplexLocalView[D, F], i: SomeInteger, val: Complex[F]) =
  ## 1D assignment for ComplexLocalView
  ##
  ## Parameters:
  ## - `view`: ComplexLocalView instance
  ## - `i`: Linear index  
  ## - `val`: Complex value to assign
  view.re[i] = val.re
  view.im[i] = val.im

proc `[]=`*[D: static[int], F](view: var ComplexLocalView[D, F], i: SomeInteger, val: SomeNumber) =
  ## 1D assignment for ComplexLocalView from real number
  ##
  ## Parameters:
  ## - `view`: ComplexLocalView instance
  ## - `i`: Linear index  
  ## - `val`: Real value to assign (imaginary part = 0)
  view.re[i] = F(val)
  view.im[i] = F(0.0)