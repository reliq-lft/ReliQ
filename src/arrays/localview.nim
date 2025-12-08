#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/arrays/localview.nim
  Contact: reliq-lft@proton.me

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

import reliq

import globalarrays/[gabase]
import kokkos/[kokkosbase]

import globalarray

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
      auto* view = new Kokkos::View<T**, Layout, Memory, Traits>(
        data, dims[0], dims[1]
      );
      return static_cast<void*>(view);
    }
    case 3: {
      auto* view = new Kokkos::View<T***, Layout, Memory, Traits>(
        data, dims[0], dims[1], dims[2]
      );
      return static_cast<void*>(view);
    }
    case 4: {
      auto* view = new Kokkos::View<T****, Layout, Memory, Traits>(
        data, dims[0], dims[1], dims[2], dims[3]
      );
      return static_cast<void*>(view);
    }
    case 5: {
      auto* view = new Kokkos::View<T*****, Layout, Memory, Traits>(
        data, dims[0], dims[1], dims[2], dims[3], dims[4]
      );
      return static_cast<void*>(view);
    }
    case 6: {
      auto* view = new Kokkos::View<T******, Layout, Memory, Traits>(
        data, dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
      );
      return static_cast<void*>(view);
    }
    case 7: {
      auto* view = new Kokkos::View<T*******, Layout, Memory, Traits>(
        data, dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]
      );
      return static_cast<void*>(view);
    }
    default: return nullptr;
} }

template<typename T>
void destroy_kokkos_view(void* handle, size_t rank) {
  using Layout = Kokkos::LayoutLeft;
  using Memory = Kokkos::HostSpace;
  using Traits = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  
  switch(rank) {
    case 1: 
      delete static_cast<Kokkos::View<T*, Layout, Memory, Traits>*>(handle); 
      break;
    case 2: 
      delete static_cast<Kokkos::View<T**, Layout, Memory, Traits>*>(handle); 
      break;
    case 3: 
      delete static_cast<Kokkos::View<T***, Layout, Memory, Traits>*>(handle); 
      break;
    case 4: 
      delete static_cast<Kokkos::View<T****, Layout, Memory, Traits>*>(handle); 
      break;
    case 5: 
      delete static_cast<Kokkos::View<T*****, Layout, Memory, Traits>*>(handle); 
      break;
    case 6: 
      delete static_cast<Kokkos::View<T******, Layout, Memory, Traits>*>(handle); 
      break;
    case 7: 
      delete static_cast<Kokkos::View<T*******, Layout, Memory, Traits>*>(handle); 
      break;
    default: break;
} }
""".}

proc createKokkosViewInt(data: pointer, rank: csize_t, dims: ptr csize_t): pointer 
  {.importc: "create_kokkos_view<int>", nodecl.}

proc createKokkosViewFloat(data: pointer, rank: csize_t, dims: ptr csize_t): pointer 
  {.importc: "create_kokkos_view<float>", nodecl.}

proc createKokkosViewDouble(data: pointer, rank: csize_t, dims: ptr csize_t): pointer 
  {.importc: "create_kokkos_view<double>", nodecl.}

proc destroyKokkosViewInt(handle: pointer, rank: csize_t) 
  {.importc: "destroy_kokkos_view<int>", nodecl.}

proc destroyKokkosViewFloat(handle: pointer, rank: csize_t) 
  {.importc: "destroy_kokkos_view<float>", nodecl.}

proc destroyKokkosViewDouble(handle: pointer, rank: csize_t) 
  {.importc: "destroy_kokkos_view<double>", nodecl.}

#[ local view constructor, destructor, copy assignment ]#

proc newLocalView*[D: static[int], T](local: LocalData[D, T]): LocalView[D, T] =
  ## Create a LocalView wrapping a LocalData with a Kokkos View
  ##
  ## Parameters:
  ## - `local`: LocalData from a GlobalArray
  ##
  ## Returns:
  ## LocalView with Kokkos View for efficient computation
  ## 
  ## Example:
  ## ```nim
  ## let localData = downcast(globalArray)
  ## let localView = newLocalView(localData)
  ## ```
  var dims: array[D, csize_t]
  
  for i in 0..<D: dims[i] = csize_t(local.hi[i] - local.lo[i] + 1)

  let pdata = cast[pointer](local.data)
  let pdims = addr dims[0]
  when T is int:
    let handle = createKokkosViewInt(pdata, csize_t(D), pdims)
  elif T is float32:
    let handle = createKokkosViewFloat(pdata, csize_t(D), pdims)
  elif T is float64:
    let handle = createKokkosViewDouble(pdata, csize_t(D), pdims)
  
  return LocalView[D, T](handle: handle, rank: D)

proc newLocalView*[D: static[int], T](ga: GlobalArray[D, T]): LocalView[D, T] =
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
  ## let localView = newLocalView(globalArray)
  ## ```
  let local = downcast(ga)
  return newLocalView(local)

proc `=destroy`*[D: static[int], T](view: LocalView[D, T]) =
  ## Destructor - cleans up Kokkos View handle
  if view.handle != nil:
    when T is int: destroyKokkosViewInt(view.handle, csize_t(view.rank))
    elif T is float: destroyKokkosViewFloat(view.handle, csize_t(view.rank))
    elif T is float64: destroyKokkosViewDouble(view.handle, csize_t(view.rank))

proc `=copy`*[D: static[int], T](dest: var LocalView[D, T], src: LocalView[D, T]) {.error.}
  ## Prevent copying of LocalView

#[ unit tests ]#

test:
  let lattice = [8, 8, 8, 8*numRanks()]
  let mpigrid = [1, 1, 1, numRanks()]
  let ghostgrid = [1, 1, 1, 1]
  
  # indirect

  var testGA = newGlobalArray(lattice, mpigrid, ghostgrid): float
  
  assert(testGA.isInitialized(), "GlobalArray initialization failed")
  
  let local = downcast(testGA)
  
  assert(local.getHandle() == testGA.getHandle(), "LocalData handle mismatch")
  
  let view = newLocalView(local)
  
  assert(view.handle != nil, "Kokkos View handle is null")
  assert(view.rank == 4, "View rank mismatch")
  
  echo "LocalView test passed: created ", view.rank, "D view with handle ", cast[uint](view.handle)

  # direct

  var testGA2 = newGlobalArray(lattice, mpigrid, ghostgrid): float64

  assert(testGA2.isInitialized(), "GlobalArray initialization failed")

  let view2 = newLocalView(testGA2)

  assert(view2.handle != nil, "Kokkos View handle is null")
  assert(view2.rank == 4, "View rank mismatch")

  echo "LocalView test passed: created ", view2.rank, "D view with handle ", cast[uint](view2.handle)