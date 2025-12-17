/*
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/kokkos/kokkostypes.hpp
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
*/

#include <Kokkos_Core.hpp>

using Layout = Kokkos::LayoutLeft;
using Memory = Kokkos::HostSpace;
using Traits = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

#define KOKKOS_VIEW1 Kokkos::View<T*, Layout, Memory, Traits>
#define KOKKOS_VIEW2 Kokkos::View<T**, Layout, Memory, Traits>
#define KOKKOS_VIEW3 Kokkos::View<T***, Layout, Memory, Traits>
#define KOKKOS_VIEW4 Kokkos::View<T****, Layout, Memory, Traits>
#define KOKKOS_VIEW5 Kokkos::View<T*****, Layout, Memory, Traits>
#define KOKKOS_VIEW6 Kokkos::View<T******, Layout, Memory, Traits>
#define KOKKOS_VIEW7 Kokkos::View<T*******, Layout, Memory, Traits>

#define CONST_ARGS1 data, dims[0]
#define CONST_ARGS2 data, dims[0], dims[1]
#define CONST_ARGS3 data, dims[0], dims[1], dims[2]
#define CONST_ARGS4 data, dims[0], dims[1], dims[2], dims[3]
#define CONST_ARGS5 data, dims[0], dims[1], dims[2], dims[3], dims[4]
#define CONST_ARGS6 data, dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]

#define ACCSS_ARGS1 idxs[0]
#define ACCSS_ARGS2 idxs[0], idxs[1]
#define ACCSS_ARGS3 idxs[0], idxs[1], idxs[2]
#define ACCSS_ARGS4 idxs[0], idxs[1], idxs[2], idxs[3]
#define ACCSS_ARGS5 idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]
#define ACCSS_ARGS6 idxs[0], idxs[1], idxs[2], idxs[3], idxs[4], idxs[5]

//
// constructions/destruction
//

template<typename T>
void* create_kokkos_view(T* data, size_t rank, const size_t* dims) {  
  /**
   * @brief Create a Kokkos View of given rank and dimensions.
   * @param data Pointer to the data.
   * @param rank Rank of the Kokkos View (1 to 6).
   * @param dims Array of dimensions for each rank.
   * @return Pointer to the created Kokkos View.
   */
  switch(rank) {
    case 1: return static_cast<void*>(new KOKKOS_VIEW1(CONST_ARGS1)); break;
    case 2: return static_cast<void*>(new KOKKOS_VIEW2(CONST_ARGS2)); break;
    case 3: return static_cast<void*>(new KOKKOS_VIEW3(CONST_ARGS3)); break;
    case 4: return static_cast<void*>(new KOKKOS_VIEW4(CONST_ARGS4)); break;
    case 5: return static_cast<void*>(new KOKKOS_VIEW5(CONST_ARGS5)); break;
    case 6: return static_cast<void*>(new KOKKOS_VIEW6(CONST_ARGS6)); break;
    default: return nullptr;
} }

template<typename T>
void destroy_kokkos_view(void* handle, size_t rank) {
  /**
   * @brief Destroy a Kokkos View of given rank.
   * @param handle Pointer to the Kokkos View to be destroyed.
   * @param rank Rank of the Kokkos View (1 to 6).
   */
  switch(rank) {
    case 1: delete static_cast<KOKKOS_VIEW1*>(handle); break;
    case 2: delete static_cast<KOKKOS_VIEW2*>(handle); break;
    case 3: delete static_cast<KOKKOS_VIEW3*>(handle); break;
    case 4: delete static_cast<KOKKOS_VIEW4*>(handle); break;
    case 5: delete static_cast<KOKKOS_VIEW5*>(handle); break;
    case 6: delete static_cast<KOKKOS_VIEW6*>(handle); break;
    default: break;
} }

//
// accessors
//

template<typename T>
T view_get(void* handle, size_t rank, const size_t* idxs) {
  /**
   * @brief Get an element from a Kokkos View of given rank.
   * @param handle Pointer to the Kokkos View.
   * @param rank Rank of the Kokkos View (1 to 6).
   * @param idxs Array of indices for each rank.
   * @return The element at the specified indices.
   */
  switch(rank) {
    case 1: return (*static_cast<KOKKOS_VIEW1*>(handle))(ACCSS_ARGS1);
    case 2: return (*static_cast<KOKKOS_VIEW2*>(handle))(ACCSS_ARGS2);
    case 3: return (*static_cast<KOKKOS_VIEW3*>(handle))(ACCSS_ARGS3);
    case 4: return (*static_cast<KOKKOS_VIEW4*>(handle))(ACCSS_ARGS4);
    case 5: return (*static_cast<KOKKOS_VIEW5*>(handle))(ACCSS_ARGS5);
    case 6: return (*static_cast<KOKKOS_VIEW6*>(handle))(ACCSS_ARGS6);
    default: return T{};
} }

template<typename T>
void view_set(void* handle, size_t rank, const size_t* idxs, T value) {
  /**
   * @brief Set an element in a Kokkos View of given rank.
   * @param handle Pointer to the Kokkos View.
   * @param rank Rank of the Kokkos View (1 to 6).
   * @param idxs Array of indices for each rank.
   * @param value The value to set at the specified indices.
   */
  switch(rank) {
    case 1: (*static_cast<KOKKOS_VIEW1*>(handle))(ACCSS_ARGS1) = value; break;
    case 2: (*static_cast<KOKKOS_VIEW2*>(handle))(ACCSS_ARGS2) = value; break;
    case 3: (*static_cast<KOKKOS_VIEW3*>(handle))(ACCSS_ARGS3) = value; break;
    case 4: (*static_cast<KOKKOS_VIEW4*>(handle))(ACCSS_ARGS4) = value; break;
    case 5: (*static_cast<KOKKOS_VIEW5*>(handle))(ACCSS_ARGS5) = value; break;
    case 6: (*static_cast<KOKKOS_VIEW6*>(handle))(ACCSS_ARGS6) = value; break;
} }