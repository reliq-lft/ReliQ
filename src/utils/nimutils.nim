#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/utils/nimutils.nim
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

import std/[cmdline, strutils]

#[ backend: constants ]#

const 
  printBreak* = "\n...............................................................\n"

#[ backend: implementation of helper procedures ]#

iterator dimensions*[T](arr: openArray[T]): int {.inline.} =
  # silly helper iterator for iterating through openArray indices
  for mu in 0..<arr.len: yield mu

# concatenate strings with forward slash between
proc `/`*(sa, sb: string): string = sa & "/" & sb

# concatenate strings with space between
proc `+`*(sa, sb: string): string = sa & " " & sb

# concatenate strings with new line
proc `&&`*(sa, sb: string): string = sa & "\n" & sb

# append to string with new line
proc `&&=`*(sa: var string, sb: string) = sa = sa && sb

# append string with space
proc `+=`*(sa: var string, sb: string) = sa = sa + sb

# division of a integer arrays
proc `/`*[A: SomeInteger, B: SomeInteger](a: openArray[A], b: openArray[B]): auto =
  assert(a.len == b.len, "array division: arrays must have same length")
  result = newSeq[type(a[0])](a.len)
  for mu in a.dimensions: result[mu] = a[mu] div b[mu]

iterator reversedDimensions*[T](arr: openArray[T], start: int = 0): int {.inline.} =
  # silly helper iterator for iteration through openArray indices in reverse
  for mu in countdown(arr.len - (start + 1), 0): yield mu

proc any*(args: varargs[bool]): bool = 
  # returns true if any arg is true
  for arg in args: 
    if arg: return true
  return false

proc all*(args: varargs[bool]): bool = 
  # returns true if all args are true
  for arg in args: 
    if not arg: return false
  return true

proc ones*(nd: int; T: typedesc): seq[T] {.inline.} =
  # constructs a sequence of ones
  result = newSeq[T](nd)
  for mu in result.dimensions: result[mu] = T(1)

# checks if integer "a" divides integer "b"
proc divides*(a, b: SomeInteger): bool {.inline.} = b mod a == 0

proc dividesSomeElement*[A: SomeInteger, B: SomeInteger](
  a: A, 
  bs: seq[B]
): bool {.inline.} =
  # checks if integer "a" divides any integer in "bs"
  for b in bs: 
    if divides(a, b): return true
  return false

proc toSeq*[I](arr: openArray[I]; T: typedesc): seq[T] {.inline.} =
  # converts open array to sequence
  result = newSeq[T](arr.len)
  for mu in arr.dimensions: result[mu] = T(arr[mu])
proc toSeq*[I](arr: openArray[I]): auto {.inline.} =
  assert(arr.len > 0, "toSeq: openArray must have at least one element")
  return arr.toSeq(typeof(arr[0]))

proc product*[T](arr: openArray[T]): T {.inline.} =
  # calculates product of open array's entries
  result = T(1)
  for el in arr: result *= el

# remove "@" in sequence printout for cleaner output
proc `$`*(sq: seq[SomeInteger]): string = "[" & sq.join(", ") & "]"

# remove "@" in sequence printout for cleaner output
proc `$`*(sq: seq[SomeNumber]): string = "[" & sq.join(", ") & "]"
