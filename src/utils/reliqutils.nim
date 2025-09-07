#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/utils/reliqutils.nim
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

import std/[macros]
import upcxx/[upcxxbase]

# macro that implements Nim "echo"; written for the fun of it, tbh
macro reliqPrint(args: varargs[untyped]): untyped =
  var statements: seq[NimNode]
  for iarg, varg in args:
    if iarg < args.len - 1:
      statements.add newCall("write", ident"stdout", newCall(ident"$", varg))
      statements.add newCall("write", ident"stdout", newLit(" "))
    else: statements.add newCall("writeLine", ident"stdout", newCall(ident"$", varg))
  return newStmtList(statements)

# print statement that only prints from rank 0
template print*(args: varargs[untyped]): untyped =
  if myRank() == 0: reliqPrint(args)

when isMainModule:
  import runtime
  reliq: print "I should only print once: ", 1, $2, [3]