#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: build/Makefile.nims
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 Curtis Taylor Peterson
  
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

import std/[os]

proc `/`*(sa, sb: string): string = sa & "/" & sb
proc `+`*(sa, sb: string): string = sa & " " & sb
proc `+=`*(sa: var string, sb: string) = sa = sa & " " & sb

### configuration specifications: users should modify these, if desired ###

const cwd = getCurrentDir()
const nimCache = cwd / "cache"

const nimArgs = [""]

const cxxFlags = [""]
const cxxLinks = [""]

### dependency specifications: users should not touch these ###

const externalDir = cwd / "external"

const 
  metaUPCXX = externalDir / "bin" / "upcxx-meta"
  metaKokkos = externalDir / "bin" / "kokkos-meta"

const
  passCXX_UPCXX = staticExec(metaUPCXX + "CXXFLAGS")
  passCPP_UPCXX = staticExec(metaUPCXX + "CPPFLAGS")
const passL_UPCXX = staticExec(metaUPCXX + "LIBFLAGS")

const passCXX_Kokkos = staticExec(metaKokkos + "CXXFLAGS")
const passL_Kokkos = staticExec(metaKokkos + "LIBFLAGS")

### execute compilation ###

proc search(dir, src: string): string {.raises: [IOError].} =
  for entry in walkDirRec(dir):
    if entry.splitPath().tail == src & ".nim": return string(entry)
  raise newException(IOError, (src & ".nim" + "not found"))
  return ""

task clean, "cleaning files":
  exec "rm" + cwd / "bin/*"

task build, "building file":
  var 
    passC = "-Ofast" + passCXX_UPCXX + passCPP_UPCXX + passCXX_Kokkos
    passL = passL_UPCXX + passL_Kokkos
  var args = newSeq[string](paramCount() - 2)
  var cmpl: string 

  for prmIdx in 0..<args.len: args[prmIdx] = paramStr(prmIdx + 3)

  for cxxFlag in cxxFlags: passC += cxxFlag
  for cxxLink in cxxLinks: passL += cxxLink

  cmpl = "external/bin/nim cpp --path:src"
  cmpl += "--nimcache:" & nimCache
  for nimArg in nimArgs: cmpl += nimArg
  cmpl += "--passC:\"" & passC & "\""
  cmpl += "--passL:\"" & passL & "\""
  cmpl += "-o:bin/" & args[^1]
  for prmIdx in 0..<(args.len - 1): cmpl += args[prmIdx]
  cmpl += search("src", args[^1])

  echo "BUILD:" + cmpl
  exec cmpl