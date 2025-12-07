#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: build/Makefile.nims
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
  metaGA = externalDir / "bin" / "ga-config"
  metaKokkos = externalDir / "bin" / "kokkos-meta"

const
  passC_GA = staticExec(metaGA + "--cflags") + 
             staticExec(metaGA + "--cppflags") +
             staticExec(metaGA + "--network_cppflags")
const 
  passL_GA = staticExec(metaGA + "--ldflags") + 
             staticExec(metaGA + "--network_ldflags") + 
             staticExec(metaGA + "--libs") +
             staticExec(metaGA + "--network_libs")
const
  passC_reliq = ""
  # Add rpath and library path to ensure we use Spack's MPI library (not system's Intel MPI)
  # This prevents linking against multiple MPI libraries simultaneously
  passL_reliq = "-Wl,-rpath," & externalDir / "lib" & 
                " -L" & externalDir / "lib" & " -lmpi"

const passC_Kokkos = staticExec(metaKokkos + "CXXFLAGS")
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
    passC = "-Ofast" + passC_GA + passC_Kokkos + passC_reliq
    passL = passL_GA + passL_Kokkos + passL_reliq
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

#[
Summary of fixed issue 12/07/2025: see docs/GLOBALARRAYS_SETUP.md

]#