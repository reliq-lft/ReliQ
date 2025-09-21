#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/reliq.nim
  Contact: reliq-lft@proton.me

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

## ====================================
## ReliQ Lattice Field Theory Framework
## ====================================
## 
## :Authors: Curtis Taylor Peterson
## :Version: 0.0.1
##
## > "Documentation is a love letter that you write to your future self." - Damian Conway
## 
## TL;DR
## =====
##
## `ReliQ` is an experimental lattice field theory framework written first and foremost with user-friendliness, performance, reliability, and portability across current and future heterogeneous architectures in mind. 
##
## * `ReliQ` is written in the elegant [Nim](https://nim-lang.org/) programming language 
## * `ReliQ` uses [Unified Parallel C++ (UPC++)](https://upcxx.lbl.gov/docs/html/guide.html) to implement its distributed memory model
## * `ReliQ` uses [Kokkos](https://kokkos.org/) to implement its shared memory model
##
## That's nice, but why another LFT codebase?
## ==========================================
##
## There is a gap in the landscape of lattice field theory codebases/frameworks that aims to address both the issue of user-friendliness and performance. There are plenty of codebases/frameworks that are user-friendly, but fail to deliver on performance for certain architectures. There are also plenty of code bases that deliver incredible performance, but push an equally incredible burden of knowledge on their users before they are able to utilize or modify said code bases in any meaningful way. There are even more code bases that strive to satisfy the needs of specific collaborations, hence disregarding general user-friendliness and performance in scenarios where either does not fit the needs of said collaboration.
##
## `ReliQ` strives to be a lattice field theory framework that fits the needs of a variety of lattice field theory practitioners and architectural targets. It is written for you now and into the future. That does *not* mean that `ReliQ` strives to be to most general lattice field theory framework out there, as decades of work on lattice field theory has taught us that such a goal is nearly impossible to achieve in practice. Rather, `ReliQ` strives to be a productive *framework* for which applications targeting their own levels of specificity are written. What that entails in practice is providing baseline parallel data structures and algorithms, nothing more, nothing less. 
## 
## > Continue... add code samples, et cetera...
## 
## Frequently asked questions
## ==========================
## > ...
#<!--- 
#Q: I've never heard of Nim. Why should I have to invest my time into learning a language that's not Fortran, C/C++, or Rust?
#A: You shouldn't. And you almost certainly won't be spending most of your time with ReliQ navigating the programmatic quirks of Nim. We have built `ReliQ` with the goal of having the average user learn Nim through using `ReliQ`, rather than the other way around. We would love any feedback that users could provide on practically realizing this goal. 
#--->

#[ ReliQ utilities and runtime environment ]#

import utils
import runtime

export utils
export runtime

#[ ReliQ lattice types ]#

import lattice/[latticeconcept]
import lattice/[simplecubiclattice]

export latticeconcept
export simplecubiclattice

#[ ReliQ field types ]#

import field/[fieldconcept]
import field/[simplecubicfield]

export fieldconcept
export simplecubicfield