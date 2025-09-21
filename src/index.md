<!--
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/index.md
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
-->

# ReliQ Lattice Field Theory Framework

> "Documentation is a love letter that you write to your future self." — Damian Conway

**Authors:** Curtis Taylor Peterson  
**Version:** 0.0.1

---

## TL;DR

`ReliQ` is an experimental lattice field theory framework written with user-friendliness, performance, reliability, and portability across heterogeneous architectures in mind.

- Written in [Nim](https://nim-lang.org/)
- Distributed memory via [UPC++](https://upcxx.lbl.gov/docs/html/guide.html)
- Shared memory via [Kokkos](https://kokkos.org/)

---

## Quick Start

```nim
import reliq
let lat = initLattice([4,4,4,4])
```

---

## Navigation

- [API Reference](reliq.html)
- [Examples](examples.html)
- [GitHub Repository](https://github.com/reliq-lft/ReliQ)

---

## Why Another LFT Codebase?

There’s a gap in the lattice field theory code landscape regarding user-friendliness versus performance. Many frameworks sacrifice one for the other, or are highly specialized.

`ReliQ` aims to be a productive *framework* with baseline parallel data structures and algorithms, leaving specific applications to layer on top.

---

## FAQ

<!--- 
<details>
<summary>I've never heard of Nim. Why should I invest my time in it?</summary>

You shouldn't *have* to! ReliQ is designed so that users can learn Nim by using ReliQ, not the other way around. Feedback on achieving this goal is very welcome.
</details>
--->

---

## More

> Continue... add code samples, further explanations, and more FAQ entries as your documentation grows.
