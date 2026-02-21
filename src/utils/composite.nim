#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/utils/composite.nim
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

## Unified ``implement`` macro for adding methods to an already-defined
## ``record`` or ``class`` outside its original body.  Also re-exports
## ``record`` and ``class``, so a single import is sufficient:
##
## .. code-block:: nim
##   import utils/[composite]   # brings in record, class, and implement
##
## The macro inspects the compile-time caches of both the ``record`` and
## ``class`` systems and dispatches to ``recordImpl`` or ``classImpl``
## accordingly — identical functionality, cleaner syntax.
##
## Usage
## -----
## .. code-block:: nim
##   import utils/[composite]
##
##   record Foo:
##     var x: int
##
##   implement Foo with:
##     method doubled(): int =
##       return this.x * 2
##
##   implement Foo with:
##     method peek(): int {.immutable.} =
##       return this.x
##
## Works identically for classes:
##
## .. code-block:: nim
##   import utils/[composite]
##
##   class Bar:
##     var y: float
##
##   implement Bar with:
##     method scaled(k: float): float =
##       return this.y * k
##
## Generic types are fully supported — generic parameters are injected
## automatically from the cached definition, just as with ``recordImpl``
## and ``classImpl``.
##
## Error handling
## --------------
## If the name is found in neither cache a compile-time error is raised:
##
##   implement Unknown with:   # → error: No record or class named 'Unknown' found.

import std/[macros]
import record/[record]
import record/[recordinternal]
import class/[class]
import class/[classinternal]

export record
export class

#[ Extract the bare type name from the impl head ]#

proc resolveImplName(head: NimNode): NimNode =
  ## Accepts all valid head forms:
  ##   implement Foo with:          → ident"Foo"
  ##   implement Foo[T] with:       → ident"Foo"   (generics come from cached def)
  ##   implement Foo* with:         → ident"Foo"   (export marker ignored)
  case head.kind
  of nnkIdent:
    return head
  of nnkCommand:
    # `implement Foo with:` — Nim parses `Foo with` as nnkCommand(Foo, with_ident)
    return resolveImplName(head[0])
  of nnkBracketExpr:
    return head[0]
  of nnkPostfix:
    if $head[0] == "*": return head[1]
  of nnkInfix:
    if $head[0] == "*": return head[1]
  else: discard
  error("Invalid implement syntax: expected a type name.", head)

#[ The unified implement macro ]#

macro implement*(head: untyped, `with`: untyped): untyped =
  ## Add methods to an already-defined ``record`` or ``class``.
  ##
  ## Dispatches to ``recordImpl`` when the name is found in the record
  ## compile-time cache, or to ``classImpl`` when found in the class cache.
  ## Raises a compile-time error if the name is in neither cache.
  let nameNode = resolveImplName(head)

  if recordDefinitionFor(nameNode) != nil:
    return newCall(bindSym"recordImpl", nameNode, `with`)

  if classDefinitionFor(nameNode) != nil:
    return newCall(bindSym"classImpl", nameNode, `with`)

  error("No record or class named '" & $nameNode & "' found. " &
        "The type must be defined before the implement block.", head)
