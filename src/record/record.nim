#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/record/record.nim
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

## Chapel-style record macro for Nim.
##
## Overview
## --------
## The ``record`` macro defines **value-type objects** that follow Chapel's
## record semantics (https://chapel-lang.org/docs/language/spec/records.html).
## A record is a plain Nim ``object`` (not ``ref object``): it lives on the
## stack, is copied on assignment, has no identity, and is never ``nil``.
##
## Records automatically receive compiler-generated:
##   - **Default initializer** (``init``) with field defaults
##   - **Convenience constructors** (``RecordName.init()``, ``newRecordName()``, ``RecordName.new()``)
##   - **Copy operations** (``copyInit``, ``copyFrom``)
##   - **Comparison operators** (``==``, ``!=``, ``<``, ``<=``)
##   - **Hash function** (``hash``, compatible with ``std/hashes``)
##   - **String representation** (``$``) in Chapel-style ``RecordName(field = value, ...)`` format
##   - **Destructor hook** (``=destroy``) when ``deinit()`` is defined
##   - **Meta helpers** (``recordName()`` static method)
##   - **Auto-export** of all fields and methods
##
## All of these can be overridden by the user externally.
##
## Quick Start
## -----------
## .. code-block:: nim
##   import record
##
##   record ActorRecord:
##     var name = "unknown"
##     var age: int = 0
##
##     method init() =
##       echo "Creating an ActorRecord"
##
##     method greet() =
##       echo "Hi, I'm " & this.name
##
##   var a = ActorRecord.init()    # or newActorRecord() or ActorRecord.new()
##   a.name = "Alice"
##   a.greet()                     # "Hi, I'm Alice"
##
##   var b = a.copyInit()          # explicit deep copy
##   assert a == b                 # field-wise equality
##   echo a                        # ActorRecord(name = Alice, age = 0)
##
## DSL Syntax
## ----------
## .. code-block:: nim
##   record RecordName:
##     ## optional doc comment
##     var field1: Type = defaultValue   # field with explicit type and default
##     var field2 = "hello"              # type inferred from literal
##     var field3: int                   # type-only, zero-initialized
##
##     method init() =                   # optional user-defined initializer
##       # `this` refers to the record instance (injected automatically)
##       echo "created " & this.field2
##
##     method deinit() =                 # optional destructor
##       echo "destroyed"
##
##     method doSomething(): int =       # instance method
##       return this.field3 + 1
##
##     method getName(): string {.immutable.} = # immutable method
##       return this.field2                 # can be called on immutable values
##
##     method classLevel() {.static.} =  # static method (no `this`)
##       echo "I am a class-level method"
##
##   record EmptyRecord                  # body-less record is also valid
##
##   record Wrapper[T]:                   # generic record
##     var data: T
##     var label: string = "default"
##
##   record Constrained[T: SomeNumber]:   # constrained generic
##     var value: T
##
## Fields
## ------
## Fields are declared with ``var``. Each field must have either an explicit
## type, a default value (from which the type is inferred), or both.
## Supported auto-detected literal types: ``char``, ``int``, ``int8``–``int64``,
## ``uint``, ``uint8``–``uint64``, ``float``, ``float32``–``float128``,
## ``string``, and ``bool``.
##
## ``let`` and ``const`` fields are **not** supported (compile-time error).
##
## Methods
## -------
## Methods are declared with ``method`` in the DSL but are compiled to
## ``proc`` (static dispatch, matching Chapel semantics). Instance methods
## receive an implicit ``this: var RecordName`` first parameter so they can
## read and mutate fields. Immutable methods are annotated with ``{.immutable.}``
## and receive ``this: RecordName`` (no ``var``), allowing them to be called
## on both mutable and immutable values. Static methods are annotated with
## ``{.static.}`` and receive ``_: typedesc[RecordName]`` instead.
##
## Constructors
## ------------
## Three convenience constructor forms are auto-generated for each ``init``:
##
## =========================  ====================================
## Form                       Description
## =========================  ====================================
## ``RecordName.init(...)``   Static call on the type
## ``newRecordName(...)``     Free-standing ``new`` + name proc
## ``RecordName.new(...)``    Alternative static ``new`` call
## =========================  ====================================
##
## All three call the user-defined ``init()`` (or the auto-generated one)
## and return the initialized record value. If ``init()`` takes parameters,
## all three constructors forward them.
##
## Copy Semantics
## --------------
## Since records are plain Nim objects, assignment ``b = a`` already performs
## a shallow field-wise copy. Two explicit helpers are also generated:
##
## - ``copyInit(other: T): T`` — creates and returns a new copy.
## - ``copyFrom(lhs: var T, rhs: T)`` — copies all fields from ``rhs`` to ``lhs``.
##
## Comparison & Hashing
## --------------------
## - ``==`` and ``!=``: field-by-field equality (all fields must match).
## - ``<`` and ``<=``: lexicographic ordering over fields in declaration order.
## - ``hash``: combines per-field hashes using ``std/hashes`` (``!&`` and ``!$``),
##   making records usable as keys in ``Table`` and ``HashSet``.
##
## String Representation
## ---------------------
## The ``$`` proc produces Chapel-style output::
##
##   RecordName(field1 = value1, field2 = value2)
##
## For empty records: ``RecordName()``.
##
## Destructor (``deinit``)
## -----------------------
## If a ``method deinit()`` is defined, a Nim ``=destroy`` hook is generated
## that calls ``deinit()`` when the record goes out of scope. The generated
## destructor uses ``{.raises: [].}`` with ``{.cast(raises: []).}`` to satisfy
## Nim's effect system.
##
## Meta
## ----
## - ``RecordName.recordName(): string`` — returns the type name as a string
##   at runtime (static method).
##
## Architecture
## ------------
## The implementation is a **hook-based compile-time plugin system** adapted
## from `jjv360/nim-classes <https://github.com/jjv360/nim-classes>`_ (MIT
## License), restructured for Chapel record semantics.
##
## **Compilation Pipeline** — Each ``record`` invocation passes through these
## stages in order:
##
## 1. ``RecordPreload`` — Early initialization (currently unused, reserved).
## 2. ``RecordGatherDefinitions`` — Parse ``var`` fields and ``method``
##    definitions from the DSL body into structured data.
## 3. ``RecordAddExtraDefinitions`` — Insert auto-generated ``init()`` if
##    missing; generate convenience constructors.
## 4. ``RecordModifyDefinitions`` — Inject field defaults into ``init``
##    bodies; set return types; auto-export all fields and methods with ``*``.
## 5. ``RecordGenerateCode`` — Emit Nim AST: object type definition with
##    RecList, forward declarations, proc bodies, copy/comparison/hash/meta
##    procs, and ``=destroy`` hook.
## 6. ``RecordFinalize`` — Post-processing (currently unused, reserved).
## 7. ``RecordDebugEcho`` — When ``-d:debugrecords=true`` or
##    ``-d:debugrecords=RecordName`` is set, echo diagnostics.
##
## **Module Map:**
##
## =======================  ================================================
## Module                   Responsibility
## =======================  ================================================
## ``recordinternal``       Core types (``RecordDescription``,
##                          ``RecordCompilerStage``, ``RecordCompilerHook``),
##                          ``newRecordDescription()``, ``compile()``
## ``recordutils``          AST helpers: ``traverseRecordStatementList``,
##                          ``variableName``, ``traverseParams``,
##                          ``unbindAllSym``
## ``recordvar``            Field (``var``) parsing, type inference, RecList
##                          code generation
## ``recordmethods``        Method parsing, ``method`` → ``proc`` conversion,
##                          ``this`` injection, static dispatch
## ``recordconstructors``   Default ``init``, field-default injection,
##                          convenience constructors
## ``recordexports``        Auto-export fields and methods with ``*``
## ``recordcopy``           ``copyInit`` and ``copyFrom`` proc generation
## ``recordcomparison``     ``==``, ``!=``, ``<``, ``<=`` operators and
##                          ``hash`` proc generation
## ``recorddestructors``    ``=destroy`` hook from ``deinit()``
## ``recordmeta``           ``recordName()`` static method, ``$`` proc
## ``recordimpl``           ``impl`` macro for adding methods outside
##                          the original ``record`` body
## =======================  ================================================
##
## Each module registers a ``RecordCompilerHook`` at compile time via
## ``static:`` blocks. The hooks are stored in the global
## ``recordCompilerHooks`` sequence and called in registration order at
## each stage.
##
## Debugging
## ---------
## Compile with ``-d:debugrecords=true`` to see the generated AST for every
## record, or ``-d:debugrecords=MyRecord`` to debug a single record.
##
## Generics
## --------
## Records support generic type parameters:
##
## .. code-block:: nim
##   record Pair[T, U]:
##     var first: T
##     var second: U
##
##     method swap(): Pair[U, T] =
##       var r: Pair[U, T]
##       r.first = this.second
##       r.second = this.first
##       return r
##
##   var p = Pair[int, string].init()
##   p.first = 42
##   p.second = "hello"
##
## Type constraints are also supported:
##
## .. code-block:: nim
##   record NumBox[T: SomeNumber]:
##     var value: T
##
## All auto-generated procs (``==``, ``hash``, ``$``, ``copyInit``, etc.)
## are properly generic and work with any valid instantiation.
##
## External Method Blocks (``impl``)
## ----------------------------------
## Methods can be added to an already-defined record outside its original
## body using the ``impl`` macro (inspired by Rust):
##
## .. code-block:: nim
##   record Foo:
##     var x: int
##
##   impl Foo:
##     method doubled(): int =
##       return this.x * 2
##
##     method describe() {.static.} =
##       echo "I am Foo"
##
##     method peek(): int {.immutable.} =
##       return this.x
##
## ``impl`` blocks support instance, immutable, and static methods.
## Generic records are fully supported — generic parameters are
## automatically injected from the cached record definition.
## Multiple ``impl`` blocks may target the same record, and they
## can appear in separate modules (as long as the record's module
## is imported first).
##
## Limitations
## -----------
## - No inheritance (``record Foo of Bar:`` is a compile-time error).
## - No ``let`` or ``const`` fields (enforced at compile time).
## - Only ``method`` declarations are allowed inside the body (not ``proc``,
##   ``func``, ``template``, ``iterator``, etc.).
## - Copy semantics are shallow — if a field is a ``ref`` type, only the
##   pointer is copied, not the referenced data.
## 
## Architecture adapted from jjv360/nim-classes 
## ============================================
## 
## MIT License
##  
## Copyright (c) 2021 jjv360
##  
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##  
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
## WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
## CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import std/[macros]
import std/[hashes]

import recordinternal
import recordvar
import recordmethods
import recordconstructors
import recordexports
import recordcopy
import recordcomparison
import recorddestructors
import recordmeta
import recordimpl

export static
export hashes
export recordimpl

## Record definition with body.
macro record*(head: untyped, body: untyped): untyped =
  return newRecordDescription("record", head, body).compile()

## Empty record definition (no body).
macro record*(head: untyped): untyped =
  return newRecordDescription("record", head, newStmtList()).compile()

#[ unit tests ]#

when isMainModule:
  import std/[unittest, hashes, tables, sets]

  # -------------------------------------------------------------------------
  # Test record definitions
  # -------------------------------------------------------------------------

  record TimeStamp:
    var time: string = "1/1/2011"

  record ActorRecord:
    ## A person with a name and age.
    var name: string = "unknown"
    var age: int = 0

    method init() =
      echo "Creating ActorRecord: " & this.name

    method greet(): string =
      return "Hi, I'm " & this.name & ", age " & $this.age

    method birthday() =
      this.age += 1

    method describe() {.static.} =
      echo "ActorRecord is a record type"

  record Empty

  record Tracked:
    var id: int = 0

    method init() =
      echo "Tracked " & $this.id & " created"

    method deinit() =
      echo "Tracked " & $this.id & " destroyed"

  record MultiField:
    var x: int = 10
    var y: int = 20
    var label: string = "point"

  record TypeInferred:
    ## Tests type auto-detection from various literal kinds.
    var anInt = 42
    var aFloat = 3.14
    var aString = "hello"
    var aBool = true
    var aChar = 'z'

  record SingleField:
    var value: int = 0

  record NoInit:
    ## A record with no user-defined init().
    var data: int = 100

  # -------------------------------------------------------------------------
  # Test suites
  # -------------------------------------------------------------------------

  suite "record: field declarations":
    test "explicit type with default value":
      var ts = TimeStamp.init()
      check ts.time == "1/1/2011"

    test "multiple fields with explicit types":
      var a = ActorRecord.init()
      check a.name == "unknown"
      check a.age == 0

    test "type inference from int literal":
      var ti = TypeInferred.init()
      check ti.anInt == 42

    test "type inference from float literal":
      var ti = TypeInferred.init()
      check ti.aFloat == 3.14

    test "type inference from string literal":
      var ti = TypeInferred.init()
      check ti.aString == "hello"

    test "type inference from bool literal":
      var ti = TypeInferred.init()
      check ti.aBool == true

    test "type inference from char literal":
      var ti = TypeInferred.init()
      check ti.aChar == 'z'

    test "multiple fields preserve order and defaults":
      var mf = MultiField.init()
      check mf.x == 10
      check mf.y == 20
      check mf.label == "point"

  suite "record: field mutation":
    test "simple field assignment":
      var a = ActorRecord.init()
      a.name = "Alice"
      a.age = 30
      check a.name == "Alice"
      check a.age == 30

    test "multiple mutations":
      var mf = MultiField.init()
      mf.x = 100
      mf.y = 200
      mf.label = "origin"
      check mf.x == 100
      check mf.y == 200
      check mf.label == "origin"

    test "mutate back to default":
      var a = ActorRecord.init()
      a.name = "Temp"
      a.name = "unknown"
      check a.name == "unknown"

  suite "record: instance methods":
    test "method with return value":
      var a = ActorRecord.init()
      a.name = "Alice"
      a.age = 30
      check a.greet() == "Hi, I'm Alice, age 30"

    test "method that mutates fields":
      var a = ActorRecord.init()
      a.age = 29
      a.birthday()
      check a.age == 30

    test "method mutation is visible after call":
      var a = ActorRecord.init()
      a.age = 0
      a.birthday()
      a.birthday()
      a.birthday()
      check a.age == 3

  suite "record: static methods":
    test "recordName() returns type name":
      check ActorRecord.recordName() == "ActorRecord"
      check TimeStamp.recordName() == "TimeStamp"
      check Empty.recordName() == "Empty"
      check MultiField.recordName() == "MultiField"

    test "user-defined static method compiles":
      # Just ensure it doesn't crash — output goes to stdout
      ActorRecord.describe()

  suite "record: constructors — RecordName.init()":
    test "basic init":
      var a = ActorRecord.init()
      check a.name == "unknown"
      check a.age == 0

    test "init sets defaults for all fields":
      var mf = MultiField.init()
      check mf.x == 10
      check mf.y == 20
      check mf.label == "point"

    test "auto-generated init when none defined":
      var ni = NoInit.init()
      check ni.data == 100

  suite "record: constructors — newRecordName()":
    test "newActorRecord":
      var b = newActorRecord()
      check b.name == "unknown"
      check b.age == 0

    test "newMultiField":
      var mf = newMultiField()
      check mf.x == 10
      check mf.y == 20

    test "newEmpty":
      var e = newEmpty()
      check $e == "Empty()"

    test "newNoInit":
      var ni = newNoInit()
      check ni.data == 100

  suite "record: constructors — RecordName.new()":
    test "ActorRecord.new()":
      var c = ActorRecord.new()
      check c.name == "unknown"
      c.name = "Carol"
      check c.name == "Carol"

    test "MultiField.new()":
      var mf = MultiField.new()
      check mf.x == 10

    test "Empty.new()":
      var e = Empty.new()
      check $e == "Empty()"

  suite "record: value semantics — assignment":
    test "assignment produces independent copy":
      var a = ActorRecord.init()
      a.name = "Alice"
      a.age = 30
      var b = a
      b.name = "Bob"
      b.age = 25
      check a.name == "Alice"
      check a.age == 30
      check b.name == "Bob"
      check b.age == 25

    test "multi-field assignment independence":
      var m1 = MultiField.init()
      m1.x = 99
      var m2 = m1
      m2.x = 1
      m2.label = "other"
      check m1.x == 99
      check m1.label == "point"
      check m2.x == 1
      check m2.label == "other"

    test "chained assignment":
      var a = ActorRecord.init()
      a.name = "A"
      var b = a
      var c = b
      c.name = "C"
      check a.name == "A"
      check b.name == "A"
      check c.name == "C"

  suite "record: value semantics — copyInit":
    test "copyInit produces independent copy":
      var a = ActorRecord.init()
      a.name = "Alice"
      a.age = 30
      var e = a.copyInit()
      e.age = 99
      check a.age == 30
      check e.age == 99

    test "copyInit preserves all fields":
      var mf = MultiField.init()
      mf.x = 7
      mf.y = 8
      mf.label = "test"
      var mf2 = mf.copyInit()
      check mf2.x == 7
      check mf2.y == 8
      check mf2.label == "test"

    test "copyInit on empty record":
      var e = Empty.init()
      var e2 = e.copyInit()
      check $e2 == "Empty()"

  suite "record: value semantics — copyFrom":
    test "copyFrom overwrites all fields":
      var a = ActorRecord.init()
      a.name = "Alice"
      a.age = 30
      var b = ActorRecord.init()
      b.copyFrom(a)
      check b.name == "Alice"
      check b.age == 30

    test "copyFrom source unchanged":
      var src = MultiField.init()
      src.x = 42
      src.y = 84
      src.label = "source"
      var dst = MultiField.init()
      dst.copyFrom(src)
      dst.x = 0
      check src.x == 42
      check dst.x == 0

  suite "record: equality (==, !=)":
    test "equal records":
      var a = ActorRecord.init()
      a.name = "Alice"
      a.age = 30
      var b = ActorRecord.init()
      b.name = "Alice"
      b.age = 30
      check a == b

    test "different name field":
      var a = ActorRecord.init()
      a.name = "Alice"
      var b = ActorRecord.init()
      b.name = "Bob"
      check a != b

    test "different age field":
      var a = ActorRecord.init()
      a.age = 10
      var b = ActorRecord.init()
      b.age = 20
      check a != b

    test "all defaults are equal":
      var a = ActorRecord.init()
      var b = ActorRecord.init()
      check a == b

    test "empty records are always equal":
      var e1 = Empty.init()
      var e2 = Empty.init()
      check e1 == e2

    test "single field equality":
      var s1 = SingleField.init()
      s1.value = 42
      var s2 = SingleField.init()
      s2.value = 42
      check s1 == s2
      s2.value = 43
      check s1 != s2

    test "copyInit produces equal record":
      var a = MultiField.init()
      a.x = 5
      a.y = 10
      a.label = "copy"
      var b = a.copyInit()
      check a == b

    test "!= is negation of ==":
      var a = ActorRecord.init()
      var b = ActorRecord.init()
      check not (a != b)
      b.name = "different"
      check a != b
      check not (a == b)

  suite "record: ordering (<, <=)":
    test "< first field differs":
      var lo = ActorRecord.init()
      lo.name = "AAA"
      var hi = ActorRecord.init()
      hi.name = "ZZZ"
      check lo < hi
      check not (hi < lo)

    test "< second field breaks tie":
      var a = ActorRecord.init()
      a.name = "same"
      a.age = 10
      var b = ActorRecord.init()
      b.name = "same"
      b.age = 20
      check a < b
      check not (b < a)

    test "< equal records are not less":
      var a = ActorRecord.init()
      a.name = "same"
      a.age = 10
      var b = ActorRecord.init()
      b.name = "same"
      b.age = 10
      check not (a < b)
      check not (b < a)

    test "<= with equal records":
      var a = ActorRecord.init()
      a.name = "X"
      var b = ActorRecord.init()
      b.name = "X"
      check a <= b
      check b <= a

    test "<= with less-than records":
      var lo = MultiField.init()
      lo.x = 1
      var hi = MultiField.init()
      hi.x = 2
      check lo <= hi
      check not (hi <= lo)

    test "multi-field lexicographic ordering":
      var a = MultiField.init()
      a.x = 1
      a.y = 100
      a.label = "zzz"
      var b = MultiField.init()
      b.x = 2
      b.y = 0
      b.label = "aaa"
      # x=1 < x=2, so a < b regardless of y and label
      check a < b

    test "multi-field tie-breaking":
      var a = MultiField.init()
      a.x = 5
      a.y = 3
      var b = MultiField.init()
      b.x = 5
      b.y = 7
      # x tied, y=3 < y=7
      check a < b

  suite "record: hash":
    test "equal records have equal hash":
      var a = ActorRecord.init()
      a.name = "Alice"
      a.age = 30
      var b = ActorRecord.init()
      b.name = "Alice"
      b.age = 30
      check hash(a) == hash(b)

    test "different records likely have different hash":
      var a = ActorRecord.init()
      a.name = "Alice"
      var b = ActorRecord.init()
      b.name = "Bob"
      # Not guaranteed but overwhelmingly likely
      check hash(a) != hash(b)

    test "empty record hash is consistent":
      var e1 = Empty.init()
      var e2 = Empty.init()
      check hash(e1) == hash(e2)

    test "hash changes when field changes":
      var a = SingleField.init()
      a.value = 1
      let h1 = hash(a)
      a.value = 2
      let h2 = hash(a)
      check h1 != h2

    test "records usable as Table keys":
      var t = initTable[SingleField, string]()
      var k1 = SingleField.init()
      k1.value = 1
      var k2 = SingleField.init()
      k2.value = 2
      t[k1] = "one"
      t[k2] = "two"
      check t[k1] == "one"
      check t[k2] == "two"

    test "records usable in HashSet":
      var s = initHashSet[SingleField]()
      var k1 = SingleField.init()
      k1.value = 10
      var k2 = SingleField.init()
      k2.value = 20
      s.incl(k1)
      s.incl(k2)
      s.incl(k1)  # duplicate
      check s.len == 2
      check k1 in s
      check k2 in s

  suite "record: $ string representation":
    test "basic toString":
      var a = ActorRecord.init()
      a.name = "Alice"
      a.age = 30
      check $a == "ActorRecord(name = Alice, age = 30)"

    test "default fields toString":
      var a = ActorRecord.init()
      check $a == "ActorRecord(name = unknown, age = 0)"

    test "empty record toString":
      var emp = Empty.init()
      check $emp == "Empty()"

    test "single field toString":
      var s = SingleField.init()
      s.value = 42
      check $s == "SingleField(value = 42)"

    test "multi-field toString":
      var mf = MultiField.init()
      check $mf == "MultiField(x = 10, y = 20, label = point)"

    test "type-inferred fields toString":
      var ti = TypeInferred.init()
      check $ti == "TypeInferred(anInt = 42, aFloat = 3.14, aString = hello, aBool = true, aChar = z)"

  suite "record: empty record":
    test "empty record can be created with init":
      var emp = Empty.init()
      check $emp == "Empty()"

    test "empty record with new":
      var emp = Empty.new()
      check $emp == "Empty()"

    test "empty record with newEmpty":
      var emp = newEmpty()
      check $emp == "Empty()"

    test "two empty records are equal":
      check Empty.init() == Empty.init()

    test "empty record < is always false":
      # No fields → no lexicographic comparison procs generated,
      # but == always returns true.
      var e1 = Empty.init()
      var e2 = Empty.init()
      check e1 == e2

    test "empty record recordName":
      check Empty.recordName() == "Empty"

  suite "record: auto-generated init":
    test "record with no user init gets default init":
      var ni = NoInit.init()
      check ni.data == 100

    test "newNoInit works":
      var ni = newNoInit()
      check ni.data == 100

    test "NoInit.new() works":
      var ni = NoInit.new()
      check ni.data == 100

  suite "record: deinit / destructor":
    test "deinit is called when record goes out of scope":
      proc testDeinit(): int =
        var t = Tracked.init()
        t.id = 42
        return t.id
      check testDeinit() == 42

    test "deinit runs for locally scoped records":
      proc scopeTest(): string =
        var t = Tracked.init()
        t.id = 99
        return "done"
      check scopeTest() == "done"

    test "record without deinit has no destructor issues":
      # ActorRecord has no deinit — should work fine
      var a = ActorRecord.init()
      a.name = "test"
      check a.name == "test"

  suite "record: comprehensive integration":
    test "full lifecycle: create, mutate, copy, compare, stringify, hash":
      # Create
      var a = ActorRecord.init()
      a.name = "Alice"
      a.age = 30

      # Mutate via method
      a.birthday()
      check a.age == 31

      # Copy
      var b = a.copyInit()
      check a == b

      # Mutate copy independently
      b.name = "Bob"
      check a != b
      check a.name == "Alice"
      check b.name == "Bob"

      # Ordering
      check a < b  # "Alice" < "Bob"

      # String representation
      check $a == "ActorRecord(name = Alice, age = 31)"

      # Hash consistency after copy
      var c = ActorRecord.init()
      c.name = "Alice"
      c.age = 31
      check hash(a) == hash(c)

      # Method
      check a.greet() == "Hi, I'm Alice, age 31"

    test "many records in a collection":
      var records: seq[ActorRecord] = @[]
      for i in 0..9:
        var r = ActorRecord.init()
        r.name = "Actor" & $i
        r.age = i * 10
        records.add(r)
      check records.len == 10
      check records[0].name == "Actor0"
      check records[9].name == "Actor9"
      check records[0].age == 0
      check records[9].age == 90
      check records[0] != records[9]
      check records[0] < records[9]  # "Actor0" < "Actor9"

    test "copyFrom then equality":
      var src = MultiField.init()
      src.x = 42
      src.y = 84
      src.label = "src"
      var dst = MultiField.init()
      check src != dst
      dst.copyFrom(src)
      check src == dst

  # -------------------------------------------------------------------------
  # Generic record definitions
  # -------------------------------------------------------------------------

  record Box[T]:
    ## A single-element generic container.
    var value: T

  record Pair[A, B]:
    ## A two-element generic container with different types.
    var first: A
    var second: B

    method swap(): Pair[B, A] =
      var r: Pair[B, A]
      r.first = this.second
      r.second = this.first
      return r

  record NumBox[T: SomeNumber]:
    ## A constrained generic: T must be a numeric type.
    var value: T

    method doubled(): T =
      return this.value * 2

  record GenericWithDefault[T]:
    ## Generic record with a non-generic field that has a default.
    var data: T
    var label: string = "unlabelled"

  record TrackedBox[T]:
    ## Generic record with deinit.
    var payload: T
    var id: int = 0

    method init() =
      echo "TrackedBox " & $this.id & " created"

    method deinit() =
      echo "TrackedBox " & $this.id & " destroyed"

  record EmptyGeneric[T]

  # -------------------------------------------------------------------------
  # Generic test suites
  # -------------------------------------------------------------------------

  suite "generic record: basic construction":
    test "Box[int] with init":
      var b = Box[int].init()
      b.value = 42
      check b.value == 42

    test "Box[string] with init":
      var b = Box[string].init()
      b.value = "hello"
      check b.value == "hello"

    test "Box[float] with new":
      var b = Box[float].new()
      b.value = 3.14
      check b.value == 3.14

    test "newBox[int]":
      var b = newBox[int]()
      b.value = 99
      check b.value == 99

    test "Pair[int, string] construction":
      var p = Pair[int, string].init()
      p.first = 1
      p.second = "one"
      check p.first == 1
      check p.second == "one"

    test "GenericWithDefault preserves string default":
      var g = GenericWithDefault[int].init()
      g.data = 10
      check g.data == 10
      check g.label == "unlabelled"

  suite "generic record: constrained generics":
    test "NumBox[int]":
      var n = NumBox[int].init()
      n.value = 21
      check n.doubled() == 42

    test "NumBox[float]":
      var n = NumBox[float].init()
      n.value = 1.5
      check n.doubled() == 3.0

  suite "generic record: methods":
    test "Pair swap method":
      var p = Pair[int, string].init()
      p.first = 42
      p.second = "answer"
      let swapped = p.swap()
      check swapped.first == "answer"
      check swapped.second == 42

    test "NumBox doubled":
      var n = NumBox[int].init()
      n.value = 7
      check n.doubled() == 14

  suite "generic record: value semantics":
    test "assignment copies independently":
      var a = Box[int].init()
      a.value = 10
      var b = a
      b.value = 20
      check a.value == 10
      check b.value == 20

    test "copyInit produces independent copy":
      var a = Pair[int, string].init()
      a.first = 5
      a.second = "five"
      var b = a.copyInit()
      b.first = 6
      check a.first == 5
      check b.first == 6
      check b.second == "five"

    test "copyFrom":
      var src = Box[string].init()
      src.value = "source"
      var dst = Box[string].init()
      dst.value = "dest"
      dst.copyFrom(src)
      check dst.value == "source"
      src.value = "changed"
      check dst.value == "source"

  suite "generic record: equality":
    test "equal Box[int]":
      var a = Box[int].init()
      a.value = 42
      var b = Box[int].init()
      b.value = 42
      check a == b

    test "unequal Box[int]":
      var a = Box[int].init()
      a.value = 1
      var b = Box[int].init()
      b.value = 2
      check a != b

    test "equal Pair":
      var a = Pair[int, string].init()
      a.first = 1
      a.second = "x"
      var b = Pair[int, string].init()
      b.first = 1
      b.second = "x"
      check a == b

    test "unequal Pair (second field)":
      var a = Pair[int, string].init()
      a.first = 1
      a.second = "x"
      var b = Pair[int, string].init()
      b.first = 1
      b.second = "y"
      check a != b

    test "empty generic equality":
      var a = EmptyGeneric[int].init()
      var b = EmptyGeneric[int].init()
      check a == b

  suite "generic record: ordering":
    test "Box[int] < ordering":
      var a = Box[int].init()
      a.value = 1
      var b = Box[int].init()
      b.value = 2
      check a < b
      check not (b < a)

    test "Box[string] lexicographic":
      var a = Box[string].init()
      a.value = "apple"
      var b = Box[string].init()
      b.value = "banana"
      check a < b

    test "Pair lexicographic by first then second":
      var a = Pair[int, string].init()
      a.first = 1
      a.second = "z"
      var b = Pair[int, string].init()
      b.first = 2
      b.second = "a"
      check a < b  # first field decides

    test "<= with equal generics":
      var a = Box[int].init()
      a.value = 5
      var b = Box[int].init()
      b.value = 5
      check a <= b
      check b <= a

  suite "generic record: hash":
    test "equal Box[int] same hash":
      var a = Box[int].init()
      a.value = 42
      var b = Box[int].init()
      b.value = 42
      check hash(a) == hash(b)

    test "different Box[int] different hash":
      var a = Box[int].init()
      a.value = 1
      var b = Box[int].init()
      b.value = 2
      check hash(a) != hash(b)

    test "generic records in Table":
      var t = initTable[Box[int], string]()
      var k1 = Box[int].init()
      k1.value = 10
      var k2 = Box[int].init()
      k2.value = 20
      t[k1] = "ten"
      t[k2] = "twenty"
      check t[k1] == "ten"
      check t[k2] == "twenty"

    test "generic records in HashSet":
      var s = initHashSet[Box[string]]()
      var a = Box[string].init()
      a.value = "hello"
      var b = Box[string].init()
      b.value = "world"
      s.incl(a)
      s.incl(b)
      s.incl(a)
      check s.len == 2

  suite "generic record: $ string representation":
    test "Box[int] toString":
      var b = Box[int].init()
      b.value = 42
      check $b == "Box(value = 42)"

    test "Box[string] toString":
      var b = Box[string].init()
      b.value = "hi"
      check $b == "Box(value = hi)"

    test "Pair toString":
      var p = Pair[int, string].init()
      p.first = 1
      p.second = "one"
      check $p == "Pair(first = 1, second = one)"

    test "GenericWithDefault toString":
      var g = GenericWithDefault[float].init()
      g.data = 2.5
      check $g == "GenericWithDefault(data = 2.5, label = unlabelled)"

    test "empty generic toString":
      var e = EmptyGeneric[int].init()
      check $e == "EmptyGeneric()"

  suite "generic record: meta":
    test "recordName for generic records":
      check Box[int].recordName() == "Box"
      check Pair[int, string].recordName() == "Pair"
      check NumBox[float].recordName() == "NumBox"

  suite "generic record: deinit":
    test "deinit fires for generic record":
      proc testGenericDeinit(): int =
        var t = TrackedBox[string].init()
        t.payload = "data"
        t.id = 77
        return t.id
      check testGenericDeinit() == 77

  suite "generic record: different instantiations":
    test "Box[int] and Box[string] are independent types":
      var bi = Box[int].init()
      bi.value = 10
      var bs = Box[string].init()
      bs.value = "ten"
      check $bi == "Box(value = 10)"
      check $bs == "Box(value = ten)"

    test "multiple Pair instantiations":
      var p1 = Pair[int, int].init()
      p1.first = 1
      p1.second = 2
      var p2 = Pair[string, string].init()
      p2.first = "a"
      p2.second = "b"
      check $p1 == "Pair(first = 1, second = 2)"
      check $p2 == "Pair(first = a, second = b)"

    test "nested generic records":
      var outer = Box[Box[int]].init()
      var inner = Box[int].init()
      inner.value = 42
      outer.value = inner
      check outer.value.value == 42

  suite "generic record: comprehensive integration":
    test "full generic lifecycle":
      # Create
      var p = Pair[int, string].init()
      p.first = 42
      p.second = "answer"

      # Copy
      var p2 = p.copyInit()
      check p == p2

      # Mutate independently
      p2.first = 0
      check p != p2
      check p.first == 42

      # Ordering
      check p2 < p  # 0 < 42

      # Hash
      var p3 = Pair[int, string].init()
      p3.first = 42
      p3.second = "answer"
      check hash(p) == hash(p3)

      # String
      check $p == "Pair(first = 42, second = answer)"

      # Swap method
      let swapped = p.swap()
      check swapped.first == "answer"
      check swapped.second == 42

  # -------------------------------------------------------------------------
  # impl block definitions
  # -------------------------------------------------------------------------

  impl ActorRecord:
    ## Methods added via impl block
    method fullGreet(): string =
      return "Hello! " & this.greet()

    method setName(n: string) =
      this.name = n

    method isAdult(): bool {.immutable.} =
      return this.age >= 18

    method typeName(): string {.static.} =
      return "ActorRecord"

  impl MultiField:
    method sum(): int {.immutable.} =
      return this.x + this.y

    method reset() =
      this.x = 0
      this.y = 0
      this.label = ""

  impl Box:
    method show(): string {.immutable.} =
      return "Box(" & $this.value & ")"

  impl NumBox:
    method tripled(): T {.immutable.} =
      return this.value * 3

  impl Pair:
    method description(): string {.immutable.} =
      return "(" & $this.first & ", " & $this.second & ")"

  # -------------------------------------------------------------------------
  # impl block test suites
  # -------------------------------------------------------------------------

  suite "impl: instance methods":
    test "impl method can call original methods":
      var a = ActorRecord.init()
      a.name = "Alice"
      a.age = 30
      check a.fullGreet() == "Hello! Hi, I'm Alice, age 30"

    test "impl method can mutate fields":
      var a = ActorRecord.init()
      a.setName("Bob")
      check a.name == "Bob"

    test "impl method on another record":
      var mf = MultiField.init()
      check mf.sum() == 30  # 10 + 20
      mf.reset()
      check mf.x == 0
      check mf.y == 0
      check mf.label == ""

  suite "impl: immutable methods":
    test "immutable impl method on mutable value":
      var a = ActorRecord.init()
      a.age = 20
      check a.isAdult() == true

    test "immutable impl method on let value":
      let a = ActorRecord.init()
      # age defaults to 0, so not adult
      check a.isAdult() == false

    test "immutable impl method on another record":
      let mf = MultiField.init()
      check mf.sum() == 30

  suite "impl: static methods":
    test "static impl method":
      check ActorRecord.typeName() == "ActorRecord"

  suite "impl: generic records":
    test "impl method on Box[int]":
      var b = Box[int].init()
      b.value = 42
      check b.show() == "Box(42)"

    test "impl method on Box[string]":
      var b = Box[string].init()
      b.value = "hello"
      check b.show() == "Box(hello)"

    test "impl method on constrained generic NumBox":
      var n = NumBox[int].init()
      n.value = 7
      check n.tripled() == 21

    test "impl method on NumBox[float]":
      var n = NumBox[float].init()
      n.value = 2.5
      check n.tripled() == 7.5

    test "impl method on multi-param generic Pair":
      var p = Pair[int, string].init()
      p.first = 42
      p.second = "answer"
      check p.description() == "(42, answer)"

    test "impl method on Pair[string, string]":
      var p = Pair[string, string].init()
      p.first = "hello"
      p.second = "world"
      check p.description() == "(hello, world)"

  suite "impl: multiple impl blocks":
    test "methods from different impl blocks coexist":
      var a = ActorRecord.init()
      a.name = "Test"
      a.age = 25
      # Original method
      check a.greet() == "Hi, I'm Test, age 25"
      # impl method
      check a.fullGreet() == "Hello! Hi, I'm Test, age 25"
      # impl static method
      check ActorRecord.typeName() == "ActorRecord"
      # impl immutable method
      check a.isAdult() == true

  # -------------------------------------------------------------------------
  # impl init with constructors
  # -------------------------------------------------------------------------

  record Widget:
    var label: string = "default"
    var size: int = 0

  impl Widget:
    method init(lbl: string; sz: int) =
      this.label = lbl
      this.size = sz

  record Container[T]:
    var payload: T

  impl Container:
    method init(val: T) =
      this.payload = val

  suite "impl: init constructors":
    test "Widget.init(args) via impl":
      var w = Widget.init("hello", 42)
      check w.label == "hello"
      check w.size == 42

    test "newWidget(args) via impl":
      var w = newWidget("world", 99)
      check w.label == "world"
      check w.size == 99

    test "Widget.new(args) via impl":
      var w = Widget.new("test", 7)
      check w.label == "test"
      check w.size == 7

    test "no-arg constructors still work":
      var w = Widget.init()
      check w.label == "default"
      check w.size == 0

    test "generic impl init: Container[int].init(val)":
      var c = Container[int].init(42)
      check c.payload == 42

    test "generic impl init: newContainer[string](val)":
      var c = newContainer[string]("hello")
      check c.payload == "hello"

    test "generic impl init: Container[float].new(val)":
      var c = Container[float].new(3.14)
      check c.payload == 3.14