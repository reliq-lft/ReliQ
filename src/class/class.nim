#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/classes/class.nim
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

## Chapel-style class macro for Nim.
##
## Overview
## --------
## The ``class`` macro defines **reference-type objects** that follow Chapel's
## class semantics (https://chapel-lang.org/docs/language/spec/classes.html).
## A class is a Nim ``ref object``: it lives on the heap, is managed by
## reference, supports inheritance, and can be ``nil``.
##
## Key Differences from Records
## -----------------------------
## =========================  =====================  =====================
## Feature                    Record                 Class
## =========================  =====================  =====================
## Nim type                   ``object``             ``ref object``
## Semantics                  Value (copy on assign) Reference (shared ref)
## Inheritance                Not supported          Single inheritance
## Dispatch                   Static (``proc``)      Dynamic (``method``)
## Nilability                 Never nil              Can be nil
## Memory                     Stack (automatic)      Heap (GC / ref counted)
## Identity                   No identity            Identity (``===``)
## Copy                       ``copyInit``           ``clone`` (explicit)
## =========================  =====================  =====================
##
## Classes automatically receive compiler-generated:
##   - **Default initializer** (``init``) with field defaults
##   - **Convenience constructors** (``ClassName.init()``, ``newClassName()``,
##     ``ClassName.new()``)
##   - **Clone** (``clone``) for explicit deep copy
##   - **Borrow** (``borrow``) — returns the same reference (Chapel borrowing)
##   - **Comparison operators** (``==``, ``!=``, ``<``, ``<=``) — field-wise
##   - **Identity operators** (``===``, ``!==``) — reference comparison
##   - **Hash function** (``hash``, compatible with ``std/hashes``)
##   - **String representation** (``$``) in Chapel-style format
##   - **Destructor** (``destroy``) when ``deinit()`` is defined
##   - **Meta helpers** (``className()`` static method, ``isNilClass()``,
##     ``isInstanceOf(T)``, ``parentClassName()``)
##   - **Auto-export** of all fields and methods
##
## All of these can be overridden by the user externally.
##
## Quick Start
## -----------
## .. code-block:: nim
##   import class
##
##   class Actor:
##     var name = "unknown"
##     var age: int = 0
##
##     method init() =
##       echo "Creating an Actor"
##
##     method greet() =
##       echo "Hi, I'm " & this.name
##
##   var a = Actor.init()         # or newActor() or Actor.new()
##   a.name = "Alice"
##   a.greet()                    # "Hi, I'm Alice"
##
##   var b = a                    # reference copy (same object!)
##   b.name = "Bob"
##   assert a.name == "Bob"       # both point to the same instance
##   assert a === b               # identity check: same object
##
##   var c = a.clone()            # explicit deep copy
##   assert a == c                # field-wise equal
##   assert not (a === c)         # different objects
##
## Inheritance
## -----------
## .. code-block:: nim
##   class Animal:
##     var name: string = "animal"
##
##     method speak(): string =
##       return this.name & " says ..."
##
##   class Dog of Animal:
##     var breed: string = "mutt"
##
##     method speak(): string {.override.} =
##       return this.name & " says woof!"
##
##   var d = Dog.init()
##   d.name = "Rex"
##   d.breed = "German Shepherd"
##   echo d.speak()               # "Rex says woof!"
##
##   var a: Animal = d            # upcast: Dog → Animal
##   echo a.speak()               # "Rex says woof!" (dynamic dispatch)
##
## DSL Syntax
## ----------
## .. code-block:: nim
##   class ClassName:
##     var field1: Type = defaultValue
##     var field2 = "hello"
##     var field3: int
##
##     method init() =
##       echo "created"
##
##     method deinit() =
##       echo "destroyed"
##
##     method doSomething(): int =
##       return this.field3 + 1
##
##     method getName(): string {.immutable.} =
##       return this.field2
##
##     method classLevel() {.static.} =
##       echo "I am a class-level method"
##
##   class ChildClass of ClassName:
##     var extra: int = 0
##
##     method doSomething(): int {.override.} =
##       return this.field3 + this.extra + 1
##
##   class EmptyClass
##
##   class Wrapper[T]:
##     var data: T
##     var label: string = "default"
##
## Fields
## ------
## Fields are declared with ``var``. Each field must have either an explicit
## type, a default value (from which the type is inferred), or both.
##
## Methods
## -------
## Methods are declared with ``method`` in the DSL:
##   - **Instance methods** use Nim ``method`` for dynamic dispatch.
##   - **Static methods** (``{.static.}``) use ``proc``.
##   - **Immutable methods** (``{.immutable.}``) receive non-var ``this``.
##   - **Override methods** (``{.override.}``) override a parent method.
##   - ``init`` and ``deinit`` always use ``proc`` (no dynamic dispatch).
##
## Instance methods receive ``this: ClassName`` (ref, so mutation is implicit).
##
## Constructors
## ------------
## Three convenience constructor forms are auto-generated:
##
## =========================  ====================================
## Form                       Description
## =========================  ====================================
## ``ClassName.init(...)``     Static call on the type
## ``newClassName(...)``       Free-standing ``new`` + name proc
## ``ClassName.new(...)``      Alternative static ``new`` call
## =========================  ====================================
##
## These allocate a ``ref object``, call ``init()``, and optionally call
## ``postinit()`` if defined.
##
## Reference Semantics
## -------------------
## Assignment ``b = a`` makes ``b`` point to the **same** instance:
##
## .. code-block:: nim
##   var a = Actor.init()
##   a.name = "Alice"
##   var b = a           # b and a are the same object
##   b.name = "Bob"
##   assert a.name == "Bob"
##
## Use ``clone()`` for an explicit deep copy, or ``borrow()`` to express
## Chapel's borrowing intent (returns the same ref).
##
## Identity vs Equality
## --------------------
## - ``==`` / ``!=``: field-by-field equality
## - ``===`` / ``!==``: reference identity (same object in memory)
## - ``hash``: field-based hash
##
## Inheritance & Dynamic Dispatch
## ------------------------------
## Classes support single inheritance with dynamic dispatch:
##
## .. code-block:: nim
##   class Base:
##     method foo(): string =
##       return "Base"
##
##   class Derived of Base:
##     method foo(): string {.override.} =
##       return "Derived"
##
##   var b: Base = Derived.init()
##   echo b.foo()  # "Derived" (dynamic dispatch)
##
## Helper methods:
## - ``isInstanceOf(T)``: runtime type check
## - ``parentClassName()``: returns parent class name
##
## External Method Blocks (``classImpl``)
## --------------------------------------
## Methods can be added to an already-defined class:
##
## .. code-block:: nim
##   class Foo:
##     var x: int
##
##   classImpl Foo:
##     method doubled(): int =
##       return this.x * 2
##
## Destructor (``deinit``)
## -----------------------
## If a ``method deinit()`` is defined, a ``destroy`` proc is generated.
## Users can call ``destroy(obj)`` explicitly. For GC-managed refs, the
## GC handles lifetime automatically.
##
## Debugging
## ---------
## Compile with ``-d:debugclasses=true`` to see the generated AST for every
## class, or ``-d:debugclasses=MyClass`` to debug a single class.
##
## Architecture
## ------------
## The implementation mirrors the record macro architecture — a hook-based
## compile-time plugin system with the same stage pipeline:
##
## 1. ``ClassPreload`` — Early initialization.
## 2. ``ClassGatherDefinitions`` — Parse fields and methods.
## 3. ``ClassAddExtraDefinitions`` — Auto-generate ``init`` if missing;
##    generate convenience constructors.
## 4. ``ClassModifyDefinitions`` — Inject field defaults; auto-export.
## 5. ``ClassGenerateCode`` — Emit Nim AST: ref object type, methods,
##    copy/comparison/hash/meta procs, destructor.
## 6. ``ClassFinalize`` — Post-processing.
## 7. ``ClassDebugEcho`` — Diagnostics.
##
## **Module Map:**
##
## =======================  ================================================
## Module                   Responsibility
## =======================  ================================================
## ``classinternal``        Core types (``ClassDescription``,
##                          ``ClassCompilerStage``, ``ClassCompilerHook``),
##                          ``newClassDescription()``, ``compile()``
## ``classutils``           AST helpers: ``traverseClassStatementList``,
##                          ``variableName``, ``traverseParams``
## ``classvar``             Field parsing, type inference, RecList generation
## ``classmethods``         Method parsing, ``method`` vs ``proc`` dispatch,
##                          ``this`` injection, ``{.base.}`` / override
## ``classconstructors``    Default ``init``, field-default injection,
##                          convenience constructors, ``postinit`` support
## ``classexports``         Auto-export fields and methods with ``*``
## ``classcopy``            ``clone``, ``copyFrom``, ``borrow`` generation
## ``classcomparison``      ``==``, ``!=``, ``<``, ``<=``, ``===``, ``!==``
##                          operators and ``hash`` proc generation
## ``classdestructors``     ``destroy`` proc from ``deinit()``
## ``classmeta``            ``className()``, ``$``, ``isNilClass()``
## ``classinheritance``     ``isInstanceOf``, ``parentClassName()``,
##                          inheritance helpers
## ``classimpl``            ``classImpl`` macro for external method blocks
## =======================  ================================================
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

import classinternal
import classvar
import classmethods
import classconstructors
import classexports
import classcopy
import classcomparison
import classdestructors
import classmeta
import classinheritance
import classimpl

export static
export hashes
export classimpl

## Class definition with body.
macro class*(head: untyped, body: untyped): untyped =
  return newClassDescription("class", head, body).compile()

## Empty class definition (no body).
macro class*(head: untyped): untyped =
  return newClassDescription("class", head, newStmtList()).compile()

#[ unit tests ]#

when isMainModule:
  import std/[unittest, tables, sets, strutils]

  # -------------------------------------------------------------------------
  # Basic class definitions
  # -------------------------------------------------------------------------

  class Actor:
    ## A person with a name and age.
    var name: string = "unknown"
    var age: int = 0

    method init() =
      echo "Creating Actor: " & this.name

    method greet(): string =
      return "Hi, I'm " & this.name & ", age " & $this.age

    method birthday() =
      this.age += 1

    method describe() {.static.} =
      echo "Actor is a class type"

  class Empty

  class Tracked:
    var id: int = 0

    method init() =
      echo "Tracked " & $this.id & " created"

    method deinit() =
      echo "Tracked " & $this.id & " destroyed"

  class MultiField:
    var x: int = 10
    var y: int = 20
    var label: string = "point"

  class TypeInferred:
    var anInt = 42
    var aFloat = 3.14
    var aString = "hello"
    var aBool = true
    var aChar = 'z'

  class SingleField:
    var value: int = 0

  class NoInit:
    var data: int = 100

  # -------------------------------------------------------------------------
  # Inheritance definitions
  # -------------------------------------------------------------------------

  class Animal:
    var name: string = "animal"
    var sound: string = "..."

    method speak(): string =
      return this.name & " says " & this.sound

    method kind(): string =
      return "Animal"

  class Dog of Animal:
    var breed: string = "mutt"

    method speak(): string {.override.} =
      return this.name & " says woof!"

    method kind(): string {.override.} =
      return "Dog"

    method fetch(): string =
      return this.name & " fetches the ball"

  class Cat of Animal:
    var indoor: bool = true

    method speak(): string {.override.} =
      return this.name & " says meow!"

    method kind(): string {.override.} =
      return "Cat"

  class Puppy of Dog:
    var toy: string = "bone"

    method speak(): string {.override.} =
      return this.name & " says yip!"

    method kind(): string {.override.} =
      return "Puppy"

  # -------------------------------------------------------------------------
  # Test suites
  # -------------------------------------------------------------------------

  suite "class: field declarations":
    test "explicit type with default value":
      var a = Actor.init()
      check a.name == "unknown"
      check a.age == 0

    test "multiple fields with explicit types":
      var mf = MultiField.init()
      check mf.x == 10
      check mf.y == 20
      check mf.label == "point"

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

  suite "class: field mutation":
    test "simple field assignment":
      var a = Actor.init()
      a.name = "Alice"
      a.age = 30
      check a.name == "Alice"
      check a.age == 30

    test "mutation via reference":
      var a = Actor.init()
      a.name = "Alice"
      var b = a
      b.name = "Bob"
      # Both point to the same instance!
      check a.name == "Bob"
      check b.name == "Bob"

  suite "class: reference semantics":
    test "assignment shares reference":
      var a = Actor.init()
      a.name = "Alice"
      a.age = 30
      var b = a
      b.name = "Bob"
      b.age = 25
      # Both point to same object
      check a.name == "Bob"
      check a.age == 25

    test "clone creates independent copy":
      var a = Actor.init()
      a.name = "Alice"
      a.age = 30
      var c = a.clone()
      c.name = "Carol"
      c.age = 99
      check a.name == "Alice"
      check a.age == 30
      check c.name == "Carol"
      check c.age == 99

    test "borrow returns same reference":
      var a = Actor.init()
      a.name = "Alice"
      var b = a.borrow()
      check a === b
      b.name = "Bob"
      check a.name == "Bob"

  suite "class: instance methods":
    test "method with return value":
      var a = Actor.init()
      a.name = "Alice"
      a.age = 30
      check a.greet() == "Hi, I'm Alice, age 30"

    test "method that mutates fields":
      var a = Actor.init()
      a.age = 29
      a.birthday()
      check a.age == 30

    test "method mutation visible through references":
      var a = Actor.init()
      a.age = 29
      var b = a
      b.birthday()
      check a.age == 30  # same object

  suite "class: static methods":
    test "className() returns type name":
      check Actor.className() == "Actor"
      check Empty.className() == "Empty"
      check MultiField.className() == "MultiField"

    test "user-defined static method compiles":
      Actor.describe()

  suite "class: constructors — ClassName.init()":
    test "basic init":
      var a = Actor.init()
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

  suite "class: constructors — newClassName()":
    test "newActor":
      var b = newActor()
      check b.name == "unknown"
      check b.age == 0

    test "newMultiField":
      var mf = newMultiField()
      check mf.x == 10
      check mf.y == 20

    test "newEmpty":
      var e = newEmpty()
      check not e.isNil

    test "newNoInit":
      var ni = newNoInit()
      check ni.data == 100

  suite "class: constructors — ClassName.new()":
    test "Actor.new()":
      var c = Actor.new()
      check c.name == "unknown"
      c.name = "Carol"
      check c.name == "Carol"

    test "MultiField.new()":
      var mf = MultiField.new()
      check mf.x == 10

  suite "class: nil handling":
    test "nil class variable":
      var a: Actor = nil
      check a.isNil

    test "initialized class is not nil":
      var a = Actor.init()
      check not a.isNil

    test "isNilClass helper":
      var a = Actor.init()
      check not a.isNilClass()

    test "$ on nil returns nil string":
      var a: Actor = nil
      check $a == "nil"

  suite "class: equality (==, !=)":
    test "equal by fields":
      var a = Actor.init()
      a.name = "Alice"
      a.age = 30
      var b = Actor.init()
      b.name = "Alice"
      b.age = 30
      check a == b

    test "different name field":
      var a = Actor.init()
      a.name = "Alice"
      var b = Actor.init()
      b.name = "Bob"
      check a != b

    test "all defaults are equal":
      var a = Actor.init()
      var b = Actor.init()
      check a == b

    test "nil == nil":
      var a: Actor = nil
      var b: Actor = nil
      check a == b

    test "nil != non-nil":
      var a: Actor = nil
      var b = Actor.init()
      check a != b

    test "clone produces field-equal instance":
      var a = MultiField.init()
      a.x = 5
      a.y = 10
      a.label = "copy"
      var b = a.clone()
      check a == b

  suite "class: identity (===, !==)":
    test "same reference is identical":
      var a = Actor.init()
      var b = a
      check a === b

    test "different instances are not identical":
      var a = Actor.init()
      var b = Actor.init()
      check a !== b

    test "clone is not identical":
      var a = Actor.init()
      var c = a.clone()
      check a !== c
      check a == c  # but field-wise equal

  suite "class: ordering (<, <=)":
    test "< first field differs":
      var lo = Actor.init()
      lo.name = "AAA"
      var hi = Actor.init()
      hi.name = "ZZZ"
      check lo < hi
      check not (hi < lo)

    test "< second field breaks tie":
      var a = Actor.init()
      a.name = "same"
      a.age = 10
      var b = Actor.init()
      b.name = "same"
      b.age = 20
      check a < b

    test "<= with equal":
      var a = Actor.init()
      var b = Actor.init()
      check a <= b

  suite "class: hash":
    test "equal objects have equal hash":
      var a = Actor.init()
      a.name = "Alice"
      a.age = 30
      var b = Actor.init()
      b.name = "Alice"
      b.age = 30
      check hash(a) == hash(b)

    test "different objects likely have different hash":
      var a = Actor.init()
      a.name = "Alice"
      var b = Actor.init()
      b.name = "Bob"
      check hash(a) != hash(b)

    test "classes usable as Table keys":
      var t = initTable[SingleField, string]()
      var k1 = SingleField.init()
      k1.value = 1
      var k2 = SingleField.init()
      k2.value = 2
      t[k1] = "one"
      t[k2] = "two"
      check t[k1] == "one"
      check t[k2] == "two"

  suite "class: $ string representation":
    test "basic toString":
      var a = Actor.init()
      a.name = "Alice"
      a.age = 30
      check $a == "Actor(name = Alice, age = 30)"

    test "default fields toString":
      var a = Actor.init()
      check $a == "Actor(name = unknown, age = 0)"

    test "empty class toString":
      var emp = Empty.init()
      check $emp == "Empty()"

    test "nil toString":
      var a: Actor = nil
      check $a == "nil"

  suite "class: empty class":
    test "empty class can be created":
      var emp = Empty.init()
      check not emp.isNil

    test "two empty classes with same fields are equal":
      var e1 = Empty.init()
      var e2 = Empty.init()
      check e1 == e2

    test "two empty classes are NOT identical (different refs)":
      var e1 = Empty.init()
      var e2 = Empty.init()
      check e1 !== e2

    test "className for empty class":
      check Empty.className() == "Empty"

  suite "class: deinit / destructor":
    test "destroy can be called explicitly":
      var t = Tracked.init()
      t.id = 42
      t.destroy()

    test "class without deinit has no issues":
      var a = Actor.init()
      a.name = "test"
      check a.name == "test"

  # -------------------------------------------------------------------------
  # Inheritance tests
  # -------------------------------------------------------------------------

  suite "class: inheritance — basic":
    test "child inherits parent fields":
      var d = Dog.init()
      check d.name == "animal"  # inherited from Animal
      check d.breed == "mutt"   # own field

    test "child inherits parent field defaults":
      var c = Cat.init()
      check c.name == "animal"
      check c.sound == "..."
      check c.indoor == true

    test "can modify inherited fields":
      var d = Dog.init()
      d.name = "Rex"
      check d.name == "Rex"

  suite "class: inheritance — dynamic dispatch":
    test "overridden method dispatches dynamically":
      var d = Dog.init()
      d.name = "Rex"
      var a: Animal = d
      check a.speak() == "Rex says woof!"

    test "cat override":
      var c = Cat.init()
      c.name = "Whiskers"
      var a: Animal = c
      check a.speak() == "Whiskers says meow!"

    test "base method without override":
      var a = Animal.init()
      a.name = "Creature"
      a.sound = "roar"
      check a.speak() == "Creature says roar"

    test "multi-level inheritance dispatch":
      var p = Puppy.init()
      p.name = "Tiny"
      var a: Animal = p
      check a.speak() == "Tiny says yip!"
      check a.kind() == "Puppy"

  suite "class: inheritance — kind method":
    test "kind returns correct type":
      check Animal.init().kind() == "Animal"
      check Dog.init().kind() == "Dog"
      check Cat.init().kind() == "Cat"
      check Puppy.init().kind() == "Puppy"

    test "kind via base class reference":
      var animals: seq[Animal] = @[
        Animal.init(),
        Dog.init(),
        Cat.init(),
        Puppy.init()
      ]
      check animals[0].kind() == "Animal"
      check animals[1].kind() == "Dog"
      check animals[2].kind() == "Cat"
      check animals[3].kind() == "Puppy"

  suite "class: inheritance — child-specific methods":
    test "child-only method":
      var d = Dog.init()
      d.name = "Rex"
      check d.fetch() == "Rex fetches the ball"

  suite "class: inheritance — type checking":
    test "isInstanceOf":
      var d = Dog.init()
      check d.isInstanceOf(Dog)
      check d.isInstanceOf(Animal)
      check d.isInstanceOf(RootObj)

    test "isInstanceOf negative":
      var a = Animal.init()
      check not (a of Dog)
      check not (a of Cat)

    test "puppy is instance of all ancestors":
      var p = Puppy.init()
      check p.isInstanceOf(Puppy)
      check p.isInstanceOf(Dog)
      check p.isInstanceOf(Animal)

  suite "class: inheritance — parentClassName":
    test "parentClassName":
      check Animal.parentClassName() == "RootObj"
      check Dog.parentClassName() == "Animal"
      check Cat.parentClassName() == "Animal"
      check Puppy.parentClassName() == "Dog"

  suite "class: inheritance — polymorphic collections":
    test "seq[Animal] holds mixed types":
      var animals: seq[Animal] = @[]
      var a = Animal.init()
      a.name = "Generic"
      a.sound = "hmm"
      animals.add(a)

      var d = Dog.init()
      d.name = "Rex"
      animals.add(d)

      var c = Cat.init()
      c.name = "Whiskers"
      animals.add(c)

      check animals.len == 3
      check animals[0].speak() == "Generic says hmm"
      check animals[1].speak() == "Rex says woof!"
      check animals[2].speak() == "Whiskers says meow!"

  # -------------------------------------------------------------------------
  # Generic class definitions
  # -------------------------------------------------------------------------

  class Box[T]:
    var value: T

  class Pair[A, B]:
    var first: A
    var second: B

    method swap(): Pair[B, A] =
      var r = Pair[B, A].init()
      r.first = this.second
      r.second = this.first
      return r

  class NumBox[T: SomeNumber]:
    var value: T

    method doubled(): T =
      return this.value * 2

  class GenericWithDefault[T]:
    var data: T
    var label: string = "unlabelled"

  class EmptyGeneric[T]

  # -------------------------------------------------------------------------
  # Generic test suites
  # -------------------------------------------------------------------------

  suite "generic class: basic construction":
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

  suite "generic class: constrained generics":
    test "NumBox[int]":
      var n = NumBox[int].init()
      n.value = 21
      check n.doubled() == 42

    test "NumBox[float]":
      var n = NumBox[float].init()
      n.value = 1.5
      check n.doubled() == 3.0

  suite "generic class: methods":
    test "Pair swap method":
      var p = Pair[int, string].init()
      p.first = 42
      p.second = "answer"
      let swapped = p.swap()
      check swapped.first == "answer"
      check swapped.second == 42

  suite "generic class: reference semantics":
    test "assignment shares reference":
      var a = Box[int].init()
      a.value = 10
      var b = a
      b.value = 20
      check a.value == 20  # same object

    test "clone produces independent copy":
      var a = Box[int].init()
      a.value = 10
      var c = a.clone()
      c.value = 20
      check a.value == 10
      check c.value == 20

  suite "generic class: equality":
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

  suite "generic class: identity":
    test "same ref is identical":
      var a = Box[int].init()
      var b = a
      check a === b

    test "different refs are not identical":
      var a = Box[int].init()
      a.value = 42
      var b = Box[int].init()
      b.value = 42
      check a !== b

  suite "generic class: $ string representation":
    test "Box[int] toString":
      var b = Box[int].init()
      b.value = 42
      check $b == "Box(value = 42)"

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

  suite "generic class: meta":
    test "className for generic classes":
      check Box[int].className() == "Box"
      check Pair[int, string].className() == "Pair"
      check NumBox[float].className() == "NumBox"

  suite "generic class: hash":
    test "equal Box[int] same hash":
      var a = Box[int].init()
      a.value = 42
      var b = Box[int].init()
      b.value = 42
      check hash(a) == hash(b)

    test "generic classes in Table":
      var t = initTable[Box[int], string]()
      var k1 = Box[int].init()
      k1.value = 10
      var k2 = Box[int].init()
      k2.value = 20
      t[k1] = "ten"
      t[k2] = "twenty"
      check t[k1] == "ten"
      check t[k2] == "twenty"

  # -------------------------------------------------------------------------
  # classImpl block definitions
  # -------------------------------------------------------------------------

  classImpl Actor:
    method fullGreet(): string =
      return "Hello! " & this.greet()

    method setName(n: string) =
      this.name = n

    method isAdult(): bool {.immutable.} =
      return this.age >= 18

    method typeName(): string {.static.} =
      return "Actor"

  classImpl MultiField:
    method sum(): int {.immutable.} =
      return this.x + this.y

    method reset() =
      this.x = 0
      this.y = 0
      this.label = ""

  classImpl Box:
    method show(): string {.immutable.} =
      return "Box(" & $this.value & ")"

  classImpl NumBox:
    method tripled(): T {.immutable.} =
      return this.value * 3

  classImpl Pair:
    method description(): string {.immutable.} =
      return "(" & $this.first & ", " & $this.second & ")"

  # -------------------------------------------------------------------------
  # classImpl test suites
  # -------------------------------------------------------------------------

  suite "classImpl: instance methods":
    test "classImpl method can call original methods":
      var a = Actor.init()
      a.name = "Alice"
      a.age = 30
      check a.fullGreet() == "Hello! Hi, I'm Alice, age 30"

    test "classImpl method can mutate fields":
      var a = Actor.init()
      a.setName("Bob")
      check a.name == "Bob"

    test "classImpl method on another class":
      var mf = MultiField.init()
      check mf.sum() == 30  # 10 + 20
      mf.reset()
      check mf.x == 0
      check mf.y == 0
      check mf.label == ""

  suite "classImpl: immutable methods":
    test "immutable classImpl method":
      var a = Actor.init()
      a.age = 20
      check a.isAdult() == true

    test "immutable classImpl on let value":
      let mf = MultiField.init()
      check mf.sum() == 30

  suite "classImpl: static methods":
    test "static classImpl method":
      check Actor.typeName() == "Actor"

  suite "classImpl: generic classes":
    test "classImpl method on Box[int]":
      var b = Box[int].init()
      b.value = 42
      check b.show() == "Box(42)"

    test "classImpl method on Box[string]":
      var b = Box[string].init()
      b.value = "hello"
      check b.show() == "Box(hello)"

    test "classImpl method on constrained generic NumBox":
      var n = NumBox[int].init()
      n.value = 7
      check n.tripled() == 21

    test "classImpl method on Pair":
      var p = Pair[int, string].init()
      p.first = 42
      p.second = "answer"
      check p.description() == "(42, answer)"

  # -------------------------------------------------------------------------
  # classImpl init with constructors
  # -------------------------------------------------------------------------

  class Widget:
    var label: string = "default"
    var size: int = 0

  classImpl Widget:
    method init(lbl: string; sz: int) =
      this.label = lbl
      this.size = sz

  class Container[T]:
    var payload: T

  classImpl Container:
    method init(val: T) =
      this.payload = val

  suite "classImpl: init constructors":
    test "Widget.init(args) via classImpl":
      var w = Widget.init("hello", 42)
      check w.label == "hello"
      check w.size == 42

    test "newWidget(args) via classImpl":
      var w = newWidget("world", 99)
      check w.label == "world"
      check w.size == 99

    test "Widget.new(args) via classImpl":
      var w = Widget.new("test", 7)
      check w.label == "test"
      check w.size == 7

    test "no-arg constructors still work":
      var w = Widget.init()
      check w.label == "default"
      check w.size == 0

    test "generic classImpl init: Container[int].init(val)":
      var c = Container[int].init(42)
      check c.payload == 42

    test "generic classImpl init: newContainer[string](val)":
      var c = newContainer[string]("hello")
      check c.payload == "hello"

    test "generic classImpl init: Container[float].new(val)":
      var c = Container[float].new(3.14)
      check c.payload == 3.14

  # -------------------------------------------------------------------------
  # Comprehensive integration tests
  # -------------------------------------------------------------------------

  suite "class: comprehensive integration":
    test "full lifecycle: create, mutate, clone, compare, stringify, hash":
      # Create
      var a = Actor.init()
      a.name = "Alice"
      a.age = 30

      # Mutate via method
      a.birthday()
      check a.age == 31

      # Clone (deep copy)
      var b = a.clone()
      check a == b       # field-wise equal
      check a !== b      # different objects

      # Mutate clone independently
      b.name = "Bob"
      check a.name == "Alice"
      check b.name == "Bob"

      # Reference semantics
      var c = a
      c.name = "Carol"
      check a.name == "Carol"  # same object
      check a === c

      # Reset
      a.name = "Alice"

      # Ordering
      check a < b  # "Alice" < "Bob"

      # String
      check $a == "Actor(name = Alice, age = 31)"

      # Hash consistency
      var d = Actor.init()
      d.name = "Alice"
      d.age = 31
      check hash(a) == hash(d)

      # Method
      check a.greet() == "Hi, I'm Alice, age 31"

    test "polymorphic collection with dynamic dispatch":
      var animals: seq[Animal] = @[]
      for i in 0..2:
        var d = Dog.init()
        d.name = "Dog" & $i
        animals.add(d)
      for i in 0..1:
        var c = Cat.init()
        c.name = "Cat" & $i
        animals.add(c)

      check animals.len == 5
      for a in animals:
        if a of Dog:
          check a.speak().contains("woof")
        elif a of Cat:
          check a.speak().contains("meow")

    test "three-level inheritance":
      var p = Puppy.init()
      p.name = "Tiny"
      p.breed = "Chihuahua"
      p.toy = "squeaker"
      check p.speak() == "Tiny says yip!"
      check p.kind() == "Puppy"
      check p.fetch() == "Tiny fetches the ball"

      var animal: Animal = p
      check animal.speak() == "Tiny says yip!"
      check animal.kind() == "Puppy"
