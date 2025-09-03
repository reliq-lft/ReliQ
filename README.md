# ReliQ
> "We all make choices. But in the end our choices make us." - Andrew Ryan (BioShock)

`ReliQ` is an experimental lattice field theory framework written first and foremost with user-friendliness, performance, reliability, and portability across current and future heterogeneous architectures in mind. As such, `ReliQ` is written in the elegant [Nim](https://nim-lang.org/) programming language. We are experimenting with the use of a [partitioned global address space](https://en.wikipedia.org/wiki/Partitioned_global_address_space) for `ReliQ`'s distributed memory model using the [Unified Parallel C++](https://upcxx.lbl.gov/docs/html/guide.html) framework. Each address space operates on the shared memory model implemented by [Kokkos](https://kokkos.org/).
