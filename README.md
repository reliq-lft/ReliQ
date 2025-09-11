# ReliQ Lattice Field Theory Framework
[![License: GPL v2](https://img.shields.io/badge/license-MIT-blue)](https://github.com/reliq-lft/ReliQ/blob/main/LICENSE) 
[![Python](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://www.python.org)
> "We all make choices. But in the end our choices make us." - Andrew Ryan (BioShock)

![](https://github.com/reliq-lft/ReliQ/blob/main/reliq/reliq.png)


`ReliQ` is an experimental lattice field theory framework written first and foremost with user-friendliness, performance, reliability, and portability across current and future heterogeneous architectures in mind. As such, `ReliQ` is written in the elegant [Nim](https://nim-lang.org/) programming language, and we are experimenting with the use of a [partitioned global address space](https://en.wikipedia.org/wiki/Partitioned_global_address_space) for `ReliQ`'s distributed memory model using the [Unified Parallel C++ (UPC++)](https://upcxx.lbl.gov/docs/html/guide.html) framework. Each address space operates on the shared memory model implemented by [Kokkos](https://kokkos.org/). 

> Please note that `ReliQ` is in the early development stage; it is not ready for production. However, if you like what you're seeing, we could use help; please contact us at [reliq-lft@protonmail.me](reliq-lft@protonmail.me). Otherwise, you can follow us by clicking the `Follow` button in the top right corner of this page or our [organization page](https://github.com/reliq-lft).
>
> Sections of this document that are labelled "__unstable__" are under active development/testing.

## Documentation (__unstable__)

Very basic documentation for `ReliQ` can be found [here](https://reliq-lft.github.io/ReliQ/). As we continue to develop `ReliQ`, the documentation page will be updated/improved.

## Installation (__unstable__)

The evolving design of `ReliQ`'s build system centers on ease of use and flexibility. Users can opt to have `ReliQ` figure out what needs to be built/installed or they can build/install the portions of `ReliQ` that they wish, and `ReliQ`'s build system will simply figure out the wiring. 

#### Note on build process and Python

Installing `ReliQ` requires `Python3`, which already exists on most systems. As long as `/user/bin/python3` exists, you should have no problem with the build process. Additionally, the `ReliQ` bootstrap script will perform a local install of the Python modules under `setup/requirements.txt` before it executes; these are needed to help `ReliQ` gather information about available hardware resources. 

### Simplest method

The simplest installation method gives `ReliQ` complete control over the build process; it will figure out what hardware is available, and it will install `Nim`, `UPC++`, and `Kokkos` appropriately. 

#### Step 1: Get `ReliQ`

We first need to clone `ReliQ`.
```
cd <reliq_src>
git clone https://github.com/reliq-lft/ReliQ.git
```
Replace `<reliq_src>` with the location that you'd like the `ReliQ` source to be located.

#### Step 2: Choose `ReliQ` build location

Now let's create a local build directory. 
```
mkdir -p <reliq_build>
cd <reliq_build>
```
Replace `<reliq_build>` with the location that you'd like `ReliQ` to be installed. 

#### Step 3: Install `ReliQ` dependencies

We now need to get `Nim`, `UPC++`, and `Kokkos`. Make sure that you are in you `<reliq_build>` directory. From there, execute the following.
```
<reliq_src>/bootstrap
```
This installs the aforementioned dependencies.

#### Step 4: Configure `ReliQ`

Finally, we need to configure `ReliQ`. From within `<reliq_build>`, execute the following.
```
<reliq_src>/configure
```
And that's it! 

#### More information (optional read)

The `<reliq_src>/bootstrap` script performs a local installation of [Spack](https://spack.io/) before using `Spack` it to install `Nim`, `UPC++`, and `Kokkos`. The following directory is created locally. 

* `<reliq_build>/external`: The local install of `Spack`, `Nim`, `UPC++`, and `Kokkos`.

The `<reliq_src>/bootstrap` script also comes with a number of options that are configurable by specifying flags. To see available options, run
```
<reliq_src>/bootstrap --help
```
or `<reliq_src>/bootstrap -h`.

Upon execution, the configuration script figures out what flags are needed to link and include `UPC++` and `Kokkos` in `ReliQ` programs. It will also create the following directories under `<reliq_build>`.

* `<reliq_build>/bin`: Path to compiled `ReliQ` programs. See "Compiling and running `ReliQ` programs" below for more information.
* `<reliq_build>/cache`: Path to `Nim` cache; upon invoking the `Nim` compiler, `Nim` code is transpiled to `C++` before said `C++` code is compiled down to machine code. The `<reliq_build>/cache` directory stores the aforementioned transpiled `C++` code.

The configuration script also creates following symbolic links under `<reliq_build>`.

* `<reliq_src>/src`: Symbolic link to `ReliQ` primary source files.
* `<reliq_src>/build`: Symbolic link to `ReliQ` source files used for building/installing `ReliQ`.

## Compiling and running `ReliQ` programs (__unstable__)

### Compiling `ReliQ` programs under `src`

It is implied that the user has thusfar built/installed `ReliQ`. Within the path `<reliq_build>` where you built/installed `ReliQ`, you can compile any `Nim` file under `<reliq_src>/src` by running the following.
```
make <program>
```
Here, `<program>` should be replaced with the name of said source file; e.g., `make simplecubiclattice`. If the compilation is successful, you should see a corresponding binary file with the same name under `<reliq_build>/bin`; e.g., `<reliq_build>/bin/simplecubiclattice`. 

### Running `ReliQ` programs

To run `ReliQ` programs, we have provided an optional parallel launcher that can (but is not required to) take the place of `mpirun`, `srun`, et cetera. To execute a program with `ReliQ`'s parallel launcher __locally__, run the following.
```
<reliq_build>/reliq -e <program> -n 2 -t 2 --local
```
Here, `<program>` refers to the name of a compiled `ReliQ` program under `<reliq_build>/bin`. The above example uses the following flags. For example, `<reliq_build>/reliq -e simplecubiclattice -n 2 -t 2 --local`.

* __`-e` (or `--executable`):__ Specifies executable name under `<reliq_build>/bin` to run
* __`-n` (or `--ntasks`):__ Number of tasks to run job with
* __`-t` (or `--nthreads`):__ Number of threads per task to run job with
* __`--local`:__ Runs with `-localhost` flag specified by `UPC++`; this is needed for running locally.

For help, run
```
<reliq_build>/reliq --help
```
or `<reliq_build>/reliq -h`. 
