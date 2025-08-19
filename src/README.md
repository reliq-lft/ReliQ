## `src`

The organization of `ReliQ`'s top-level `src` tree is designed to separate exposure to `ReliQ` varying degrees of source depth. It is expected that most users will simply use the code that is exported by `src/reliq.nim`; we'll refer to this as "tier 1" source code. Tier 2 source code is then categorized as any other source that exports `ReliQ`'s functionalities through imports of other `src`-level modules. Tier 3 source must be imported by specifying the `src` directory and the file contained within that directory containing the target symbols. 

What is and what is not exposed to the user is subject to change as `ReliQ` continues to evolve alongside the demands of its userbase. The `ReliQ` developers should consider rolling out regular stable releases that allow users to pick and choose versions that they have grown accustomed to; albeit, at the cost of newer features. 

Speaking from experience, this is not unlike how many folks have used the `MILC` code base, especially in the beyond Standard Model physics community, of which many prefer to stick with `MILCv6` over the latest iteration of `MILC`.