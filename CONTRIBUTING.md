# Contributing

Thanks for your interest in **blackholesim**.

This repository is an active scientific/numerical project. Contributions are welcome, but correctness and reproducibility take priority over feature count.

## Before opening a pull request

Please:

1. keep the change focused;
2. explain the physical or numerical motivation;
3. document assumptions and conventions;
4. add or update tests;
5. run the relevant validation suites;
6. avoid mixing unrelated renderer, GRHD/GRMHD and documentation changes in one PR.

## Scientific changes

A new numerical or physical method should normally be accompanied by at least one of:

- an analytic solution;
- an invariant;
- an independent implementation;
- a published reference result;
- a convergence study;
- a symmetry or adversarial counterexample.

A result that merely looks plausible is not sufficient validation.

## Numerical guards

Floors, fallbacks, clamps and repairs must be:
- explicit;
- documented;
- instrumented;
- absent from validation runs unless the test is specifically designed to exercise them.

Do not silently repair an invalid state.

## Code conventions

- C++17 for the current codebase.
- Prefer clear numerical contracts over clever abstractions.
- Keep CPU/CUDA behavior aligned when a feature exists in both paths.
- State coordinate, metric and unit conventions explicitly.
- Avoid hidden global state in scientific kernels.
- Keep tests deterministic.

## Commit and PR style

Use concise, descriptive commit messages, for example:

```text
grmhd: add Bondi stationary-flow validation
renderer: align CUDA redshift convention with CPU
docs: document FMKS snapshot contract
```

A PR description should include:
- what changed;
- why it changed;
- tests executed;
- numerical results;
- known limitations.

## Validation

Run the relevant CMake test targets before submission. For changes that affect the renderer, include the Carter-constant regression result. For solver changes, report convergence/error metrics rather than only pass/fail status.

## Scope

Large architectural changes should be discussed before implementation. In particular:
- primitive-variable conventions;
- coordinate systems;
- HDF5 schema;
- GRMHD recovery;
- constrained transport;
- polarized-transfer conventions.

These choices affect multiple milestones and should not change implicitly.
