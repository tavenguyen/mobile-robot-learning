---
name: python-workflow
description: Implement, debug, test, or refactor Python code using the project's interpreter and existing tooling. Use for Python source changes; preserve project conventions and verify observable behavior.
---

# Python workflow

## Establish the project context

Read applicable project instructions and inspect the affected code, callers, dependency manifests, and test configuration. Select the existing interpreter and package manager before running commands. In this project, use the system `python3` (currently 3.11) on Linux and macOS; the checked-in `.venv` is a Windows-style environment (`Scripts/`, `Include/`) and is not usable on Linux. On Windows, prefer `.venv/Scripts/python.exe` when present. Confirm imports rather than assuming the environment is isolated or complete. Use `python3 -m <tool>` through that interpreter for tools such as unittest and pytest.

Reuse configured test, formatting, lint, and type-check tools. If a tool is absent, report that limit and use an appropriate available check; do not silently install packages or replace the environment. When dependency changes are part of the request, follow the existing dependency workflow. Keep commands compatible with the current shell.

## Choose the relevant workflow

- **Feature:** inspect nearby conventions and callers, define the expected inputs and outputs, then make the smallest coherent change. Preserve public interfaces unless changing them is requested or necessary and explained. Add useful type annotations to new interfaces consistent with the project; avoid unrelated annotation churn.
- **Bug:** reproduce the reported behavior with a small input or existing test, identify the cause, patch it, and rerun the original reproduction. Add a regression test when the bug and impact warrant one. If reproduction is impossible, state what evidence is missing instead of claiming a confirmed cause.
- **Refactor:** record representative outputs or run existing tests before editing, preserve signatures and behavior, and compare afterward. Avoid combining unrelated behavior changes with restructuring.
- **Performance:** measure a representative workload before optimizing. Compare timing and output correctness afterward; account for measurement noise. Do not infer a speedup from shorter code alone.

For numeric Python, check array shapes, units, and dtype where they affect the result. Compare floating-point outputs using tolerances justified by the algorithm and scale. Keep GUI rendering out of pure calculation tests; a headless plotting check does not establish that an interactive window works.

## Verify and deliver

Run the checks relevant to the affected behavior. Prefer existing tests; add focused behavior or boundary tests for meaningful changes rather than tests that copy the implementation. For a small reversible edit, an appropriate smoke check can suffice. Check success, invalid input, and regression cases when relevant to the contract.

Keep temporary experiments separate from production code. Inspect the final diff for unrelated edits and avoid changing environment configuration merely to make a check pass. If dependencies, network access, or permissions block verification, report the exact limitation and distinguish completed checks from unverified behavior.

Respond in Vietnamese unless the user requests another language. Summarize the changed behavior, checks actually run and their outcomes, and any remaining limits. Include a runnable command when it helps the user use the result.
