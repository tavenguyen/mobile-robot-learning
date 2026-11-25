# Downloaded review skills

Retrieved on 2026-11-05 through web retrieval of raw GitHub files. Shell network access is blocked (WinError 10013). Existing python-workflow was unchanged at retrieval time; it was later updated locally on 2026-11-11 for the Linux environment (see below).

| Local folder | Upstream repository | Upstream path | Ref | License |
|---|---|---|---|---|
| code-review | https://github.com/vercel-labs/open-agents | .agents/skills/code-review | main, commit unresolved | MIT, LICENSE.md included |
| clean-code | https://github.com/wondelai/skills | clean-code | main, commit unresolved | MIT, LICENSE included |

Status: installed for local use on 2026-11-11 through the project `opencode.json` (`"skills": ["./skills"]`), so OpenCode discovers `code-review`, `clean-code`, and `python-workflow` automatically. The two downloaded snapshots are still NOT commit-pinned; GitHub API and git commit lookup were unavailable when retrieved. Web responses may be cached, so files are not guaranteed to represent a single atomic upstream revision. Do not treat these as reproducible pinned installations.

code-review contains SKILL.md. clean-code contains SKILL.md and six linked reference documents. Upstream text is preserved from line-numbered raw web responses after removing retrieval line prefixes. No upstream scripts were executed.

python-workflow is the project's own skill, not an upstream snapshot. On 2026-11-11 its interpreter guidance was updated for the current environment: use the system `python3` (3.11) on Linux, because the checked-in `.venv` is a Windows-style environment (`Scripts/`, `Include/`) that cannot run here. Windows guidance for `.venv/Scripts/python.exe` is preserved.

## Usage

Load the skill by ID (`@code-review`, `@clean-code`, or `@python-workflow`) or read `skills/<id>/SKILL.md` directly.

For code-review: review a specified diff; report only, do not edit code. This skill focuses on changes, not an unrestricted whole-project audit. If the project has no Git history, supply the before/after code or a diff.

For clean-code: assess readability of `src/simulation.py`; report only, do not edit code.

clean-code mentions refactoring-patterns and clean-architecture; those separate skills are not installed. Its scoring and small-function preferences are review heuristics, not authorization to refactor or override project conventions. Separate behavioral defects from maintainability suggestions. User scope and project instructions take precedence.

OpenCode discovers all three skills through the project `opencode.json` skills array; no copying or symlinks are used, so the files in this directory remain the single source. The relative path `./skills` resolves from the active working directory, so it assumes OpenCode starts at the project root; if it is started from a subdirectory, change the entry to the absolute path `/usr/src/app/skills`. To finish a pinned installation later, resolve upstream commits, retrieve these folders at those commits, and compare before replacing snapshots.

## Verification

- code-review: basic metadata and local reference links passed
- clean-code: basic metadata and local reference links passed
- Discovery: `opencode.json` registers `./skills`; on 2026-11-11 OpenCode advertised `code-review`, `clean-code`, and `python-workflow` with their descriptions.
- Official quick_validate.py could not run: PyYAML is unavailable. Basic checks do not replace YAML validation.
- Manual review exercise: changing average(xs) from sum(xs)/len(xs) to sum(xs)/(len(xs)-1) creates wrong outputs and division by zero for one item; report as a behavioral defect.
- Manual readability exercise: a function named f combining calculation and printing warrants a naming/responsibility suggestion, not an automatic bug finding or edit. These are manual walkthroughs, not independent agent evaluations.

## Local file SHA-256

- skills/python-workflow/SKILL.md: ce7444ea5019a8fe477c5594f4a07c46b5d99ce0af0bcd0f3c5d7d6d0fea5039
- skills/code-review/LICENSE.md: 7e3c5c6f3d3d9ae480aad1f7274c2ac641390c77589b3b4eea2dabb5dced7a11
- skills/code-review/SKILL.md: 7092290f26448781304bdbbd2e5a7311338774a7f265e5d2c052787b7faffe86
- skills/clean-code/LICENSE: 320c5d43b6337e8857f495130af0bdff209716ed613a1067676df701d4089057
- skills/clean-code/references/code-smells.md: 35ce4f6e123e37377d3df550b436c44954e23b64018e3ff4b875573844c7460e
- skills/clean-code/references/comments-formatting.md: d9abd8f14badb47370d329e7a1c97bac3c518aa6e3df2c0f1a60256b6a7199b6
- skills/clean-code/references/error-handling.md: 216290e38e39ca0bd20ab9ab100489f6fbf8f907e2f7ffa59936c5ebf851e6ce
- skills/clean-code/references/functions-and-methods.md: 31cc9408b2c68a933b1382f99772e3fb71b288b1b29ac6d5b886470783a810b7
- skills/clean-code/references/naming-conventions.md: fd9a476d531ba9cdea93107f9c585205c38b2f419fe4d43986addc02fe31e7ef
- skills/clean-code/references/testing-principles.md: 3cd0ead70319aa446ae01a53c003416397acaec796d08355cb824caac9549023
- skills/clean-code/SKILL.md: b2687ed94360089356672bfafad285e61712cde67bf0d3861d462355eb02453f
