---
name: caveman-commit
description: Commit finished work in caveman English, one commit per completed feature, fix, or refactor. Use after tests pass at the end of a unit of work, when the user asks to commit, or when the project rule says to commit each done feature. Never push.
---

# Caveman commit

Commit when one unit of work is done. Feature done. Fix done. Refactor done. Test pass. Then commit once.

## Before commit

1. Check what changed: `git status --short` and `git diff`.
2. Run the checks relevant to the work. Do not commit when the related tests or checks fail. Fix first, or report the failure and stop.
3. Stage only the files that belong to the finished work. Do not stage unrelated edits, other half-finished changes, `.venv`, `__pycache__`, secrets, logs, or temporary files.
4. Inspect the staged diff: `git diff --cached --stat` and `git diff --cached`. Confirm only intended files are staged.

Stage with explicit paths, for example `git add path/one.py path/two.py`. Never use `git add -A` or `git add .` when unrelated changes exist.

## Commit message

Write in English. Short caveman style. Present or bare imperative, all caps optional but consistent. Say what changed and that checks passed.

Shape: `<VERB> <THING>. <RESULT>.`

Examples:

- `ADD BUG1 FULL LOOP LEAVE POINT. TESTS PASS.`
- `FIX TANGENT BUG LEAVE LINE. TESTS PASS.`
- `ADD CAVEMAN COMMIT SKILL. TESTS PASS.`
- `REFACTOR RANGE SENSOR RAYCAST. TESTS PASS.`

Keep one line for small work. Add a short body only when the reason is not obvious. No emojis. No marketing tone.

## Commit

Run plain `git commit`. If the repository needs a safe-directory override, use `git -c safe.directory=<path> ...`; this is allowed.

Never use `--no-verify`, `--force`, `--amend`, `rebase`, or `push`. Never rewrite history. Never push in any situation unless the user explicitly overrides the no-push rule in a later message, and even then confirm first.

One completed unit of work gets exactly one commit. Do not split into many commits and do not bunch several unrelated features into one.

## After commit

Report the commit hash, the committed files, the checks that passed, and state clearly that nothing was pushed.
