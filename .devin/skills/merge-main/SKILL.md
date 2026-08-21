---
name: merge-main
description: "Pull origin/main into the current feature branch and resolve conflicts while preserving the BaseACPClient refactor."
allowed-tools:
  - read
  - edit
  - grep
  - exec
permissions:
  allow:
    - Exec(git)
    - Exec(python)
    - Read(**)
    - Write(**)
---

# Merge `origin/main` into the current branch

Pull the latest `main` into the current branch and reconcile conflicts. This skill is tuned for the `codex/devin-acp-provider` style branch where a `BaseACPClient` refactor must survive the merge.

## Procedure

### 1. Check state
- Run `git branch --show-current` and `git status --short`.
- Abort any half-finished merge with `git merge --abort` only if the tree is actually in a merge state and you cannot continue.

### 2. Pull main
- Run `git pull origin main`.
- If it fast-forwards, verify with `git log --oneline -5` and stop.
- If it reports conflicts, continue below.

### 3. Inventory conflicts
- Run `git diff --name-only --diff-filter=U` to get the conflicted file list.
- For each file, determine whether it is ACP/copilot-specific or a general shared file.

### 4. Resolve conflicts

**ACP / copilot / grok / devin / antigravity ACP files**
- Preserve the `BaseACPClient` subclass architecture.
- When `main` adds new copilot behavior (e.g. `--acp` probe, deprecation detection, env-var handling), integrate it as a subclass hook or a module-level helper in the provider file; do not copy `main`'s monolithic `CopilotACPClient` back in.
- Update `agent/acp_client_base.py` only when the new provider capability genuinely needs a shared hook (e.g. `_pre_spawn_check` for the `--acp` probe).
- Adjust `tests/agent/test_copilot_acp_client.py` so patch targets match the refactored code (`agent.acp_client_base.subprocess.Popen`, `_run_conversation_prompt`, etc.). Move provider-agnostic content-part tests to `tests/agent/test_acp_client_base.py`.

**General shared files**
- `hermes_cli/commands.py`: keep the cross-platform `os.path.commonpath` skill-root filtering; merge in `main`'s `get_project_skills_dirs` support by extending `_allowed_roots`.
- `gateway/run.py`: keep both billing and connection error regexes; they are independent features.
- Desktop/TS files: use `git merge-file -p --union` for adjacent independent tests, then fix the missing braces `})`. If `main` has refactored a file into a directory (e.g. `gateway-event.ts` → `gateway-event/`), accept `main`'s structure and `git rm` the old monolithic file; port only critical behavior if it is missing.
- Docs (`website/docs/integrations/providers.md`): union the provider tables and ACP sections.

### 5. Verify
- Run `python -m py_compile` on every changed Python file.
- Run `python -m pytest tests/agent/test_copilot_acp_client.py tests/agent/test_acp_client_base.py -q`.
- If a test is structurally incompatible after the refactor, fix the test (e.g. patch the method the refactor actually calls) rather than un-refactoring the code.

### 6. Commit
- Ensure `git diff --name-only --diff-filter=U` returns nothing.
- `git add` all resolved files.
- Commit with a concise message (≤150 chars, verb-first, explains why): `Merge origin/main into <branch>. Resolve N conflicts: <high-level strategy>.`

## Pitfalls
- Do not blindly use `git checkout --theirs` on ACP files; it will undo the `BaseACPClient` refactor.
- Do not accept `main`'s `startswith` skill filtering on Windows unless the skill paths are guaranteed to use `/` separators.
- Do not drop the billing error regex in `gateway/run.py`; it is required for Grok Build balance-exhausted messages.
