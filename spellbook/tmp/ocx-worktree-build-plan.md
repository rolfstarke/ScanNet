# Action plan: install kdcokenny/opencode-worktree (OCX) — global install

For: build agent · Context: research report at spellbook/tmp/ocx-worktree-research.md (read it first)
Goal: OCX binary + worktree plugin installed GLOBALLY (~/.config/opencode), smoke-tested in /home/rolf/GIT/test2, ScanNet repo untouched except pre-created `.opencode/worktree.jsonc`.

## Decision (user-approved)
- Global install: `ocx add kdco/worktree --from https://registry.kdco.dev --global`
- ScanNet gets only a pre-created `.opencode/worktree.jsonc` (untracked; DO NOT commit, DO NOT touch other files)

## Steps

### 1. Install OCX binary
```
curl -fsSL https://ocx.kdco.dev/install.sh | sh
```
- Expect install to `~/.local/bin/ocx` (no sudo available; /usr/local/bin not writable).
- Verify: `~/.local/bin/ocx --version` (expect >= 1.0.16). If `ocx` not on PATH in new shells: `export PATH="$HOME/.local/bin:$PATH"` (already on PATH per research).

### 2. Install worktree component globally
```
cd ~/.config/opencode && ocx add kdco/worktree --from https://registry.kdco.dev --global
```
- Verify:
  - `~/.config/opencode/plugin/worktree.ts` + `plugin/worktree/{state,terminal,launch-context}.ts` exist
  - `~/.config/opencode/plugin/kdco-primitives/` has 10 files
  - `~/.config/opencode/package.json` now contains deps `jsonc-parser@3.3.1` + `zod@4.3.5` (merged with existing `@opencode-ai/plugin@1.18.11` — keep it intact)
  - `~/.config/opencode/opencode.json` UNCHANGED (OCX must NOT touch the plugin array; if it did, restore from git backup of the file)
- Back up `~/.config/opencode/package.json` + `opencode.json` before running (`cp -a` to /tmp/opencode/ocx-global-backup/).

### 3. Pre-create ScanNet config
Create `/home/rolf/GIT/ScanNet/.opencode/worktree.jsonc`:
```jsonc
{
  "$schema": "https://registry.kdco.dev/schemas/worktree.json",
  "sync": { "copyFiles": [], "symlinkDirs": [], "exclude": [] },
  "hooks": { "postCreate": [], "preDelete": [] }
}
```
- Do NOT commit. Do NOT modify any other file in /home/rolf/GIT/ScanNet.

### 4. Smoke test in /home/rolf/GIT/test2 (NOT ScanNet)
- Start a fresh tmux session: `tmux new -s ocxtest -d`
- Inside it: `cd /home/rolf/GIT/test2 && opencode` (plugin loads; zod/jsonc-parser auto-installed by opencode into global node_modules on first launch — wait for it, no manual bun needed)
- Check load success: `~/.local/share/opencode/log` for worktree plugin errors; `opencode tool list` if the command exists (else check logs).
- Prompt the model: "Call worktree_create with branch ocx-smoke-test". Verify:
  - new tmux window appeared in session ocxtest (window name ocx-smoke-test)
  - dir `~/.local/share/opencode/worktree/<projectId>/ocx-smoke-test` exists (git worktree)
  - the forked opencode session started in the new window
- In the new window prompt: "Call worktree_delete with reason smoke test cleanup". Verify: worktree removed, branch ocx-smoke-test still exists in test2 repo (`git branch`).
- If worktree_create fails on terminal spawn (not inside tmux / no desktop terminal): rerun opencode INSIDE the tmux session so TMUX env is set — tmux window spawn is the expected path.
- If fork/API fails (session.idle or /session/{id}/fork route missing): report error text; fallback = native git worktrees + document; do NOT patch plugin code.

### 5. Report
- `git -C /home/rolf/GIT/ScanNet status --short` must show ONLY `?? .opencode/` (untracked worktree.jsonc)
- Summarize: install paths, generated files, test results, any error messages verbatim.
- Update spellbook/PROJECT_STATUS.md with the OCX/worktree setup.
- Clean up test: `tmux kill-session -t ocxtest`; delete test2 smoke branch if left.

## Constraints
- No sudo. No bun/node installs. No commits. No edits to ScanNet files except step 3.
- Keep opencode-with-claude global plugin working (verify `opencode --version` still fine and logs show both plugins loading).
