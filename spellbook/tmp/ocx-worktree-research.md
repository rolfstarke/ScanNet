# OCX worktree plugin — research report (READ-ONLY)

Date: 2026-08-09 · Researcher: main session · Source: kdcokenny/ocx @ main (cloned to /tmp/opencode/ocx), opencode v1.18.15 SDK/plugin npm packages, opencode release notes

## Verdict
Installing kdcokenny/opencode-worktree via OCX is feasible and compatible with OpenCode 1.18.15 (latest, 2026-08-07). Plugin API shape used by OCX matches the new 1.18.x `Plugin = (input) => Promise<Hooks>` API. No bun needed. One worktree per feature works; persistent reuse is supported but resume UX is manual.

## Verified facts

### 1. Install commands (current, verified from source)
- OCX binary: `curl -fsSL https://ocx.kdco.dev/install.sh | sh` → standalone compiled binary (no bun/node needed; npm variant requires bun). Installs to `/usr/local/bin` if writable else `~/.local/bin` (user has no sudo → `~/.local/bin`, already on PATH). SHA-256 verified download.
- Component: `ocx add kdco/worktree --from https://registry.kdco.dev` (ephemeral registry flag; README command). `--global` flag exists (flattened mode → `~/.config/opencode/`). `ocx init` optional prerequisite.
- Manual alternative (NOT recommended): copy `src/` to `.opencode/plugin/` + jsonc-parser + zod; loses updates + dependency management.

### 2. Exact generated files (local install in repo root)
- `.opencode/plugin/worktree.ts` (+ `worktree/state.ts`, `worktree/terminal.ts`, `worktree/launch-context.ts`)
- `.opencode/plugin/kdco-primitives/` (10 files: get-project-id, with-timeout, log-warn, types, mutex, shell, temp, cmux, terminal-detect, index)
- `.opencode/package.json` + `bun.lock` (deps: `jsonc-parser@3.3.1`, `zod@4.3.5`) + `.opencode/.gitignore` (keeps pkg files tracked)
- `.opencode/node_modules/` — installed by **opencode itself on first launch** (bundled bun; no bun on PATH required; loader.ts retries file plugins after dep install)
- `.opencode/worktree.jsonc` — auto-created on first tool call (not by install)
- OCX receipt file in `.opencode/`
- Runtime artifacts: `.git/opencode` (project-id cache, first root commit SHA), `~/.local/share/opencode/plugins/worktree/<projectId>.sqlite` (state DB), worktrees at `~/.local/share/opencode/worktree/<projectId>/<branch>/` (default; `worktreePath` config can override, supports `~`)

### 3. Config schema — `.opencode/worktree.jsonc` (zod-parsed, additionalProperties:false)
```jsonc
{
  "$schema": "https://registry.kdco.dev/schemas/worktree.json",
  "worktreePath": "~/worktrees",          // optional; default ~/.local/share/opencode/worktree
  "sync": { "copyFiles": [], "symlinkDirs": [], "exclude": [] },  // exclude: reserved/unused
  "hooks": { "postCreate": [], "preDelete": [] }                  // bash -c, run in worktree dir, non-fatal on failure
}
```
Note: symlink targets use absolute paths; copy is per-file.

### 4. Behavior (from plugin source)
- `worktree_create(branch, baseBranch?)`: validates branch (git ref rules, no shell metachars) → `git worktree add` (re-checks-out existing branch if present, else `-b` from baseBranch/HEAD) → sync files → postCreate hooks → forks current session via `client.session.fork` → spawns terminal with `opencode --session <forkedID>`.
- `worktree_delete(reason)`: marks pending-delete in shared SQLite; on next `session.idle` event: preDelete hooks → `git add -A` + commit `chore(worktree): session snapshot` (--allow-empty) → `git worktree remove --force`. Branch remains; merging is manual, normal git.
- Terminal spawn: priority tmux (if inside tmux) → cmux → platform terminals (Linux: kitty/wezterm/alacritty/ghostty/warp/foot/gnome/konsole/xfce/xterm fallback chain). **tmux new-window is only used when TMUX env var is set** (opencode running inside tmux). Outside tmux on a headless box → "No terminal emulator found" error (worktree already created, terminal fails).
- Persistent reuse: yes. Worktree+branch survive until delete. `worktree_create(sameBranch)` re-attaches if branch has no worktree. Resume of a live worktree = manual `cd` + `opencode` (or `opencode -s <forked-id>`; a NEW session id breaks `worktree_delete` → manual `git worktree remove`).
- Multiple worktrees in parallel: fully independent sessions. Serial heavy tests: no coordination built in; run tests from main repo or serialize manually.

### 5. OpenCode 1.18.15 compatibility (verified against published npm packages @opencode-ai/plugin@1.18.15, @opencode-ai/sdk@1.18.15, release v1.18.15)
- 1.18.x rewrote the plugin API: `Plugin = (input: {client, directory, project, worktree, ...}) => Promise<Hooks>` with `Hooks.tool`/`Hooks.event`. OCX plugin's `(ctx) => Promise<{tool, event}>` shape matches (runtime-compatible; toolCtx.sessionID, Event incl. `session.idle` all present).
- `client.session.fork/get/delete`, `client.app.log` (legacy v1 gen client) all ship in @opencode-ai/sdk@1.18.15 (lockstep with server; `RequestResult.data` matches plugin usage). Server-side v1 route support is the residual risk (SDK/server release in lockstep = strong evidence, plus opencode-with-claude legacy-ish npm plugin works on this install).
- `opencode --session <id>` CLI flag exists in 1.18.15 (`-s`).
- bun:sqlite + Bun.* globals: plugin runs inside opencode's bundled bun runtime (verified user's global `opencode-with-claude` plugin runs fine).
- Registry declares min `opencode: 1.1.0`, `ocx: 1.0.16` — satisfied.
- Compat gate: file plugins skip the npm version-compat check.

### 6. Conflicts / flags for this machine
- **spellbook policy**: plugin hardcodes `.opencode/` in repo root (config path `loadWorktreeConfig` + OCX install root). Cannot live under spellbook without patching. Options: accept `.opencode/` at repo root (local install) OR `ocx add --global` → `~/.config/opencode/plugin/` + merge deps into existing global package.json (repo stays clean; worktree.jsonc still auto-created per project root).
- `.git/opencode` cache file written inside .git (untracked, invisible to git status).
- Current shell is NOT inside tmux (TMUX unset) → terminal spawn needs opencode run inside tmux; user runs tmux (sessions: llama-server, ulbricht) so workable.
- OpenCode 1.18.15 has its OWN native worktree/workspaces feature (PluginInput.worktree, experimental_workspace API; evidenced by `~/.local/share/opencode/worktree/46d1c6a.../{neon-lagoon,witty-knight}` from /home/rolf/GIT/test2) — native alternative exists but lacks terminal-spawn/sync/hooks/auto-commit.
- /data/scannet + model repos/envs are outside repo → unaffected (not synced; hooks can reference absolute paths).

## Minimal install plan (local mode)
1. `curl -fsSL https://ocx.kdco.dev/install.sh | sh` → verify `ocx --version`
2. `cd /home/rolf/GIT/ScanNet && ocx add kdco/worktree --from https://registry.kdco.dev`
3. Pre-create `.opencode/worktree.jsonc` with desired config (worktreePath default; hooks empty initially)
4. Restart opencode in ScanNet (loads plugin, installs .opencode/node_modules)
5. Use from inside tmux: `worktree_create("feature/x")` → new tmux window with forked session; `worktree_delete("reason")` → snapshot commit + cleanup; manual `git merge feature/x` into master afterwards.

## Suggested ScanNet config
```jsonc
{
  "$schema": "https://registry.kdco.dev/schemas/worktree.json",
  "sync": { "copyFiles": [], "symlinkDirs": [], "exclude": [] },
  "hooks": { "postCreate": [], "preDelete": [] }
}
```
(no root node_modules to symlink; external dirs are outside repo; hooks added later per feature need)

## Cleanup/rollback
- Plugin: `ocx remove worktree` (or delete `.opencode/plugin/`, `.opencode/package.json`, `.opencode/node_modules`, `.opencode/worktree.jsonc`; nothing in git except .opencode if committed)
- OCX binary: delete `~/.local/bin/ocx`
- Worktrees: `git worktree list` / `git worktree remove --force <path>`; branches remain for merge
