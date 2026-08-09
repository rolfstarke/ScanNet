# Pilot: OpenCode native workspaces — go/no-go for ScanNet

Research by deepseek v4 flash (2026-08-09), opencode v1.18.15 (installed), source = anomalyco/opencode dev @ 38e10eb14.
Executed manually by user. Wipe this file after the decision. Plan NOT an edit to any ScanNet file.

## What workspaces actually are (verified in source)

- Feature: "workspace" = opencode-managed **git worktree**, project-scoped, rows in
  `~/.local/share/opencode/opencode.db` (`WorkspaceTable`, keyed by `project_id`); sessions get a `workspace_id` column.
- Built-in adapter "Worktree": `git worktree add --no-checkout --detach <dir> HEAD` where
  `<dir> = ~/.local/share/opencode/worktree/<projectID>/<slug>`, then `git reset --hard` + instance bootstrap
  (packages/opencode/src/control-plane/adapters/worktree.ts, src/worktree/index.ts).
  **Only mutation of the source repo: `.git/worktrees/<slug>` metadata. No branch (detached), no checkout changes.**
- Project identity is shared between main checkout and worktrees via `git-common-dir`
  (packages/core/src/project.ts `Project.resolve`) -> workspace list is per project, not per dir.
- TUI: `/warp` slash command opens the Warp dialog (create new / choose existing connected / "None" = local project);
  warping moves the session (sets `workspace_id`) and injects a system-reminder about the cwd change.
  **`/warp` is gated by env var `OPENCODE_EXPERIMENTAL_WORKSPACES=true`** (tui prompt/index.tsx `enabled: Flag.OPENCODE_EXPERIMENTAL_WORKSPACES`). EXPERIMENTAL.
- `/undo` (ctrl+x u) / `/redo` (ctrl+x r) use an internal snapshot git repo per (project, worktree dir):
  `~/.local/share/opencode/snapshot/<projectID>/<hash(worktreeDir)>` (src/snapshot/index.ts).
  => undo state is keyed per worktree -> expected isolation between workspaces AND vs. main checkout. Needs git repo (ours are).
- Reset: **HTTP-only**, no TUI button: `POST /experimental/worktree/reset` `{directory}` — refuses the primary worktree,
  resets to default branch, `git clean -ffdx`, `git submodule update --init --recursive --force`. EXPERIMENTAL.
- Remove: workspace list dialog (from Warp dialog, "View all workspaces") -> delete (press delete twice).
  Deletes the workspace's sessions too, runs `git worktree remove --force`, deletes branch, cleans dir, deletes db row.
- HTTP API under `/experimental/workspace/*`, `/experimental/worktree/*`; OpenAPI spec at `http://127.0.0.1:<port>/doc`.
  No auth by default (no OPENCODE_SERVER_PASSWORD set) -> curl works. TUI starts its own server: use `opencode --port 4199` for a fixed port.
- Session persistence: sqlite db; `opencode session list`, `opencode session delete <id>`, `opencode export <id>`,
  `opencode db "<sql>"` (documented CLI). Resume: `-c/--continue` or `-s/--session`, or `/sessions` in TUI.
- ScanNet specifics: `external/mLib` is a git submodule (niessner/mLib); `external/cutil` is tracked;
  `spellbook/eval/`, `.venv`, etc. are gitignored. Worktree creation does NOT init submodules -> `external/mLib` empty dir
  in a fresh workspace (reset endpoint initializes it).

## Phase 0 — Preconditions + baseline

1. `opencode --version` must be 1.18.15 (already confirmed).
2. Baseline snapshot (record exact outputs, keep for comparison):
   - `git -C /home/rolf/GIT/ScanNet worktree list --porcelain`  (expect exactly 1 entry)
   - `git -C /home/rolf/GIT/ScanNet status --porcelain`  (expect empty)
   - `ls ~/.local/share/opencode/worktree`  (expect: no such dir)
   - `ls ~/.local/share/opencode/snapshot`  (record dir names + count)
3. Build disposable repo (NOT under /home/rolf/GIT):
   ```
   mkdir -p /tmp/opencode/pilot-repo/src && cd /tmp/opencode/pilot-repo
   git init -b master && git config user.name pilot && git config user.email pilot@local
   printf 'v1 baseline\n' > notes.md
   printf 'print("hi")\n' > src/demo.py
   printf 'scratch/\nspellbook/eval/\n' > .gitignore
   mkdir -p /tmp/opencode/sub-src && cd /tmp/opencode/sub-src && git init && printf 'lib v1\n' > lib.c \
     && git add . && git -c user.name=x -c user.email=x@y commit -m init
   cd /tmp/opencode/pilot-repo && git submodule add /tmp/opencode/sub-src external/mLib
   git add -A && git commit -m "initial"
   ```
   Expected: clean `git status`; `git submodule status` shows external/mLib.
4. Terminal setup: tmux session "pilot" with 3 panes: (A) pilot-repo main checkout, (B) opencode TUI #1,
   (C) TUI #2. Export `OPENCODE_EXPERIMENTAL_WORKSPACES=true` in B and C only.

## Phase 1 — Disposable repo (tests T1–T8)

Start TUI 1 (pane B): `cd /tmp/opencode/pilot-repo && OPENCODE_EXPERIMENTAL_WORKSPACES=true opencode --port 4199`
Start TUI 2 (pane C): same but `--port 4198`.

### T1 Workspace creation
Steps: in TUI 1 type `/warp` -> dialog "Warp" -> "New workspace" (Worktree) -> confirm.
Expected:
- Worktree dir exists: `~/.local/share/opencode/worktree/<projectID>/<slug>` with checked-out files (notes.md, src/demo.py).
- `git -C /tmp/opencode/pilot-repo worktree list` shows 2 entries (primary + worktree).
- `/tmp/opencode/pilot-repo/.git/worktrees/<slug>/` exists; `git status` in primary still clean.
- A system-reminder about the cwd change appears in the session.
Abort: create fails with error dialog; worktree dir empty; primary `git status` dirty; a branch `opencode/*` was created (should be detached).

### T2 Correct cwd
Steps: in the warped session run `!pwd`, `!git rev-parse --show-toplevel`, `!git status --short`.
Expected: cwd == worktree dir; git toplevel == worktree dir; status clean. Then ask "append T2 to notes.md";
verify `cat /tmp/opencode/pilot-repo/notes.md` in pane A is UNCHANGED, worktree copy has T2.
Abort: cwd/toplevel mismatch; write lands in primary checkout.

### T3 Two concurrent workspaces
Steps: in TUI 1 `/warp` -> "New workspace" (second worktree B). In TUI 2 start a fresh session (`/new`), `/warp` ->
choose existing workspace B ("Choose workspace" -> recent connected list).
Expected: both workspace labels visible (top bar shows workspace name); `git worktree list` shows 3 entries;
both worktrees present; primary untouched.
Abort: workspace B not listed/selectable ("connected" status missing); TUI 2 lands in the wrong worktree.

### T4 Independent same-name file edits
Steps: TUI 1 (ws A): "append A1 to notes.md". TUI 2 (ws B): "append B1 to notes.md".
Expected: worktreeA/notes.md ends with A1; worktreeB/notes.md ends with B1; primary notes.md still "v1 baseline".
Abort: any cross-contamination between A and B or primary.

### T5 /undo isolation
Steps: in TUI 1: `/undo` (or ctrl+x u). Then `/redo` (ctrl+x r).
Expected: after /undo, worktreeA/notes.md no longer contains A1; worktreeB/notes.md STILL contains B1;
primary untouched. After /redo, A1 is back in worktreeA only.
Abort: undo affects worktreeB or primary; /redo unavailable/fails.

### T6 Session persistence
Steps: quit TUI 1 (`/exit`), restart `opencode --port 4199` in pilot-repo with flag; `/sessions` -> resume session 1.
In pane A: `opencode session list` (must run OUTSIDE the TUI's own server or from another terminal; fine in pane A).
Expected: session 1 listed; after resume: history intact, cwd still worktreeA, workspace label still ws A.
Abort: session missing; workspace attribution lost (session appears in local project instead); cwd wrong.

### T7 Remove
Steps: in TUI 2: `/warp` -> "View all workspaces" -> select workspace B -> delete -> confirm ("Press delete again").
Expected: `git worktree list` back to 2 entries; worktreeB dir gone; `~/.local/share/opencode/worktree/` cleaned;
`opencode db "select id,type,name from workspace"` shows only ws A;
session B (and its `session_diff` file) deleted.
Abort: worktree metadata left in `.git/worktrees/`; dir not removed; db row persists.

### T8 Reset (HTTP-only, experimental)
Steps: dirty ws A: in TUI 1 ask "create scratch/dirty.txt and append R1 to notes.md". Then from pane A:
```
curl -s -X POST "http://127.0.0.1:4199/experimental/worktree/reset?directory=$(ls -d ~/.local/share/opencode/worktree/*/ | tail -1)" \
  -H 'Content-Type: application/json' -d '{"directory":"<worktreeA dir>"}'
```
(Confirm exact query/payload params via `http://127.0.0.1:4199/doc` if 400.)
Expected: worktreeA notes.md back to baseline, scratch/dirty.txt gone, `git clean -ffdx` ran, `external/mLib` now
initialized (lib.c present — reset runs submodule init; NOTE this modifies the WORKTREE only).
Abort: reset fails (e.g. "Cannot reset the primary workspace"); primary repo modified; submodule metadata in primary .git changed.

### T9 External/mLib behavior (fold into T1/T8 checks)
Expected results:
- Fresh workspace: `external/cutil` present (tracked), `external/mLib` exists but EMPTY (submodule not initialized), `spellbook/eval/`-like ignored files absent, `scratch/` absent.
- After T8 reset: external/mLib initialized. (Note: submodule init writes .git/modules metadata of the PRIMARY repo —
  verify in Phase 1 that primary `.git/modules` is unchanged; record as observation for Phase 2.)
- Absolute paths like /data/scannet work from any workspace (bash `!ls /data/scannet | head -3`).

## Phase 2 — ScanNet-native (only if Phase 1 fully passes)

1. Re-verify Phase 0 baseline (worktree list, status, snapshot dirs).
2. Start: `cd /home/rolf/GIT/ScanNet && OPENCODE_EXPERIMENTAL_WORKSPACES=true opencode --port 4197`, `/warp` -> new workspace.
3. Expected (compare vs. Phase 0): `.git/worktrees/<slug>` added (the ONLY primary-repo change); `git status` still clean;
   AGENTS.md of ScanNet loaded (worktree checkout contains repo AGENTS.md); worktree cwd correct;
   `external/mLib` empty dir present; `/data/scannet` reachable.
4. Abort immediately (and clean up) if: primary `git status` dirty; `.git/worktrees` gains entries beyond the expected one;
   any edit attempt lands in /home/rolf/GIT/ScanNet; opencode db session rows for ScanNet project get deleted.
5. Cleanup (must, before any go decision): delete the workspace via warp dialog; verify:
   `git worktree list` == baseline; `git status` clean; `.git/worktrees/` empty; no worktree dirs under
   `~/.local/share/opencode/worktree/<ScanNet projectID>/`; workspace table empty for ScanNet project.

## Cleanup (always, both phases)

- Delete all pilot workspaces via warp dialog (or `curl -X DELETE http://127.0.0.1:4199/experimental/workspace/<id>?directory=/tmp/opencode/pilot-repo`).
- `cd /tmp/opencode/pilot-repo && git worktree prune`; remove `/tmp/opencode/pilot-repo`, `/tmp/opencode/sub-src`; kill TUIs.
- Verify ScanNet baseline identical to Phase 0 snapshot (worktree list, status, snapshot dir list, no new workspace rows in db).
- Wipe this file and any pilot notes from spellbook/tmp.

## Go / No-Go

- GO: T1–T6 fully pass; T7/T8 pass with documented caveats; Phase 2 passes with zero residue; cleanup leaves ScanNet byte-identical (except ~/.local/share/opencode db/log growth).
- NO-GO (record findings, wipe): any isolation failure (T4/T5), persistence failure (T6), primary-worktree contamination,
  cleanup residue, or Phase 2 leaving `.git/worktrees`/branch/db residue.
- Partial: only T7/T8 caveats (HTTP-only reset quirks) -> still GO for daily use; UI gaps (no TUI reset button) noted as improvement.

## Experimental callouts (do not rely on for production decisions)

- `/warp`, workspace dialogs, `/experimental/workspace/*`, `/experimental/worktree/*`: experimental, can change without notice.
- `OPENCODE_EXPERIMENTAL_WORKSPACES=true` env var: required for `/warp`; not documented on opencode.ai (found in CLI docs experimental env list + source).
- Reset endpoint: no TUI; verify params against /doc each run.
- `session move` (`/move`) is a separate project-copy feature (git_worktree strategy under `~/.local/share/opencode/worktree/`) — out of scope, noted to avoid confusion with workspaces.
