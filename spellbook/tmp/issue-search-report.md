# Issue search report (2026-10-28)

Source: all 70 issues + 299 comments of rolfstarke/ScanNet (public GitHub API; `gh` not installed).

## 1. Processing-time limit: 3x scene length
1. Rule (comments, 2026-09-15, quoted identically in #41, #42, #50, #61): "Non-Metashape reconstructions must run in less than three times the scene length." Limits: 9004 = 120.487348 s -> 361.462044 s; 9009 = 157.111741 s -> 471.335223 s.
2. Restated: #51 comment 2026-09-16 ("new 3x limit"); #48 comment 2026-09-26 ("Reconstruction budget is 3x scene duration"); #50 comment 2026-10-01 ("`3 × SVO elapsed` cap", solver time only).
3. Only other value: #52 comment 2026-09-14 calls BundleFusion "realtime FAIL" (1x idea, before the rule existed). No 2x rule found.
4. Later limits use MCAP duration: 9004 120.400110 s -> 361.200330 s (#61, 2026-09-26); 9009 157.043787 s -> 471.131361 s (#51).
5. Metashape is exempt.

## 2. Original ZED .svo2 files
Found in #42 comment 2026-09-15 (issuecomment-5688002407), from zed_override `svo_path`:
1. scene9004 -> `HD1200_SN46307300_18-27-20_raum312_v2.svo2`
2. scene9009 -> `HD1080_SN46307300_19-33-28.svo2` (#21 also says scene9009 is 1080p)
3. Canonical symlinks: `/data/scannet/custom/raw/scene9004.svo2` and `scene9009.svo2` (spellbook/reconstruct/__init__.py:36, extract.py:261, PROJECT_STATUS.md:119).
4. `/data/scannet` cannot be reached from this host, so the symlink targets and run.json/svo_path.txt were not checked.
