# Action plan: Chamfer-L1 score + two-cloud density overlay (scene9004)

**Status.** Ready to execute.  
**Goal.** Replace thresholded F1 ranking with symmetric nearest-neighbour Chamfer-L1 (cm, lower better) and replace the matched/missing scatter PNG with a neutral two-cloud projected-density overlay.

**Out of scope.** Alignment/ICP changes; CAD/visibility rebuild; engine debugging; multi-threshold F1; M3C2; signed distance heatmaps.

---

## Locked decisions

| # | Decision |
|---|---|
| 1 | **Single ranking score** = Chamfer-L1 in centimetres (symmetric mean of directional means). Lower is better. Not bounded to `[0,100]`. |
| 2 | **Diagnostics always written:** `accuracy_mean_cm` (recon→CAD), `completeness_mean_cm` (CAD→recon). |
| 3 | **Remove entirely:** F1, precision, recall, `distance_threshold_mm`, matched-mask scatter colours, TP/FP/FN legend. |
| 4 | **Sampling unchanged:** same fixed 10 mm grid, same observed-visibility CAD mask, same recon surface voxels after gauge (axisAlignment once + floor/centre + four yaw). |
| 5 | **Yaw selection objective:** minimise Chamfer-L1 on the coarse subsampled pair (not maximise F1). Tie-break: lower yaw. |
| 6 | **Retention eviction (scene9004):** among 10 scored slots, evict **highest** `geometry_score_cm` (worst). Incomplete/failed/absent still preferred over eviction. |
| 7 | **Overlay PNG:** two density clouds only — CAD (orange) + recon (blue), shared bins, count→alpha, transparent empty. Plan x/y and elevation x/z. No match colours. |
| 8 | **Filename:** keep `recon/cad_comparison.png` (avoid YAML/docs churn); content is an overlay, not F1 matching. |
| 9 | **YAML key:** `geometry_score_cm` (not `geometry_score`). Old F1 YAMLs are non-comparable. |
| 10 | **Metric id:** `observed_surface_voxel_chamfer_l1_v1`. |

### Metric definitions

Let \(C\) = centres of **visible CAD** surface voxels (after visibility ∩ CAD surface).  
Let \(R\) = centres of **recon** surface voxels after gauge alignment.

\[
\begin{aligned}
d(a,B) &= \min_{b\in B}\|a-b\|_2 \\
\text{accuracy\_mean\_m} &= \frac{1}{|R|}\sum_{r\in R} d(r,C) \\
\text{completeness\_mean\_m} &= \frac{1}{|C|}\sum_{c\in C} d(c,R) \\
\text{geometry\_score\_cm} &= 100\cdot\frac{
  \text{accuracy\_mean\_m}+\text{completeness\_mean\_m}}{2}
\end{aligned}
\]

Empty \(C\) or \(R\) → hard fail (`RuntimeError`), same spirit as empty F1 sets.

KD-tree: `scipy.spatial.cKDTree` (already imported). Distances in metres internally; write cm with 6 decimal places in YAML.

### Why not F1 anymore

- Threshold τ is arbitrary and discards error magnitude.  
- Chamfer-L1 is the continuous Atlas/NeuralRecon “accuracy+completeness” pair averaged (Chamfer-L1).  
- User ranking goal = single continuous geometric error, not “% within 5 cm”.

### Why not accuracy alone

Recon→CAD mean ignores missing walls (incomplete room can look “accurate”). Completeness is required; the score is their mean.

---

## Target YAML (`recon/geometry_score.yaml`)

```yaml
metric: observed_surface_voxel_chamfer_l1_v1
geometry_score_cm: 42.123456
accuracy_mean_cm: 35.0
completeness_mean_cm: 49.246912
scene: scene9004_41
reference_sha256: <unchanged>
visible_voxels_sha256: <unchanged>
voxel_mm: 10
alignment:
  axis_alignment_applied: true
  yaw_deg: 270
  transform: [...]
comparison: recon/cad_comparison.png
```

No `distance_threshold_mm`. No precision/recall/f1.

---

## Target overlay (`render_comparison` → density)

**Algorithm (keep simple):**

1. Inputs: full `ref_centres` (CAD), `recon_centres` (recon) — **no decimate** for hist (use all voxels; hist is O(N)).
2. For each view dims `(0,1)` plan and `(0,2)` elevation:
   - Bounds = axis-aligned box of **union** of both clouds, + 2% padding (or 0.1 m min pad). Do **not** use the huge configured ±30 m bounds for display.
   - Choose square-ish bins: target ~400 bins on the longer axis; `bin = max(0.02, extent_long/400)` metres (~2 cm, ≥ voxel).
   - `H_cad = np.histogram2d(cad[:,i], cad[:,j], bins=..., range=...)`
   - `H_rec` same edges.
   - Shared alpha scale: `s = sqrt(max(H_cad.max(), H_rec.max(), 1))`
   - RGBA layers:
     - CAD: RGB `(1.0, 0.45, 0.0)` (orange), α = `clip(sqrt(H_cad)/s, 0, 1) * 0.85`
     - Recon: RGB `(0.15, 0.35, 1.0)` (blue), α = `clip(sqrt(H_rec)/s, 0, 1) * 0.85`
   - Composite over white:  
     `out = white * (1-α) + rgb * α` per layer, CAD then recon (or average of both composites — pick CAD then recon for simple over-operator).
   - `ax.imshow(out, origin="lower", extent=[xmin,xmax,ymin,ymax], interpolation="nearest")`
   - Equal aspect; titles `plan x/y`, `elevation x/z`.
3. Legend: two patches “CAD”, “recon”.
4. Supertitle: `{scan_id}  chamfer_l1  score={geometry_score_cm:.2f} cm  (acc={..:.2f} comp={..:.2f})`
5. `figsize=(12,5)`, `dpi=140`, tight_layout.

No third class. No green/red match colours.

---

## File-by-file edits

### 1. `spellbook/reconstruct/geometry_reference.yaml`
- `metric: observed_surface_voxel_chamfer_l1_v1`
- **Delete** `distance_threshold_mm`
- Keep `voxel_mm: 10`, hashes, visibility, bounds, half-extents unchanged

### 2. `spellbook/reconstruct/geometry_score.py`
- Docstring: Chamfer-L1 scorer; CLI example unchanged path
- `METRIC = "observed_surface_voxel_chamfer_l1_v1"`
- **Remove** `match_f1`
- **Add** `directional_means(centres_a, centres_b)` → `(mean_a_to_b_m, mean_b_to_a_m)` via two KD-trees / `query`
- **Add** `chamfer_l1_cm(ref, recon)` → `(score_cm, acc_cm, comp_cm)`
- `gauge_normalize`: drop `tau`; yaw loop minimises chamfer on subsampled clouds (same stride ~80k); store best by lowest score, then lower yaw
- `score_sets` → return chamfer triple only (no match masks)
- `render_comparison(path, ref_centres, recon_centres, scene_id, score_cm, acc_cm, comp_cm)` — density overlay as above; drop match args and bounds_min/max display args (compute from data)
- `score_scan`:
  - no `distance_mm` CLI override (remove arg or ignore with deprecation print — prefer remove)
  - write new YAML keys
  - print `[geometry] {sid} chamfer={score:.2f} cm (acc={a:.2f} comp={c:.2f})`
- Validate: finite, `score_cm >= 0`; no upper bound

### 3. `spellbook/reconstruct/run.py`
- `_score_comparable(path)`:
  - require `metric == observed_surface_voxel_chamfer_l1_v1` (from config METRIC)
  - require keys `geometry_score_cm`, `accuracy_mean_cm`, `completeness_mean_cm`
  - **drop** `distance_threshold_mm` check
  - require finite `geometry_score_cm >= 0`
- `_read_score`: read `geometry_score_cm`
- Eviction sort: **highest** score first  
  `scored.sort(key=lambda c: (-c[4], c[5], c[1]))`  
  then take `[0]` as worst  
  log: `[evict] {sid} score={scv:.2f} cm`
- Import METRIC if needed for comparable check

### 4. `spellbook/PROJECT_STATUS.md` (structure/goals only)
- Goal line: Chamfer-L1 cm, not F1
- Structure: `geometry_score.py` description; `geometry_score.yaml` + overlay png
- Decision 9: eviction **highest** cm score
- Decision 14: rewrite for Chamfer-L1 + density overlay; no PASS/FAIL; lower better
- Commands: standalone score still `python -m spellbook.reconstruct.geometry_score --scan-id …`

### 5. Optional cleanup
- Delete or ignore `spellbook/tmp/cad-scan-metrics-literature.md` (agent research; archive under `archive/` if keeping)

---

## Data migration (existing scans)

| Scan | Action |
|---|---|
| `scene9004_40`, `scene9004_41` | Re-run scorer only (no reconstruct). Overwrites YAML + PNG. |
| `scene9009_40` | No score (non-benchmark scene). Untouched. |

```bash
conda activate 3disspellbook
cd /home/rolf/GIT/ScanNet
python -m spellbook.reconstruct.geometry_score --scan-id scene9004_40
python -m spellbook.reconstruct.geometry_score --scan-id scene9004_41
```

Expect **identical** scores for `_40` and `_41` (same mesh family / open3d baseline). Old F1 6.64 is discarded.

If YAML still has old metric after failed write, `_score_comparable` must return False so allocate treats slot as rescore/failed.

---

## Smoke tests

1. `python -m compileall -q spellbook/reconstruct/geometry_score.py spellbook/reconstruct/run.py`
2. Unit-ish (inline or one-liner): two identical centre clouds → score ≈ 0; shifted cloud → score ≈ shift magnitude in cm
3. Rescore `_41`; confirm:
   - `metric: observed_surface_voxel_chamfer_l1_v1`
   - `geometry_score_cm` present; no `geometry_score` / no `distance_threshold_mm`
   - PNG regenerated; visually two colours, density walls/floors readable, no green/red legend
4. `_40` and `_41` scores match within 1e-3 cm
5. Grep gate (live code, not archive/tmp):
   ```bash
   grep -rn "distance_threshold_mm\|match_f1\|observed_surface_voxel_f1\|geometry_score:" \
     spellbook --include='*.py' --include='*.yaml' --include='*.md' \
     | grep -v archive | grep -v tmp/ || true
   ```
   Only expected hits: this plan until archived; none in live modules after.

---

## Execution order

1. Edit `geometry_reference.yaml` (metric + drop τ).
2. Rewrite score path + overlay in `geometry_score.py`.
3. Update `run.py` comparable/read/evict.
4. Update `PROJECT_STATUS.md`.
5. `compileall` + synthetic zero/shift check.
6. Rescore `scene9004_40` and `scene9004_41`.
7. Open PNGs; confirm overlay readability.
8. Commit (when requested):  
   `feat: Chamfer-L1 geometry score + density CAD/recon overlay`
9. After green: archive this plan + literature note under `spellbook/archive/`; empty tmp leftovers per hygiene README.
10. FF engine worktrees to new master tip before debugging sessions.

---

## Constraints

- CPU-only scorer; never inside GPU lease (unchanged).
- No scale/ICP; yaw about CAD room centre only (unchanged).
- Visible mask remains the only CAD cells scored (unchanged).
- Do not touch ScanNet upstream.
- All code under `spellbook/`.
- Keep implementation simple: histogram2d + RGBA composite; no seaborn/plotly.

---

## Risk checklist

| Risk | Mitigation |
|---|---|
| Eviction still “lowest wins” after metric flip | Explicit sort by `-score`; smoke: unit test sort key mentally on fake list |
| Old YAML treated as valid | `_score_comparable` requires new metric string + `geometry_score_cm` |
| Overlay too dark/washed | shared sqrt norm; α cap 0.85; white background |
| Huge memory on hist | 2D hist only; centres already ~1e5–1e6; fine |
| Yaw flip if chamfer landscape flat | tie-break lower yaw; log yaw in YAML |
| Worktrees stale | FF after commit |

---

## Success criteria

- [ ] Live code has zero F1 / τ threshold scoring path
- [ ] YAML uses Chamfer-L1 keys; score is cm, lower better
- [ ] Eviction uses highest cm score
- [ ] PNG is two-cloud density overlay (CAD orange, recon blue)
- [ ] `_40` and `_41` rescored and agree
- [ ] PROJECT_STATUS decisions/commands match
- [ ] compileall clean

---

## Suggested improvement (follow-up, not this PR)

Report `accuracy_median_cm` / `completeness_median_cm` as extra YAML fields if means are dominated by a few drifted surfaces; keep ranking on mean Chamfer-L1 unless evidence says otherwise.
