import os
import sys
import unittest

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reconstruct.geometry_score import (  # noqa: E402
    METRIC, load_config, mean_bidirectional_distance_cm, pack_cells, surface_voxels, unpack_cells,
    voxel_centres)

CELL = 0.02


def _mesh(vertices, triangles):
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)))


class SurfaceVoxelTests(unittest.TestCase):
    def test_huge_floor_triangle_has_no_gaps(self):
        idx = surface_voxels(_mesh([[0.001, 0.001, 0.005], [12.001, 0.001, 0.005],
                                    [0.001, 8.001, 0.005]], [[0, 1, 2]]), np.zeros(3), CELL)
        self.assertTrue((idx[:, 2] == 0).all())
        # Every cell whose footprint lies inside the triangle, against the hypotenuse.
        ii, jj = np.meshgrid(np.arange(600), np.arange(400), indexing="ij")
        inside = ((ii + 1) * CELL) / 12.0 + ((jj + 1) * CELL) / 8.0 <= 1.0
        have = set(pack_cells(idx).tolist())
        want = np.stack([ii[inside], jj[inside], np.zeros(inside.sum(), int)], 1)
        self.assertTrue(all(k in have for k in pack_cells(want).tolist()))
        self.assertAlmostEqual(len(idx) * CELL ** 2 / 48.0, 1.0, delta=0.03)

    def test_tilted_large_triangle_is_connected_and_area_consistent(self):
        idx = surface_voxels(_mesh([[0, 0, 0], [6, 0, 6], [0, 6, 0]], [[0, 1, 2]]),
                             np.zeros(3) - 0.5 * CELL, CELL)
        area = 0.5 * 6 * 6 * np.sqrt(2)
        self.assertGreater(len(idx) * CELL ** 2, 0.7 * area)         # 45 deg plane: cos(45) of area
        self.assertLess(len(idx) * CELL ** 2, 2.5 * area)
        # Each cell column along z is hit: the plan projection has no holes.
        plan = {(i, j) for i, j in idx[:, :2].tolist()}
        missing = [(i, j) for i in range(0, 290) for j in range(0, 290 - i) if (i, j) not in plan]
        self.assertEqual(missing, [])

    def test_sliver_is_memory_safe_and_continuous(self):
        idx = surface_voxels(_mesh([[0.005, 0.005, 0.005], [40.005, 0.005, 0.005],
                                    [0.005, 0.006, 0.005]], [[0, 1, 2]]), np.zeros(3), CELL)
        self.assertEqual(sorted(idx[:, 0].tolist()), list(range(2001)))

    def test_small_triangle_keeps_its_one_cell(self):
        idx = surface_voxels(_mesh([[0.002, 0.002, 0.002], [0.004, 0.002, 0.002],
                                    [0.002, 0.004, 0.002]], [[0, 1, 2]]), np.zeros(3), CELL)
        self.assertEqual(idx.tolist(), [[0, 0, 0]])

    def test_cells_sorted_with_negative_indices(self):
        idx = surface_voxels(_mesh([[-0.1, -0.1, 0.1], [0.1, -0.1, 0.1], [-0.1, 0.1, 0.1]],
                                   [[0, 1, 2]]), np.zeros(3), CELL)
        self.assertEqual(idx.tolist(), sorted(idx.tolist()))
        np.testing.assert_array_equal(unpack_cells(pack_cells(idx)), idx)
        self.assertLess(idx.min(), 0)

    def test_cap_raises(self):
        with self.assertRaises(RuntimeError):
            surface_voxels(_mesh([[0, 0, 0.01], [1, 0, 0.01], [0, 1, 0.01]], [[0, 1, 2]]),
                           np.zeros(3), CELL, max_voxels=10)


class ScoreTests(unittest.TestCase):
    def test_same_mesh_scores_zero(self):
        mesh = _mesh([[0, 0, 0.01], [3, 0, 0.01], [0, 3, 0.01]], [[0, 1, 2]])
        c = voxel_centres(surface_voxels(mesh, np.zeros(3), CELL), np.zeros(3), CELL)
        self.assertEqual(mean_bidirectional_distance_cm(c, c), (0.0, 0.0, 0.0))

    def test_distances_are_plain_in_both_directions(self):
        ref = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        rec = np.array([[0.0, 0, 0], [0.0, 0, 200.0]])       # 200 m outlier stays in the mean
        score, acc, comp = mean_bidirectional_distance_cm(ref, rec)
        self.assertAlmostEqual(acc, 10000.0)
        self.assertAlmostEqual(comp, 50.0)
        self.assertAlmostEqual(score, 5025.0)


class ConfigTests(unittest.TestCase):
    def test_reference_matches_code(self):
        cfg = load_config()
        self.assertEqual(cfg["metric"], METRIC)
        self.assertEqual(cfg["voxel_mm"], 20)
        self.assertEqual(cfg["plot_distance_max_cm"], 100)


if __name__ == "__main__":
    unittest.main()
