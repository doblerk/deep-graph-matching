import unittest
import numpy as np
from gnnged.heuristics.hull_utils import ConvexHull2D, ConvexHull3D

class TestHullFeatures(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        print('TestHullFeatures.__init__')
        super(TestHullFeatures, self).__init__(*args, **kwargs)

        # Simple square in 2D
        self.points_2d = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0.5, 0.5]  # inner point (should be ignored by hull)
        ])

        # Simple cube in 3D
        self.points_3d = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0.5, 0.5, 0.5]  # inner point
        ])

    def test_convex_hull_2d(self):
        hull2d = ConvexHull2D(self.points_2d)

        # Ensure expected properties
        centroid = hull2d.centroid
        self.assertEqual(len(centroid), 2)
        self.assertTrue(np.all(np.isfinite(centroid)))

        hull_size = hull2d.hull_size
        self.assertEqual(hull_size, 4)

        diameter = hull2d.diameter
        self.assertEqual(diameter, np.sqrt(2))
        self.assertTrue(diameter > 0)

        edge_stats = hull2d.calc_edge_statistics()
        self.assertEqual(len(edge_stats), 4)
        self.assertEqual(edge_stats[-1], 1.0)
        self.assertTrue(np.all(np.isfinite(edge_stats)))

        angle_stats = hull2d.angle_statistics
        self.assertEqual(len(angle_stats), 4)
        self.assertTrue(np.all(np.isfinite(angle_stats)))

        compactness = hull2d.compactness
        self.assertTrue(0 <= compactness <= 1)

        # Ensure cached property is computed only once
        val1 = hull2d.circularity
        val2 = hull2d.circularity
        self.assertEqual(val1, val2)

    def test_convex_hull_3d(self):
        hull3d = ConvexHull3D(self.points_3d)

        # Ensure expected properties
        centroid = hull3d.centroid
        self.assertEqual(len(centroid), 3)
        self.assertTrue(np.all(np.isfinite(centroid)))

        hull_size = hull3d.hull_size
        self.assertEqual(hull_size, 8)

        diameter = hull3d.diameter
        self.assertEqual(diameter, np.sqrt(3))
        self.assertTrue(diameter > 0)

        axis_lengths = hull3d.axis_lengths
        self.assertEqual(len(axis_lengths), 3)
        self.assertTrue(np.all(np.isfinite(axis_lengths)))

        edge_stats = hull3d.calc_edge_statistics()
        self.assertEqual(len(edge_stats), 4)
        self.assertAlmostEqual(edge_stats[-1], 1.0 / np.sqrt(2), places=6)
        self.assertTrue(np.all(np.isfinite(edge_stats)))

        compactness = hull3d.compactness
        self.assertTrue(compactness >= 0)

        sphericity = hull3d.sphericity
        self.assertTrue(0 <= sphericity <= 1)

        # Ensure cached property is computed only once
        val1 = hull3d.sphericity
        val2 = hull3d.sphericity
        self.assertEqual(val1, val2)
    
    def test_feature_vector_shapes(self):
        feats_2d = ConvexHull2D(self.points_2d).compute_all()
        feats_3d = ConvexHull3D(self.points_3d).compute_all()

        self.assertIsInstance(feats_2d, list)
        self.assertIsInstance(feats_3d, list)
        self.assertGreater(len(feats_2d), 10)
        self.assertGreater(len(feats_3d), 10)
        self.assertTrue(np.all(np.isfinite(feats_2d)))
        self.assertTrue(np.all(np.isfinite(feats_3d)))

    def test_invalid_input_handling(self):
        # Too few points for convex hull
        too_few_2d = np.array([[0, 0], [1, 1]])
        with self.assertRaises(Exception):
            ConvexHull2D(too_few_2d).compute_all()

        too_few_3d = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        with self.assertRaises(Exception):
            ConvexHull3D(too_few_3d).compute_all()


if __name__ == "__main__":
    unittest.main()