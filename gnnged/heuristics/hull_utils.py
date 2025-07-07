import os
import h5py
import argparse

import numpy as np

from time import time
from scipy.special import gamma
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from torch_geometric.datasets import TUDataset

from scipy.stats import skew, kurtosis
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean, pdist, squareform


class ConvexHullBase(ConvexHull):
    def __init__(self, points):
        super().__init__(points, qhull_options='QJ')
        self._input_points = points
        self._dimension = points.shape[1]

        # Caches
        self._volume = None
        self._centroid = None
        self._diameter = None
        self._perimeter = None
        self._pca = None
        self._vol_n_sphere = None
        self._axis_lengths = None
        self._axis_directions = None
    
    @property
    def vertices_points(self):
        return self.points[self.vertices]

    def calc_centroid(self):
        if self._centroid is None:
            self._centroid = np.mean(self.vertices_points, axis=0)
        return self._centroid

    def calc_hull_size(self):
        return len(self.vertices)

    def calc_perimeter(self):
        # Note: in dimension > 2, this is a looped sum of vertex distances, not the true surface perimeter
        if self._perimeter is None:
            vertices = self.vertices.tolist() + [self.vertices[0]]
            self._perimeter = sum(
                euclidean(self.points[i], self.points[j]) 
                for i, j in zip(vertices, vertices[1:])
            )
        return self._perimeter

    def calc_volume(self):
        if self._volume is None:
            self._volume = self.volume
        return self._volume
    
    def calc_diameter(self):
        if self._diameter is None:
            self._diameter = np.max(pdist(self.vertices_points, metric='euclidean'))
        return self._diameter
    
    def _calc_vol_n_sphere(self):
        radius = self.calc_diameter() / 2
        if self._vol_n_sphere is None:
            self._vol_n_sphere = (np.pi ** (self._dimension / 2) * radius ** self._dimension) / gamma((self._dimension / 2) + 1)
        return self._vol_n_sphere
    
    def calc_compactness(self):
        volume = self.calc_volume()
        vol_n_sphere = self._calc_vol_n_sphere()
        return volume / vol_n_sphere if vol_n_sphere > 0 else 0
    
    def _compute_pca(self):
        if self._pca is None:
            self._pca = PCA(n_components=self._dimension)
            self._pca.fit(self.vertices_points)
            axis_lengths = np.sqrt(self._pca.explained_variance_)
            axis_directions = self._pca.components_
            sorted_indices = np.argsort(axis_lengths)[::-1]  # descending order
            axis_lengths = axis_lengths[sorted_indices]
            axis_directions = axis_directions[sorted_indices]
    
    def calc_major_minor_axes(self):
        self._compute_pca()
        return {
            'axis_lengths': self._axis_lengths,         # e.g., [major, ..., minor]
            'axis_directions': self._axis_directions    # unit vectors for each axis
        }
    
    def calc_spatial_distribution(self):
        centroid = self.calc_centroid()
        distances = np.linalg.norm(self.points - centroid, axis=1)
        return np.mean(distances)
    
    # def calc_isoperimetric_quotient(self):
    #     perimeter = self.calc_perimeter()
    #     area = self.calc_volume()
    #     r_circle = perimeter / (2 * np.pi)
    #     area_circle = np.pi * r_circle**2
    #     return area / area_circle

    def calc_minimum_bounding_sphere(self):
        radius = self.calc_diameter() / 2
        vol_n_sphere = self._calc_vol_n_sphere()
        compactness = self.calc_volume() / vol_n_sphere if vol_n_sphere > 0 else 0
        return {
            'sphere_radius': radius,
            'sphere_volume': vol_n_sphere,
            'compactness': compactness
        }
    
    def calc_edge_statistics(self):
        vertices = self.vertices.tolist() + [self.vertices[0]]
        edges = [
            euclidean(self.points[i], self.points[j]) 
            for i, j in zip(vertices, vertices[1:])]
        shortest = np.min(edges)
        longest = np.max(edges)
        return {
            'mean_edge_length': np.mean(edges),
            'shortest_edge': shortest,
            'longest_edge': longest,
            'edge_length_ratio': shortest / longest
        }
    
    def calc_point_density(self):
        return self.calc_hull_size() / self.calc_volume()

    def calc_relative_shape(self):
        return self.calc_perimeter() / self.calc_volume()

    def calc_relative_spatial(self):
        return self.calc_diameter() / self.calc_volume()

    def compute_all(self):
        self._compute_pca()
        
        feats = {
            'centroid': self.calc_centroid(),
            'hull_size': self.calc_hull_size(),
            'perimeter': self.calc_perimeter(),
            'volume': self.calc_volume(),
            'diameter': self.calc_diameter(),
            'compactness': self.calc_compactness(),
            'spatial_distribution': self.calc_spatial_distribution(),
            'point_density': self.calc_point_density(),
            'relative_shape': self.calc_relative_shape(),
            'relative_spatial': self.calc_relative_spatial(),
        }

        feats.update(self.calc_major_minor_axes())
        feats.update(self.calc_minimum_bounding_sphere())
        feats.update(self.calc_edge_statistics())

        return feats


class ConvexHull2D(ConvexHullBase):
    
    def __init__(self, points):
        super().__init__(points)
        self._perimeter = self.area
        self._area = self.volume
    
    def calc_circularity(self):
        return (4 * np.pi * self._area) / (self._perimeter ** 2) if self._perimeter > 0 else 0
    
    def calc_eccentricity(self):
        a, b = self._axis_lengths[:2]
        return np.sqrt(1 - (b ** 2) / (a ** 2)) if a > 0 else 0

    def calc_elongation(self):
        a, b = self._axis_lengths[:2]
        return a / b if b > 0 else 0
    
    def calc_shape_compactness(self):
        return self._area / (self._perimeter ** 2) if self._perimeter > 0 else 0

    def compute_all(self, config=None):
        feats = super().compute_all(config)
        feats.update({
            'circularity': self.calc_circularity(),
            'eccentricity': self.calc_eccentricity(),
            'elongation': self.calc_elongation(),
            'shape_compactness': self.calc_shape_compactness()
        })
        return feats


class ConvexHull3D(ConvexHullBase):

    def __init__(self, points):
        super().__init__(points)
        self._surface_area = self.area
    
    def calc_sphericity(self):
        volume = self.calc_volume()
        return (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / self._surface_area if self._surface_area > 0 else 0
    
    def calc_asphericity(self):
        a, c = self._axis_lengths[0], self._axis_lengths[2]
        return (a - c) / a if a > 0 else 0
    
    def calc_elongation(self):
        a, c = self._axis_lengths[0], self._axis_lengths[2]
        return a / c if c > 0 else 0
    
    def calc_shape_compactness(self):
        return self._surface_area / self.calc_volume()
    
    def compute_all(self, config=None):
        feats = super().compute_all(config)
        feats.update({
            'sphericity': self.calc_sphericity(),
            'asphericity': self.calc_asphericity(),
            'elongation': self.calc_elongation(),
            'shape_compactness': self.calc_shape_compactness()
        })
        return feats