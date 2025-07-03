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
        super().__init__(self, points, qhull_options='QJ')
        self.points = points
        self._dimension = points.shape[1]
        self._volume = self.volume  # Area in 2D, volume in 3D
        self._centroid = None
        self._diameter = None
        self._perimeter = None
        self._pca = None
        
    def calc_centroid(self):
        if self._centroid is None:
            self._centroid = [
                np.mean(self.points[self.vertices, i])
                for i in range(self._dimension)
            ]
        return self._centroid

    def calc_hull_size(self):
        return len(self.vertices)

    def calc_perimeter(self):
        # Note: in dimension > 2, this is a looped sum of vertex distances, not the true surface perimeter
        if self._perimeter is None:
            vertices = self.vertices.tolist() + [self.vertices[0]]
            self._perimeter = sum(
                euclidean(x, y) 
                for x, y in zip(self.points[vertices], self.points[vertices[1:]])
            )
        return self._perimeter

    def calc_volume(self):
        return self._volume
    
    def calc_diameter(self):
        if self._diameter is None:
            self._diameter = np.max(
                pdist(self.points[self.vertices], metric='euclidean')
            )
        return self._diameter
    
    def calc_compactness(self):
        radius = self.calc_diameter() / 2
        vol_n_sphere = (np.pi ** (self._dimension / 2) * radius ** self._dimension) / gamma((self._dimension / 2) + 1)
        return self._volume / vol_n_sphere if vol_n_sphere > 0 else 0
    
    def _compute_pca(self):
        if self._pca is None:
            self._pca = PCA(n_components=self.points.shape[1])
            self._pca.fit(self.points[self.vertices])
    
    def calc_major_minor_axes(self):
        self._compute_pca()
        axis_lengths = np.sqrt(self._pca.explained_variance_)
        axis_directions = self._pca.components_
        return {
            'axis_lengths': axis_lengths,         # e.g., [major, ..., minor]
            'axis_directions': axis_directions    # unit vectors for each axis
        }
    
    def calc_elongation(self):
        self._compute_pca()
        axis_lengths = np.sqrt(self._pca.explained_variance_)
        return axis_lengths[0] / axis_lengths[-1] if axis_lengths[-1] > 0 else float('inf')
    
    def calc_spatial_distribution(self):
        distances = np.linalg.norm(self.points - self._centroid, axis=1)
        return np.mean(distances)

    # def calc_isoperimetric_quotient(self):
    #     perimeter = self.calc_perimeter()
    #     area = self.calc_volume()
    #     r_circle = perimeter / (2 * np.pi)
    #     area_circle = np.pi * r_circle**2
    #     return area / area_circle

    def calc_minimum_bounding_sphere(self):
        radius = self.calc_diameter() / 2
        vol_n_sphere = (np.pi ** (self._dimension / 2) * radius ** self._dimension) / gamma((self._dimension / 2) + 1)
        compactness = self.calc_volume() / vol_n_sphere if vol_n_sphere > 0 else 0
        return {
            'sphere_radius': radius,
            'sphere_volume': vol_n_sphere,
            'compactness': compactness
        }
    
    def calc_edge_statistics(self):
        vertices = self.vertices.tolist() + [self.vertices[0]]
        edges = [euclidean(x, y) for x, y in zip(self.points[vertices], self.points[vertices[1:]])]
        return {
            'mean_edge_length': np.mean(edges),
            'shortest_edge': min(edges),
            'longest_edge': max(edges),
            'ratio_lengths': min(edges) / max(edges)
        }
    
    def calc_point_density(self):
        return self.calc_hull_size() / self.calc_volume()

    def calc_relative_shape(self):
        return self.calc_perimeter() / self.calc_volume()

    def calc_relative_spatial(self):
        return self.calc_diameter() / self.calc_volume()

    def compute_all(self, config=None):
        if config is None:
            config = {}

        feats = []
        feats.extend(self.calc_centroid())
        feats.append(self.calc_hull_size())
        feats.append(self.calc_perimeter())
        feats.append(self.calc_volume())
        feats.append(self.calc_diameter())
        feats.append(self.calc_compactness())

        mbs = self.calc_minimum_bounding_sphere()
        feats.extend([mbs['sphere_radius'], mbs['sphere_volume'], mbs['compactness']])

        edge_stats = self.calc_edge_statistics()
        feats.extend(edge_stats.values())

        feats.append(self.calc_point_density())
        feats.append(self.calc_relative_shape())
        feats.append(self.calc_relative_spatial())

        # try:
        #     feats.append(self.calc_elongation())
        #     if self._dimension in [2, 3]:
        #         feats.append(self.calc_eccentricity())
        # except:
        #     feats.append(None)

        return feats


class HullCloudStats:
    def __init__(self, points):
        self.points = points
    
    def mean(self):
        return np.mean(self.points, axis=0)
    
    def variance(self):
        return np.var(self.points, axis=0)
    
    def covariance_matrix(self):
        return np.cov(self.points, axis=0)
    
    def skewness(self):
        return skew(self.points, axis=0, bias=False)
    
    def kurtosis(self):
        return kurtosis(self.points, axis=0, bias=False)
    
    def summary(self):
        return {
            "mean": self.mean(),
            "variance": self.variance(),
            "covariance_matrix": self.covariance_matrix(),
            "skewness": self.skewness(),
            "kurtosis": self.kurtosis()
        }


class ConvexHull2D(ConvexHullBase):
    
    def __init__(self, points):
        super().__init__(points)
        self._area = self._volume
    
    def calc_circularity(self):
        perimeter = self.calc_perimeter()
        return (4 * np.pi * self._area) / (perimeter ** 2) if perimeter > 0 else 0
    
    def calc_eccentricity(self):
        axes = self.calc_major_minor_axes()['axis_lengths']
        a, b = axes[0], axes[1]
        return np.sqrt(1 - (b ** 2) / (a ** 2)) if a > 0 else 0
    
    def compute_all(self, config=None):
        feats = super().compute_all(config)
        feats.append(self.calc_circularity())
        feats.append(self.calc_eccentricity())
        return feats


class ConvexHull3D(ConvexHullBase):

    def __init__(self, points):
        super().__init__(points, qhull_options='QJ')
    
    def calc_sphericity(self):
        surface_area = self.area
        return (np.pi ** (1 / 3)) * ((6 * self._volume) ** (2 / 3)) / surface_area if surface_area > 0 else 0
    
    def calc_eccentricity(self):
        axes = self.calc_major_minor_axes()['axis_lengths']
        a, b = axes[0], axes[1]
        return np.sqrt(1 - (b ** 2) / (a ** 2)) if a > 0 else 0
    
    def compute_all(self, config=None):
        feats = super().compute_all(config)
        feats.append(self.calc_sphericity())
        feats.append(self.calc_eccentricity())
        return feats