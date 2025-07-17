import numpy as np
from functools import cached_property
from scipy.special import gamma
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean, pdist


class ConvexHullBase(ConvexHull):
    def __init__(self, points):
        points = np.asarray(points)
        num_points, num_dims = points.shape
        if num_points <= num_dims:
            raise ValueError(f"Too few points ({num_points}) to compute a hull in {num_dims}D (need at least {num_dims + 1})")
        super().__init__(points, qhull_options='QJ')
        self._input_points = points
        self._dimension = points.shape[1]
    
    @property
    def vertices_points(self):
        return self.points[self.vertices]

    @cached_property
    def centroid(self):
        return np.mean(self.vertices_points, axis=0) 

    @cached_property
    def hull_size(self):
        return len(self.vertices)

    @cached_property
    def perimeter(self):
        if self._dimension == 2:
            return self.area
        else:
            # note: in dimension > 2, this is a looped sum of vertex distances, not the true surface perimeter
            edges = self._extract_edges_from_simplices
            edge_lengths = [euclidean(self.points[i], self.points[j]) for i, j in edges]
            return sum(edge_lengths)

    @cached_property
    def volume(self):
        return self.volume
    
    @cached_property
    def diameter(self):
        return np.max(pdist(self.vertices_points, metric='euclidean'))
    
    @cached_property
    def vol_n_sphere(self):
        radius = self.diameter / 2
        return (np.pi ** (self._dimension / 2) * radius ** self._dimension) / gamma((self._dimension / 2) + 1)
    
    @cached_property
    def compactness(self):
        return self.volume / self.vol_n_sphere if self.vol_n_sphere > 0 else 0
    
    @cached_property
    def pca(self):
        model = PCA(n_components=self._dimension)
        model.fit(self.vertices_points)
        return model

    @cached_property
    def sorted_idx(self):
        axis_lengths = np.sqrt(self.pca.explained_variance_)
        return np.argsort(axis_lengths)[::-1] # descending order

    @cached_property
    def axis_lengths(self):
        axis_lengths = np.sqrt(self.pca.explained_variance_)
        return axis_lengths[self.sorted_idx]
    
    @cached_property
    def axis_directions(self):
        axis_directions = self.pca.components_
        return axis_directions[self.sorted_idx]

    def calc_major_minor_axes(self):
        return [self.axis_lengths, self.axis_directions]
    
    @cached_property
    def spatial_distribution(self):
        distances = np.linalg.norm(self.points - self.centroid, axis=1)
        return np.mean(distances)
    
    @cached_property
    def _extract_edges_from_simplices(self):
        edge_set = set()
        for simplex in self.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    edge = tuple(sorted((simplex[i].tolist(), simplex[j].tolist())))
                    edge_set.add(edge)
        return list(edge_set)

    def calc_minimum_bounding_sphere(self):
        radius = self.diameter / 2
        return [radius, self.vol_n_sphere, self.compactness]
    
    def calc_edge_statistics(self):
        if self._dimension == 2:
            vertices = self.vertices.tolist() + [self.vertices[0].tolist()]
            edges = [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]
        else:
            edges = self._extract_edges_from_simplices
        edge_lengths = [euclidean(self.points[i], self.points[j]) for i, j in edges]
        shortest = np.min(edge_lengths)
        longest = np.max(edge_lengths)
        return [np.mean(edge_lengths), shortest, longest, shortest / longest]
    
    @cached_property
    def point_density(self):
        return self.hull_size / self.volume

    @cached_property
    def relative_shape(self):
        return self.perimeter / self.volume

    @cached_property
    def relative_spatial(self):
        return self.diameter / self.volume

    def compute_all(self):
    
        feats = (
            list(self.centroid) +
            [self.hull_size, self.perimeter, self.volume, self.diameter, self.compactness,
            self.spatial_distribution, self.point_density, self.relative_shape, self.relative_spatial] +
            list(self.calc_minimum_bounding_sphere()) +
            list(self.calc_edge_statistics())
        )
        
        flat_feats = [
            x.item() if isinstance(x, (np.generic, np.ndarray)) else x
            for x in feats
        ]
        
        return flat_feats


class ConvexHull2D(ConvexHullBase):
    
    def __init__(self, points):
        super().__init__(points)
        self._perimeter = self.area # in 2D, "area" of ConvexHull is the perimeter
        self._area = self.volume    # in 3D, "volume" of ConvexHull is the area
    
    @cached_property
    def circularity(self):
        return (4 * np.pi * self._area) / (self._perimeter ** 2) if self._perimeter > 0 else 0
    
    @cached_property
    def eccentricity(self):
        a, b = self.axis_lengths[:2]
        return np.sqrt(1 - (b ** 2) / (a ** 2)) if a > 0 else 0

    @cached_property
    def elongation(self):
        a, b = self.axis_lengths[:2]
        return a / b if b > 0 else 0
    
    @cached_property
    def shape_compactness(self):
        return self._area / (self._perimeter ** 2) if self._perimeter > 0 else 0
    
    def _safe_normalize(self, v, eps=1e-8):
        norm = np.linalg.norm(v)
        return v / norm if norm > eps else np.zeros_like(v)
    
    @cached_property
    def angle_statistics(self):
        points = self.vertices_points
        n = len(points)

        if n < 3:
            return [0.0, 0.0, 0.0]
        
        angles = []
        for i in range(n):
            p1 = points[i - 1]          # previous vertex
            p2 = points[i]              # current vertex
            p3 = points[(i + 1) % n]    # next vertex
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            # normalize vectors
            v1 = self._safe_normalize(v1)
            v2 = self._safe_normalize(v2)

            # compute angle in radians
            dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(dot)
            normalized_angle = angle / np.pi
            
            angles.append(normalized_angle)

        angles = np.array(angles)

        return [
            np.mean(angles),
            np.std(angles),
            np.min(angles),
            np.max(angles)
        ]

    def compute_all(self):
        feats = super().compute_all()
        feats.extend([
            self.circularity,
            self.eccentricity,
            self.elongation,
            self.shape_compactness,
            *self.angle_statistics
        ])

        flat_feats = [
            x.item() if isinstance(x, (np.generic, np.ndarray)) else x
            for x in feats
        ]

        return flat_feats


class ConvexHull3D(ConvexHullBase):

    def __init__(self, points):
        super().__init__(points)
        self._surface_area = self.area
    
    @cached_property
    def sphericity(self):
        return (np.pi ** (1 / 3)) * ((6 * self.volume) ** (2 / 3)) / self._surface_area if self._surface_area > 0 else 0
    
    @cached_property
    def asphericity(self):
        a, c = self.axis_lengths[0], self.axis_lengths[2]
        return (a - c) / a if a > 0 else 0
    
    @cached_property
    def elongation(self):
        a, c = self.axis_lengths[0], self.axis_lengths[2]
        return a / c if c > 0 else 0
    
    @cached_property
    def shape_compactness(self):
        return self._surface_area / self.volume if self.volume > 0 else 0
    
    def compute_all(self):
        feats = super().compute_all()
        feats.extend([
            self.sphericity,
            self.asphericity,
            self.elongation,
            self.shape_compactness
        ])

        flat_feats = [
            x.item() if isinstance(x, (np.generic, np.ndarray)) else x
            for x in feats
        ]

        return flat_feats