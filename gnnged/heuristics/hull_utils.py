import numpy as np
from functools import cached_property
from scipy.special import gamma
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean, pdist


# class ConvexHullBase(ConvexHull):
#     def __init__(self, points):
#         super().__init__(points, qhull_options='QJ')
#         self._input_points = points
#         self._dimension = points.shape[1]

#         # Caches
#         self._volume = None
#         self._centroid = None
#         self._diameter = None
#         self._perimeter = None
#         self._pca = None
#         self._vol_n_sphere = None
#         self._axis_lengths = None
#         self._axis_directions = None
    
#     @property
#     def vertices_points(self):
#         return self.points[self.vertices]

#     def calc_centroid(self):
#         if self._centroid is None:
#             self._centroid = np.mean(self.vertices_points, axis=0)
#         return self._centroid

#     def calc_hull_size(self):
#         return len(self.vertices)

#     def calc_perimeter(self):

#         if self._perimeter is None:
#             if self._dimension == 2:
#                 self._perimeter = self.area
#                 return self._perimeter
#             else:
#                 # Note: in dimension > 2, this is a looped sum of vertex distances, not the true surface perimeter
#                 vertices = self.vertices.tolist() + [self.vertices[0]]
#                 self._perimeter = sum(
#                     euclidean(self.points[i], self.points[j]) 
#                     for i, j in zip(vertices, vertices[1:])
#                 )
#                 return self._perimeter
#         return self._perimeter

#     def calc_volume(self):
#         if self._volume is None:
#             self._volume = self.volume
#         return self._volume
    
#     def calc_diameter(self):
#         if self._diameter is None:
#             self._diameter = np.max(pdist(self.vertices_points, metric='euclidean'))
#         return self._diameter
    
#     def _calc_vol_n_sphere(self):
#         radius = self.calc_diameter() / 2
#         if self._vol_n_sphere is None:
#             self._vol_n_sphere = (np.pi ** (self._dimension / 2) * radius ** self._dimension) / gamma((self._dimension / 2) + 1)
#         return self._vol_n_sphere
    
#     def calc_compactness(self):
#         volume = self.calc_volume()
#         vol_n_sphere = self._calc_vol_n_sphere()
#         return volume / vol_n_sphere if vol_n_sphere > 0 else 0
    
#     def _compute_pca(self):
#         if self._pca is None:
#             self._pca = PCA(n_components=self._dimension)
#             self._pca.fit(self.vertices_points)
#             axis_lengths = np.sqrt(self._pca.explained_variance_)
#             axis_directions = self._pca.components_
#             sorted_indices = np.argsort(axis_lengths)[::-1]  # descending order
#             self._axis_lengths = axis_lengths[sorted_indices]
#             self._axis_directions = axis_directions[sorted_indices]
#             self._pca = -1
    
#     def calc_major_minor_axes(self):
#         self._compute_pca()
#         return [self._axis_lengths, self._axis_directions]
    
#     def calc_spatial_distribution(self):
#         centroid = self.calc_centroid()
#         distances = np.linalg.norm(self.points - centroid, axis=1)
#         return np.mean(distances)
    
#     # def calc_isoperimetric_quotient(self):
#     #     perimeter = self.calc_perimeter()
#     #     area = self.calc_volume()
#     #     r_circle = perimeter / (2 * np.pi)
#     #     area_circle = np.pi * r_circle**2
#     #     return area / area_circle

#     def calc_minimum_bounding_sphere(self):
#         radius = self.calc_diameter() / 2
#         vol_n_sphere = self._calc_vol_n_sphere()
#         compactness = self.calc_volume() / vol_n_sphere if vol_n_sphere > 0 else 0
#         return [radius, vol_n_sphere, compactness]
    
#     def calc_edge_statistics(self):
#         vertices = self.vertices.tolist() + [self.vertices[0]]
#         edges = [
#             euclidean(self.points[i], self.points[j]) 
#             for i, j in zip(vertices, vertices[1:])]
#         shortest = np.min(edges)
#         longest = np.max(edges)
#         return [np.mean(edges), shortest, longest, shortest/longest]
    
#     def calc_point_density(self):
#         return self.calc_hull_size() / self.calc_volume()

#     def calc_relative_shape(self):
#         return self.calc_perimeter() / self.calc_volume()

#     def calc_relative_spatial(self):
#         return self.calc_diameter() / self.calc_volume()

#     def compute_all(self):
#         self._compute_pca()

#         feats = list()
#         feats.extend(self.calc_centroid())
#         feats.append(self.calc_hull_size())
#         feats.append(self.calc_perimeter())
#         feats.append(self.calc_volume())
#         feats.append(self.calc_diameter())
#         feats.append(self.calc_compactness())
#         feats.append(self.calc_spatial_distribution())
#         feats.append(self.calc_point_density())
#         feats.append(self.calc_relative_shape())
#         feats.append(self.calc_relative_spatial())
#         feats.extend(self.calc_minimum_bounding_sphere())
#         feats.extend(self.calc_edge_statistics())

#         for i in range(len(feats)):
#             if isinstance(feats[i], np.ndarray):
#                 feats[i] = feats[i].tolist()
#             elif isinstance(feats[i], (np.floating, np.integer)):
#                 feats[i] = float(feats[i])

#         return feats


class ConvexHullBase(ConvexHull):
    def __init__(self, points):
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
            vertices = self.vertices.tolist() + [self.vertices[0]]
            return sum(
                euclidean(self.points[i], self.points[j]) 
                for i, j in zip(vertices, vertices[1:])
            )

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
    
    def calc_minimum_bounding_sphere(self):
        radius = self.diameter / 2
        return [radius, self.vol_n_sphere, self.compactness]
    
    def calc_edge_statistics(self):
        vertices = self.vertices.tolist() + [self.vertices[0]]
        edges = [
            euclidean(self.points[i], self.points[j]) 
            for i, j in zip(vertices, vertices[1:])]
        shortest = np.min(edges)
        longest = np.max(edges)
        return [np.mean(edges), shortest, longest, shortest / longest]
    
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



# class ConvexHull2D(ConvexHullBase):
    
#     def __init__(self, points):
#         super().__init__(points)
#         self._perimeter = self.area
#         self._area = self.volume
    
#     def calc_circularity(self):
#         return (4 * np.pi * self._area) / (self._perimeter ** 2) if self._perimeter > 0 else 0
    
#     def calc_eccentricity(self):
#         self._compute_pca()
#         a, b = self._axis_lengths[:2]
#         return np.sqrt(1 - (b ** 2) / (a ** 2)) if a > 0 else 0

#     def calc_elongation(self):
#         self._compute_pca()
#         a, b = self._axis_lengths[:2]
#         return a / b if b > 0 else 0
    
#     def calc_shape_compactness(self):
#         return self._area / (self._perimeter ** 2) if self._perimeter > 0 else 0

#     def compute_all(self):
#         feats = super().compute_all()
#         feats.append(self.calc_circularity())
#         feats.append(self.calc_eccentricity())
#         feats.append(self.calc_elongation())
#         feats.append(self.calc_shape_compactness())

#         for i in range(len(feats)):
#             if isinstance(feats[i], np.ndarray):
#                 feats[i] = feats[i].tolist()
#             elif isinstance(feats[i], (np.floating, np.integer)):
#                 feats[i] = float(feats[i])

#         return feats


# class ConvexHull3D(ConvexHullBase):

#     def __init__(self, points):
#         super().__init__(points)
#         self._surface_area = self.area
    
#     def calc_sphericity(self):
#         volume = self.calc_volume()
#         return (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / self._surface_area if self._surface_area > 0 else 0
    
#     def calc_asphericity(self):
#         self._compute_pca()
#         a, c = self._axis_lengths[0], self._axis_lengths[2]
#         return (a - c) / a if a > 0 else 0
    
#     def calc_elongation(self):
#         self._compute_pca()
#         a, c = self._axis_lengths[0], self._axis_lengths[2]
#         return a / c if c > 0 else 0
    
#     def calc_shape_compactness(self):
#         return self._surface_area / self.calc_volume()
    
#     def compute_all(self):
#         feats = super().compute_all()
#         feats.append(self.calc_sphericity())
#         feats.append(self.calc_asphericity())
#         feats.append(self.calc_elongation())
#         feats.append(self.calc_shape_compactness())

#         for i in range(len(feats)):
#             if isinstance(feats[i], np.ndarray):
#                 feats[i] = feats[i].tolist()
#             elif isinstance(feats[i], (np.floating, np.integer)):
#                 feats[i] = float(feats[i])

#         return feats

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

    def compute_all(self):
        feats = super().compute_all()
        feats.extend([
            self.circularity,
            self.eccentricity,
            self.elongation,
            self.shape_compactness,
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
        a, c = self._axis_lengths[0], self._axis_lengths[2]
        return (a - c) / a if a > 0 else 0
    
    @cached_property
    def elongation(self):
        a, c = self._axis_lengths[0], self._axis_lengths[2]
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