"""Exact 2D finite triangular tool with only its cutting vertex filleted.

The input vertex zero is the *virtual intersection* of rake and clearance.
The microscopic edge radius is unrelated to the insert's corner radius.
Contact uses the existing runtime's rigid_unilateral_impulse with the frame
returned by project(), including its spatially varying arc normal.
"""
from __future__ import annotations

import math

GEOMETRY_VERSION = 'finite-circular-edge-v1'


def tool_vertices(settings, time=0.0):
    """Original finite wedge; tool_edge_y_m denotes its virtual vertex."""
    edge = (settings['tool_initial_x_m'] + settings['tool_velocity_m_s'] * time,
            settings['tool_edge_y_m'])
    alpha, beta = map(math.radians, (settings['tool_rake_angle_deg'], settings['tool_clearance_angle_deg']))
    return (edge,
            (edge[0] - math.sin(alpha) * settings['tool_rake_length_m'],
             edge[1] + math.cos(alpha) * settings['tool_rake_length_m']),
            (edge[0] - math.cos(beta) * settings['tool_clearance_length_m'],
             edge[1] + math.sin(beta) * settings['tool_clearance_length_m']))


def tool_polygon(settings, time=0.0, arc_segments=64):
    """Display polygon only; physical contact always uses the exact circle."""
    vertices = tool_vertices(settings, time)
    if not settings.get('tool_edge_radius_m', 0):
        return vertices
    if type(arc_segments) is not int or arc_segments < 2:
        raise ValueError('Display arc needs at least two segments')
    tool = RoundedTool(vertices, settings['tool_edge_radius_m'])
    return (*vertices[1:], *(tool.boundary_point(i / arc_segments)[0]
                             for i in range(arc_segments, -1, -1)))


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _add(a, b, scale=1.0):
    return (a[0] + scale * b[0], a[1] + scale * b[1])


class RoundedTool:
    """Convex finite tool; one circular arc joins two tangent flat faces.

    faces retains the historical rake/clearance/posterior indices; edge_arc
    is index 3. Its stored frame is the arc midpoint frame for metadata only.
    query/project always return the correct local normal and tangent.
    """

    def __init__(self, vertices, edge_radius_m):
        if (len(vertices) != 3 or any(len(p) < 2 for p in vertices)
                or not all(math.isfinite(v) for p in vertices for v in p)
                or not math.isfinite(edge_radius_m) or edge_radius_m <= 0):
            raise ValueError("Three finite vertices and a positive microscopic radius are required")
        self.vertices = tuple(tuple(p[:2]) for p in vertices)
        self.radius = edge_radius_m
        edge, a, b = self.vertices
        ta, tb = _sub(a, edge), _sub(b, edge)
        la, lb = math.hypot(*ta), math.hypot(*tb)
        if min(la, lb) <= 0:
            raise ValueError("Degenerate tool")
        ta, tb = tuple(v / la for v in ta), tuple(v / lb for v in tb)
        cross = ta[0] * tb[1] - ta[1] * tb[0]
        if abs(cross) < 1e-12:
            raise ValueError("Degenerate or numerically singular tool angle")
        self.orientation = 1 if cross > 0 else -1
        na = (self.orientation * ta[1], -self.orientation * ta[0])
        nb = (-self.orientation * tb[1], self.orientation * tb[0])
        bisector = _add(ta, tb)
        norm = math.hypot(*bisector)
        bisector = tuple(v / norm for v in bisector)
        half_sine = math.sqrt(max(0.0, (1.0 - _dot(ta, tb)) / 2.0))
        self.center = _add(edge, bisector, edge_radius_m / half_sine)
        qa, qb = _add(self.center, na, edge_radius_m), _add(self.center, nb, edge_radius_m)
        trim = edge_radius_m * math.sqrt((1.0 + _dot(ta, tb)) / (1.0 - _dot(ta, tb)))
        if trim >= min(la, lb):
            raise ValueError("Edge radius consumes a cutting face or intersects the posterior")
        self.tangent_points = (qa, qb)
        self.trim_length = trim
        self.arc_start = math.atan2(na[1], na[0])
        # Rake to clearance normals travel opposite to triangle orientation.
        self.arc_sign = -self.orientation
        self.arc_sweep = math.pi - math.acos(max(-1.0, min(1.0, _dot(ta, tb))))
        mid = self.arc_start + self.arc_sign * self.arc_sweep / 2.0
        nm = (math.cos(mid), math.sin(mid))
        tp = _sub(b, a)
        lp = math.hypot(*tp)
        tp = tuple(v / lp for v in tp)
        np = (self.orientation * tp[1], -self.orientation * tp[0])
        self.faces = (
            ("rake", na, ta, la - trim, qa),
            ("clearance", nb, tb, lb - trim, qb),
            ("posterior", np, tp, lp, a),
            ("edge_arc", nm, (-nm[1], nm[0]), edge_radius_m * self.arc_sweep, qa),
        )
        self._segments = ((qa, a), (qb, b), (a, b))
        self.bounds = (min(p[0] for p in self.vertices), max(p[0] for p in self.vertices),
                       min(p[1] for p in self.vertices), max(p[1] for p in self.vertices))

    def _arc_parameter(self, normal):
        angle = math.atan2(normal[1], normal[0])
        return (self.arc_sign * (angle - self.arc_start)) % (2.0 * math.pi)

    def boundary_point(self, angle_fraction):
        """Arc point and normal for fraction in [0, 1], rake to clearance."""
        if not math.isfinite(angle_fraction) or not 0 <= angle_fraction <= 1:
            raise ValueError("Arc fraction must be in [0, 1]")
        angle = self.arc_start + self.arc_sign * self.arc_sweep * angle_fraction
        normal = (math.cos(angle), math.sin(angle))
        return _add(self.center, normal, self.radius), normal

    @property
    def lowest_point_offset(self):
        """Physical lowest y minus the virtual vertex y; preserves feed depth."""
        points = [*self.vertices[1:], *self.tangent_points]
        if self._arc_parameter((0.0, -1.0)) <= self.arc_sweep:
            points.append((self.center[0], self.center[1] - self.radius))
        return min(p[1] for p in points) - self.vertices[0][1]

    def query(self, point):
        """Return signed Euclidean distance (negative inside) and closest frame.

        At the non-smooth posterior vertices, an exterior query uses the
        outward normal in the normal cone. Ties inside choose the first face.
        The rounded cutting face transitions have a unique continuous normal.
        """
        if len(point) not in (2, 3) or not all(math.isfinite(v) for v in point):
            raise ValueError("Expected finite 2D or 3D point")
        p = tuple(point[:2])
        candidates = []
        for i, (first, last) in enumerate(self._segments):
            name, normal, tangent, length, origin = self.faces[i]
            along = max(0.0, min(length, _dot(_sub(p, first), tangent)))
            q = _add(first, tangent, along)
            candidates.append((math.hypot(*_sub(p, q)), i, q, normal, tangent, along))
        delta = _sub(p, self.center)
        length = math.hypot(*delta)
        normal = tuple(v / length for v in delta) if length else self.faces[3][1]
        parameter = self._arc_parameter(normal)
        if parameter <= self.arc_sweep:
            q = _add(self.center, normal, self.radius)
            candidates.append((abs(length - self.radius), 3, q, normal,
                               (-normal[1], normal[0]), parameter * self.radius))
        # Endpoints are also segment candidates. Exact convex support test:
        # all original planes plus every tangent plane on the circular arc.
        planes_inside = all(_dot(_sub(p, face[4]), face[1]) <= 0 for face in self.faces[:3])
        arc_support = length if parameter <= self.arc_sweep else max(
            _dot(delta, self.faces[0][1]), _dot(delta, self.faces[1][1]))
        inside = planes_inside and arc_support <= self.radius
        distance, index, q, normal, tangent, along = min(candidates, key=lambda row: row[0])
        if not inside and index < 3 and distance > 0 and along in (0.0, self.faces[index][3]):
            # Only a non-smooth exterior endpoint needs a normal-cone frame.
            # On arcs use the analytic radial frame: dividing p-q by
            # abs(|p-C|-r) loses orthonormality near a translated boundary.
            direction = _sub(p, q)
            normal = tuple(v / math.hypot(*direction) for v in direction)
        # Use one handedness on every feature. Metadata face tangents locate
        # along_m but need not follow the same boundary traversal (clearance
        # is stored away from the tip). A consistent contact tangent preserves
        # signed tangential-velocity continuity at both smooth arc transitions.
        tangent = (-normal[1], normal[0])
        return {"signed_distance_m": -distance if inside else distance,
                "inside": inside, "closest_point": q, "normal": normal,
                "tangent": tangent, "face_index": index, "face_name": self.faces[index][0],
                "along_m": along, "surface_length_m": self.faces[index][3]}

    def project(self, point, clearance=0.0, tolerance=1e-12):
        """Position repair only; no penetration/dt velocity term is introduced."""
        if len(point) != 3 or not all(math.isfinite(v) for v in (*point, clearance, tolerance)) or min(clearance, tolerance) < 0:
            raise ValueError("Expected finite 3D point and nonnegative distances")
        query = self.query(point)
        gap = query["signed_distance_m"]
        if gap > tolerance:
            return None
        scale = max(abs(v) for p in (*self.vertices, point) for v in p)
        target = max(clearance, 4 * tolerance, 64 * math.ulp(scale))
        distance = target - gap
        normal = query["normal"]
        correction = [distance * normal[0], distance * normal[1], 0.0]
        post = [point[i] + correction[i] for i in range(3)]
        post_query = self.query(post)
        return {**query, "position_post": post, "position_correction": correction,
                "position_correction_m": distance, "penetration_before_m": max(-gap, 0.0),
                "penetration_after_m": max(-post_query["signed_distance_m"], 0.0),
                "inside_after": post_query["signed_distance_m"] <= tolerance}
