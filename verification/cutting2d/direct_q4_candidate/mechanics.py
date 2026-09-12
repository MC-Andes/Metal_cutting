"""Direct conforming Q4 mechanics with fixed physical consistent sparse mass.

Nodal velocity is the complete material Q4 field: no B2 restriction, APIC
inertia, dense energy basis, SVD, lumping or mass scaling. Constitutive state
and global transaction/geometry acceptance belong to the caller.
"""
from __future__ import annotations

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

VERSION = 'direct-q4-consistent-reference-mass-v1'
CORNER_SIGNS = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
GAUSS_POINTS = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]) / np.sqrt(3.)


def readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _two_sum(a, b):
    """Error-free sum of two normal finite floats, subject to no overflow."""
    high = a + b
    recovered_b = high - a
    low = (a - (high - recovered_b)) + (b - recovered_b)
    return high, low


def add_compensated(hi, lo, du):
    """Add a nodal increment to a normalized two-float displacement.

    Inputs are not mutated. The three input values are accumulated as an
    expansion and renormalized to two floats; the remaining rounding is in
    the low part, not in the global position. This does not promise an exact
    representation of every possible sum of three arbitrary floats.
    """
    hi = np.asarray(hi, float); lo = np.asarray(lo, float); du = np.asarray(du, float)
    if hi.shape != lo.shape or hi.shape != du.shape or not all(np.isfinite(a).all() for a in (hi, lo, du)):
        raise ValueError('Compensated addition requires equal finite array shapes')
    try:
        with np.errstate(over='raise', invalid='raise'):
            summed, error = _two_sum(hi, du)
            tail, tail_error = _two_sum(lo, error)
            new_hi, new_lo = _two_sum(summed, tail)
            new_hi, new_lo = _two_sum(new_hi, new_lo + tail_error)
    except FloatingPointError as error:
        raise ValueError('Compensated displacement sum is not representable') from error
    if not np.isfinite(new_hi).all() or not np.isfinite(new_lo).all():
        raise ValueError('Compensated displacement sum is not finite')
    return new_hi, new_lo


def shape_functions(points):
    """Q4 N,D at reference-square points, D[a,j]=d N_a / d xi_j."""
    points = np.asarray(points, float)
    factors = 1 + points[:, None, :] * CORNER_SIGNS[None]
    N = np.prod(factors, axis=2) / 4
    D = np.stack((CORNER_SIGNS[None, :, 0]*factors[:, :, 1],
                  CORNER_SIGNS[None, :, 1]*factors[:, :, 0]), axis=2) / 4
    return N, D


def det2(matrix):
    return matrix[..., 0, 0]*matrix[..., 1, 1] - matrix[..., 0, 1]*matrix[..., 1, 0]


def inverse2(matrix, determinant):
    result = np.empty_like(matrix)
    result[..., 0, 0] = matrix[..., 1, 1]
    result[..., 1, 1] = matrix[..., 0, 0]
    result[..., 0, 1] = -matrix[..., 0, 1]
    result[..., 1, 0] = -matrix[..., 1, 0]
    return result / determinant[..., None, None]


class DirectQ4:
    """Shared Q4 reference mesh and force/mass operators.

    Corners are CCW (-,-),(+,-),(+,+),(-,+). GP order within each cell is
    (-g,-g),(-g,+g),(+g,-g),(+g,+g), identical to FiniteJCQ4 and DG/SIPG.
    X/cells and all reference arrays are immutable. Global overlap/topology
    checks are separate from the local corner-Jacobian guard here.
    """

    def __init__(self, X, cells, rho, width=1., *, solve_tolerance=1e-11):
        started = time.perf_counter()
        X = np.asarray(X, float); cells = np.asarray(cells)
        if (X.ndim != 2 or X.shape[1] != 2 or len(X) < 4 or not np.isfinite(X).all() or
                cells.ndim != 2 or cells.shape[1] != 4 or len(cells) < 1 or cells.dtype.kind not in 'iu' or
                np.any(cells < 0) or np.any(cells >= len(X))):
            raise ValueError('Finite reference nodes and integer CCW Q4 connectivity required')
        if (np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0) or
                len(np.unique(cells)) != len(X)):
            raise ValueError('Q4 corners must be distinct and every reference node used')
        if (not np.isfinite(rho) or not np.isfinite(width) or rho <= 0 or width <= 0 or
                not np.isfinite(rho*width) or not np.isfinite(solve_tolerance) or not 0 < solve_tolerance < 1):
            raise ValueError('Positive finite rho, width and solve tolerance required')
        self.X = readonly(X); self.cells = readonly(cells, np.int64)
        self.rho = float(rho); self.width = float(width); self.solve_tolerance = float(solve_tolerance)
        self.nodes = len(X); self.elements = len(cells); self.point_count = 4*self.elements
        N, D = shape_functions(GAUSS_POINTS); _, corners_D = shape_functions(CORNER_SIGNS)
        self._D = readonly(D); self._corner_D = readonly(corners_D)
        reference = self.X[self.cells] - self.X[self.cells[:, :1]]
        corner_J = np.einsum('eai,gaj->egij', reference, corners_D)
        corners = det2(corner_J)
        if np.any(corners <= 0):
            raise ValueError('Reference Q4 inverted or singular at a corner')
        J = np.einsum('eai,gaj->egij', reference, D)
        determinant = det2(J)
        if np.any(determinant <= 0) or not np.isfinite(determinant).all():
            raise ValueError('Positive representable reference quadrature required')
        G = np.einsum('gaj,egjk->egak', D, inverse2(J, determinant))
        self.ids = readonly(np.repeat(self.cells, 4, axis=0))
        self.N = readonly(np.tile(N, (self.elements, 1)))
        self.G_ref = readonly(G.reshape(self.point_count, 4, 2))
        self.volume0 = readonly((self.width*determinant).reshape(-1))
        self.mass = readonly(self.rho*self.volume0)
        self.reference_points = readonly(np.einsum('qa,qai->qi', self.N, self.X[self.ids]))
        self.reference_corner_jacobians = readonly(corners)
        self._reference_corner_matrices = readonly(corner_J)
        if np.any(self.mass <= 0) or not np.isfinite(self.mass).all():
            raise ValueError('Positive finite quadrature mass required')
        local = np.einsum('eg,ga,gb->eab', (self.rho*self.width)*determinant, N, N)
        rows = np.repeat(self.cells, 4, axis=1).ravel()
        cols = np.tile(self.cells, (1, 4)).ravel()
        self.M = sparse.coo_matrix((local.ravel(), (rows, cols)), shape=(self.nodes, self.nodes)).tocsr()
        self.M.sum_duplicates(); self.M.sort_indices()
        # This diagonal is diagnostic/preconditioning data, never lumped inertia.
        self.diagonal = readonly(self.M.diagonal())
        if np.any(self.diagonal <= 0) or not np.isfinite(self.M.data).all():
            raise ValueError('Physical consistent mass has invalid diagonal or entries')
        symmetry = float(sparse.linalg.norm(self.M - self.M.T)) / float(sparse.linalg.norm(self.M))
        total_mass = float(self.M.sum()); quadrature_mass = float(self.mass.sum())
        if symmetry > 1e-13 or abs(total_mass/quadrature_mass - 1.) > 1e-13:
            raise ValueError('Physical mass symmetry or quadrature identity failed')
        self._factors = {}
        self.diagnostics = dict(schema=VERSION, nodes=self.nodes, elements=self.elements,
            gauss_points=self.point_count, mass_kg=total_mass, mass_nnz=self.M.nnz,
            mass_sparse_bytes=self.M.data.nbytes+self.M.indices.nbytes+self.M.indptr.nbytes,
            mass_symmetry_relative=symmetry, no_mass_lumping=True, no_mass_scaling=True,
            representation='complete shared material Q4 nodal velocity', factors=[], solves=0,
            maximum_solve_relative_residual=0., construction_seconds=time.perf_counter()-started)

    def _nodal(self, value, name):
        value = np.asarray(value, float)
        if value.shape != (self.nodes, 2) or not np.isfinite(value).all():
            raise ValueError(f'{name} requires finite (nodes,2)')
        return value

    def corner_jacobians(self, q):
        q = self._nodal(q, 'Current coordinates')
        local = q[self.cells] - q[self.cells[:, :1]]
        return det2(np.einsum('eai,gaj->egij', local, self._corner_D))

    def deformation(self, q):
        q = self._nodal(q, 'Current coordinates')
        corners = self.corner_jacobians(q)
        if not np.isfinite(corners).all() or np.any(corners <= 0):
            raise ValueError('Current Q4 inverted or singular at a corner')
        F = np.broadcast_to(np.eye(3), (self.point_count, 3, 3)).copy()
        local = q[self.ids] - q[self.ids[:, :1]]
        F[:, :2, :2] = np.einsum('qai,qaJ->qiJ', local, self.G_ref)
        if not np.isfinite(F).all() or np.any(det2(F[:, :2, :2]) <= 0):
            raise ValueError('Nonpositive Gauss-point deformation determinant')
        return F, corners

    def deformation_displacement(self, hi, lo):
        """Canonical F and corner determinants from material displacement.

        F = I + grad_X(hi+lo); reference corner matrices plus grad_xi u
        determine admissibility. The two displacement parts are evaluated
        separately, avoiding an early rounded global-coordinate sum. Trial,
        commit, and restart must pass the same candidate pair to this method.
        """
        hi = self._nodal(hi, 'High displacement')
        lo = self._nodal(lo, 'Low displacement')
        with np.errstate(over='ignore', invalid='ignore'):
            high_gp = hi[self.ids] - hi[self.ids[:, :1]]
            low_gp = lo[self.ids] - lo[self.ids[:, :1]]
            displacement_gradient = (np.einsum('qai,qaJ->qiJ', high_gp, self.G_ref)
                                     + np.einsum('qai,qaJ->qiJ', low_gp, self.G_ref))
            F = np.broadcast_to(np.eye(3), (self.point_count, 3, 3)).copy()
            F[:, :2, :2] += displacement_gradient
            high_cell = hi[self.cells] - hi[self.cells[:, :1]]
            low_cell = lo[self.cells] - lo[self.cells[:, :1]]
            parametric_gradient = (np.einsum('eai,gaj->egij', high_cell, self._corner_D)
                                   + np.einsum('eai,gaj->egij', low_cell, self._corner_D))
            matrices = self._reference_corner_matrices + parametric_gradient
            corners = det2(matrices)
        if not np.isfinite(F).all() or not np.isfinite(corners).all():
            raise ValueError('Compensated displacement geometry is not finite')
        if np.any(corners <= 0):
            raise ValueError('Displacement Q4 inverted or singular at a corner')
        if np.any(det2(F[:, :2, :2]) <= 0):
            raise ValueError('Nonpositive displacement Gauss-point determinant')
        # Forward-error envelopes are diagnostics only: no perturbation of F,
        # constraint relaxation, or enlargement of checkpoint tolerance.
        u = np.finfo(float).eps / 2; gamma = 32*u/(1-32*u)
        with np.errstate(over='ignore', invalid='ignore'):
            envelope = (abs(hi[self.ids]) + abs(hi[self.ids[:, :1]])
                        + abs(lo[self.ids]) + abs(lo[self.ids[:, :1]]))
            bound = gamma * (np.eye(2) + np.einsum('qai,qaJ->qiJ', envelope, abs(self.G_ref)))
            corner_envelope = (abs(hi[self.cells]) + abs(hi[self.cells[:, :1]])
                               + abs(lo[self.cells]) + abs(lo[self.cells[:, :1]]))
            corner_bound = gamma * (abs(self._reference_corner_matrices)
                            + np.einsum('eai,gaj->egij', corner_envelope, abs(self._corner_D)))
        if not np.isfinite(bound).all() or not np.isfinite(corner_bound).all():
            raise ValueError('Displacement roundoff envelope is not representable')
        self.diagnostics['displacement_representation'] = 'material_two_float_expansion_v1'
        self.diagnostics['maximum_displacement_F_roundoff_bound'] = max(
            self.diagnostics.get('maximum_displacement_F_roundoff_bound', 0.), float(np.max(bound)))
        self.diagnostics['maximum_displacement_corner_matrix_roundoff_bound_m'] = max(
            self.diagnostics.get('maximum_displacement_corner_matrix_roundoff_bound_m', 0.), float(np.max(corner_bound)))
        return F, corners

    def display_positions(self, hi, lo):
        """Rounded global-position view of the canonical displacement pair."""
        hi = self._nodal(hi, 'High displacement')
        lo = self._nodal(lo, 'Low displacement')
        summed, error = add_compensated(self.X, np.zeros_like(self.X), hi)
        summed, error = add_compensated(summed, error, lo)
        position = summed + error
        if not np.isfinite(position).all():
            raise ValueError('Global position view is not finite')
        u = np.finfo(float).eps / 2; gamma = 16*u/(1-16*u)
        bound = gamma*abs(self.X) + gamma*abs(hi) + gamma*abs(lo)
        if not np.isfinite(bound).all():
            raise ValueError('Global position roundoff envelope is not finite')
        self.diagnostics['maximum_display_position_roundoff_bound_m'] = max(
            self.diagnostics.get('maximum_display_position_roundoff_bound_m', 0.), float(np.max(bound)))
        return position

    def point_positions(self, q):
        q = self._nodal(q, 'Current coordinates')
        return np.einsum('qa,qai->qi', self.N, q[self.ids])

    def force_from_piola(self, P):
        P = np.asarray(P, float)
        if P.shape not in ((self.point_count, 2, 2), (self.point_count, 3, 3)) or not np.isfinite(P).all():
            raise ValueError('Finite first Piola at every fixed Q4 Gauss point required')
        local = -self.volume0[:, None, None] * np.einsum('qaJ,qiJ->qai', self.G_ref, P[:, :2, :2])
        force = np.zeros((self.nodes, 2))
        np.add.at(force, self.ids.ravel(), local.reshape(-1, 2))
        return force

    def force_from_cauchy(self, q, sigma):
        sigma = np.asarray(sigma, float)
        if sigma.shape not in ((self.point_count, 2, 2), (self.point_count, 3, 3)) or not np.isfinite(sigma).all():
            raise ValueError('Finite Cauchy stress at every fixed Q4 Gauss point required')
        F, _ = self.deformation(q); small = F[:, :2, :2]; J = det2(small)
        P = J[:, None, None] * (sigma[:, :2, :2] @ np.swapaxes(inverse2(small, J), 1, 2))
        return self.force_from_piola(P)

    def _factor(self, free):
        key = free.tobytes()
        if key not in self._factors:
            started = time.perf_counter(); ids = np.flatnonzero(free)
            reduced = self.M[ids][:, ids].tocsc()
            factor = splu(reduced) if len(ids) else None
            self._factors[key] = (ids, reduced, factor)
            self.diagnostics['factors'].append(dict(free_nodes=len(ids), nnz=reduced.nnz,
                seconds=time.perf_counter()-started))
        return self._factors[key]

    def solve(self, force, fixed=None):
        """Physical M_ff inverse per component, zeros on prescribed fixed DOFs.

        ``force`` is a nodal force or impulse. The caller supplies the time
        factor for a force kick. Supports are stationary in this version.
        """
        force = self._nodal(force, 'Force/impulse')
        if fixed is None:
            fixed = np.zeros((self.nodes, 2), bool)
        fixed = np.asarray(fixed)
        if fixed.shape != (self.nodes, 2) or fixed.dtype != bool:
            raise ValueError('fixed requires bool (nodes,2)')
        answer = np.zeros_like(force)
        for component in range(2):
            ids, reduced, factor = self._factor(~fixed[:, component])
            if not len(ids):
                continue
            rhs = force[ids, component]; solved = factor.solve(rhs)
            residual = np.linalg.norm(reduced @ solved - rhs)
            scale = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
            relative = float(residual/scale)
            if not np.isfinite(solved).all() or relative > self.solve_tolerance:
                raise ValueError('Physical free mass solve failed its residual')
            answer[ids, component] = solved
            self.diagnostics['maximum_solve_relative_residual'] = max(
                relative, self.diagnostics['maximum_solve_relative_residual'])
        self.diagnostics['solves'] += 1
        return answer

    def multiply(self, velocity):
        return np.asarray(self.M @ self._nodal(velocity, 'Velocity'))

    def project_velocity(self, velocity, fixed):
        """M-orthogonal stationary support projection, not component clipping."""
        return self.solve(self.multiply(velocity), fixed)

    def kinetic(self, velocity):
        velocity = self._nodal(velocity, 'Velocity')
        return .5*float(np.sum(velocity*self.multiply(velocity)))

    def momentum(self, velocity):
        return self.multiply(velocity).sum(axis=0)

    def angular_momentum(self, q, velocity):
        q = self._nodal(q, 'Current coordinates'); momentum = self.multiply(velocity)
        return float(np.sum(q[:, 0]*momentum[:, 1] - q[:, 1]*momentum[:, 0]))

    def minimum_corner_determinant(self, q):
        return float(np.min(self.corner_jacobians(q)))
