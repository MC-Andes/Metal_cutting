"""Local quadratic B-spline APIC on a sparse, unbounded Cartesian grid.

The stored affine state is A (1/s), with B = A D and D = h**2 I / 4.
This module implements the explicit, fixed-grid transfer. It does not evolve
positions, deformation, constitutive history, boundary conditions or contact.
In particular, A is not a substitute for the B-spline velocity gradient.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Stencil:
    """Nine full support slots per particle; structural zero slots are retained."""

    nodes: np.ndarray
    lattice_nodes: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    gradients: np.ndarray
    offsets: np.ndarray
    positions: np.ndarray
    h: float
    origin: np.ndarray

    @property
    def second_moment(self) -> np.ndarray:
        return np.eye(2) * (self.h * self.h / 4.0)


def _array(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return value


def _mass(value, count):
    value = _array(value, (count,), "particle mass")
    if np.any(value <= 0.0):
        raise ValueError("particle mass must be strictly positive")
    return value


def build_stencil(x, h, origin=None) -> Stencil:
    """Build the complete 3x3 tensor B2 support; no truncation or normalization.

    ``nodes`` are physical coordinates and ``lattice_nodes`` are integer grid
    addresses. A sparse union includes the halo of every particle, including
    particles beyond any previous support. Grid bounds are not physical walls.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("particle positions must be finite with shape (N, 2)")
    h = float(h)
    if not np.isfinite(h) or h <= 0.0 or not np.isfinite(h * h) or h * h == 0:
        raise ValueError("h must be positive with representable squared spacing")
    origin = _array(np.zeros(2) if origin is None else origin, (2,), "origin")
    q = (x - origin) / h
    if not np.isfinite(q).all() or np.any(np.abs(q) >= 2**50):
        raise ValueError("grid coordinates cannot resolve the local particle phase")
    base = np.floor(q - 0.5).astype(np.int64)
    fraction = q - base
    w = np.stack((0.5 * (1.5 - fraction)**2,
                  0.75 - (fraction - 1.0)**2,
                  0.5 * (fraction - 0.5)**2), axis=1)
    dw = np.stack((fraction - 1.5, -2.0 * (fraction - 1.0),
                   fraction - 0.5), axis=1) / h
    ix, iy = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    ij = np.column_stack((ix.ravel(), iy.ravel()))
    addresses = base[:, None, :] + ij[None, :, :]
    lattice_nodes, inverse = np.unique(addresses.reshape(-1, 2), axis=0,
                                       return_inverse=True)
    weights = w[:, ij[:, 0], 0] * w[:, ij[:, 1], 1]
    gradients = np.stack((dw[:, ij[:, 0], 0] * w[:, ij[:, 1], 1],
                          w[:, ij[:, 0], 0] * dw[:, ij[:, 1], 1]), axis=2)
    # Local coordinates avoid subtracting two large physical node coordinates.
    offsets = (ij[None, :, :] - fraction[:, None, :]) * h
    result = Stencil(nodes=origin + lattice_nodes * h,
                     lattice_nodes=lattice_nodes,
                     indices=inverse.reshape(len(x), 9), weights=weights,
                     gradients=gradients, offsets=offsets, positions=x.copy(),
                     h=h, origin=origin.copy())
    for value in vars(result).values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return result


def _scatter(stencil, values):
    ids = stencil.indices.ravel()
    n = len(stencil.nodes)
    if values.ndim == 2:
        return np.bincount(ids, weights=values.ravel(), minlength=n)
    return np.column_stack([np.bincount(ids, weights=values[..., k].ravel(),
                                       minlength=n)
                            for k in range(values.shape[-1])])


def p2g(stencil: Stencil, mass, velocity, affine):
    """Deposit lumped mass and affine momentum; return (mass, momentum)."""
    n = len(stencil.positions)
    mass = _mass(mass, n)
    velocity = _array(velocity, (n, 2), "particle velocity")
    affine = _array(affine, (n, 2, 2), "particle affine state")
    weighted_mass = mass[:, None] * stencil.weights
    local_velocity = velocity[:, None, :] + np.einsum(
        "pab,pib->pia", affine, stencil.offsets)
    return (_scatter(stencil, weighted_mass),
            _scatter(stencil, weighted_mass[..., None] * local_velocity))


def grid_velocity(grid_mass, grid_momentum):
    """Divide every strictly positive mass, without a dimensioned mass cutoff.

    Structural zero-mass nodes have zero velocity and must have zero momentum.
    Nonfinite quotients are explicit errors, never masked by mass augmentation.
    """
    grid_mass = np.asarray(grid_mass, dtype=np.float64)
    if grid_mass.ndim != 1 or not np.isfinite(grid_mass).all() or np.any(grid_mass < 0):
        raise ValueError("grid mass must be a finite nonnegative vector")
    momentum = _array(grid_momentum, (len(grid_mass), 2), "grid momentum")
    active = grid_mass > 0.0
    if np.any(momentum[~active] != 0.0):
        raise ValueError("nonzero momentum at a zero-mass node")
    result = np.zeros_like(momentum)
    np.divide(momentum, grid_mass[:, None], out=result, where=active[:, None])
    if not np.isfinite(result).all():
        raise FloatingPointError("grid velocity overflow; state requires rescaling")
    return result


def g2p(stencil: Stencil, grid_velocity):
    """Recover both v and A from the same accepted grid velocity.

    Uses the pre-advection stencil. For the explicit update x_new=x+dt*v_new,
    the orbital angular increment v_new cross (m*v_new) is zero.
    """
    velocity = _array(grid_velocity, (len(stencil.nodes), 2), "grid velocity")
    local = velocity[stencil.indices]
    vp = np.einsum("pi,pia->pa", stencil.weights, local)
    affine = np.einsum("pi,pia,pib->pab", stencil.weights, local,
                       stencil.offsets) * (4.0 / stencil.h**2)
    return vp, affine


def velocity_gradient(stencil: Stencil, grid_velocity):
    """L_ab = sum_i v_ia * partial_b(w_ip), conjugate to internal_force."""
    velocity = _array(grid_velocity, (len(stencil.nodes), 2), "grid velocity")
    return np.einsum("pia,pib->pab", velocity[stencil.indices], stencil.gradients)


def internal_force(stencil: Stencil, volume, stress):
    """f_i = -sum_p V_current,p sigma_p grad(w_ip), SI Cauchy stress.

    A 3D plane-strain stress is accepted: in-plane balance uses its xy block.
    The constitutive state retains sigma_zz; it is not set to zero here.
    """
    n = len(stencil.positions)
    volume = _array(volume, (n,), "current volume")
    if np.any(volume <= 0.0):
        raise ValueError("current volume must be strictly positive")
    stress = np.asarray(stress, dtype=np.float64)
    if stress.shape not in ((n, 2, 2), (n, 3, 3)) or not np.isfinite(stress).all():
        raise ValueError("Cauchy stress must be finite with shape (N, 2, 2) or (N, 3, 3)")
    local = -volume[:, None, None] * np.einsum(
        "pab,pib->pia", stress[:, :2, :2], stencil.gradients)
    return _scatter(stencil, local)


def particle_diagnostics(stencil: Stencil, mass, velocity, affine):
    """APIC representation energy and angular momentum about global origin.

    D is isotropic, hence spin = m*h²/4*(A_yx-A_xy). Representation energy
    includes the transported A; it is not a correction fitted to a deficit.
    """
    n = len(stencil.positions)
    mass = _mass(mass, n)
    velocity = _array(velocity, (n, 2), "particle velocity")
    affine = _array(affine, (n, 2, 2), "particle affine state")
    d = stencil.h**2 / 4.0
    point = float(0.5 * np.einsum("p,pa,pa->", mass, velocity, velocity))
    aff = float(0.5 * d * np.einsum("p,pab,pab->", mass, affine, affine))
    orbital = float(np.sum(mass * (stencil.positions[:, 0] * velocity[:, 1] -
                                   stencil.positions[:, 1] * velocity[:, 0])))
    spin = float(np.sum(mass * d * (affine[:, 1, 0] - affine[:, 0, 1])))
    return {"mass": float(mass.sum()), "kinetic_point": point,
            "kinetic_affine": aff, "kinetic_representation": point + aff,
            "linear_momentum": np.einsum("p,pa->a", mass, velocity),
            "angular_orbital": orbital, "angular_affine": spin,
            "angular_momentum": orbital + spin}


def grid_diagnostics(nodes, mass, velocity):
    """Grid kinetic energy and momenta using its actual lumped masses."""
    mass = np.asarray(mass, dtype=np.float64)
    if mass.ndim != 1 or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("grid mass must be a finite nonnegative vector")
    nodes = _array(nodes, (len(mass), 2), "grid nodes")
    velocity = _array(velocity, (len(mass), 2), "grid velocity")
    return {"mass": float(mass.sum()),
            "kinetic": float(0.5 * np.einsum("p,pa,pa->", mass, velocity, velocity)),
            "linear_momentum": np.einsum("p,pa->a", mass, velocity),
            "angular_momentum": float(np.sum(mass * (
                nodes[:, 0] * velocity[:, 1] - nodes[:, 1] * velocity[:, 0])))}
