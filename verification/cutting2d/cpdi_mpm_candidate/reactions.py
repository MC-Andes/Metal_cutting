"""Separate a physical bottom impulse from the MPM representation residual.

With C = R[bottom], solve R.T r = C.T jb; the remaining material nodal
impulse has zero virtual work in the *unrestricted* background space. It is
not a support load. For the stationary flat bottom C = Wx tensor wy. The
small, column-balanced inverse of Wx.T chooses the minimum Euclidean norm
bottom distribution; its resultant and torque are independent of this gauge.

All residuals are evaluated after the same full-grid column equilibration
used before support elimination. A small unscaled grid residual alone can
hide errors in weak B2 columns. No residual is redistributed to close a
balance, and no force, mass, velocity, or accepted state is modified here.
"""
import numpy as np
from scipy import sparse


class ReactionFailure(ValueError):
    """A reaction identity failed; retain its measurements for the caller."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _norm(value):
    # Scaling also handles very small physical impulse magnitudes without
    # squaring them below the floating-point range.
    value = np.asarray(value, float)
    largest = float(np.max(np.abs(value), initial=0.))
    return 0. if largest == 0. else largest * float(np.linalg.norm(value/largest))


def _relative(value, scale):
    magnitude = _norm(value)
    return magnitude/scale if scale > 0. else (0. if magnitude == 0. else np.inf)


class ReactionRecovery:
    """Reusable extractor for one immutable GridMechanics assembly.

    ``reference`` is an impulse envelope, in Ns, of shape (material_nodes,2).
    Prefer an arithmetic envelope formed *before* subtraction, for example
    abs(M0)@(abs(vnew)+abs(vold))+abs(dt*f)+abs(J). Separate equation checks
    on the applied loads remain necessary; this bounds cancellation error.
    The default 1e-11 relative guard matches GridMechanics. The separate
    1e-14 Ns resultant bound is dimensional; its torque counterpart uses
    the largest nodal distance from the torque origin (or one grid spacing).
    These checks certify the split, not acceptance of an integrated cut.
    """
    def __init__(self, op, *, relative_tolerance=1e-11, momentum_absolute_Ns=1e-14):
        if (not np.isfinite(relative_tolerance) or relative_tolerance <= 0. or
                not np.isfinite(momentum_absolute_Ns) or momentum_absolute_Ns < 0.):
            raise ValueError('Finite positive reaction tolerance and nonnegative impulse bound required')
        self.op = op
        self.relative_tolerance = float(relative_tolerance)
        self.momentum_absolute_Ns = float(momentum_absolute_Ns)
        transfer = op.transfer
        norms = np.sqrt(np.asarray(transfer.H.multiply(transfer.H).sum(0)).ravel())
        scale = np.ones(transfer.nodes)
        scale[norms > 0.] = 1./norms[norms > 0.]
        self.scale = scale
        self.A = (transfer.R @ sparse.diags(scale)).tocsr()
        self.groups = []
        self.factor = None
        self.factor_diagnostics = {}
        if not np.any(op.fixed):
            return
        trace = op.trace
        if (trace is None or not np.array_equal(op.q[op.bottom], trace.q) or
                transfer.stencil.h != trace.h or
                not np.array_equal(transfer.stencil.origin, trace.origin)):
            raise ValueError('Reaction recovery requires the current certified bottom trace')
        lattice = transfer.stencil.lattice_nodes
        lookup = {tuple(address): i for i, address in enumerate(lattice)}
        C = transfer.R[op.bottom]
        # Own the horizontal map and its factorization; BottomTrace is not
        # mutated and no unscaled dense global nullspace is constructed.
        self.Wx = np.column_stack([
            np.asarray(C[:, lattice[:, 0] == ix].sum(1)).ravel() for ix in trace.xs])
        column_norms = np.linalg.norm(self.Wx, axis=0)
        if np.any(column_norms <= 0.):
            raise ValueError('Reaction horizontal trace contains a structural zero column')
        inverse_norms = 1./column_norms
        U, singular, Vt = np.linalg.svd(self.Wx*inverse_norms, full_matrices=False)
        cutoff = max(self.Wx.shape)*np.finfo(float).eps*singular[0]
        if self.Wx.shape[0] < self.Wx.shape[1] or singular[-1] <= 32.*cutoff:
            raise ValueError('Reaction horizontal trace rank is not certified')
        self.factor = (U, singular, Vt, inverse_norms)
        tensor_error = 0.
        for col, ix in enumerate(trace.xs):
            ids = np.array([lookup[(ix, iy)] for iy in trace.ys])
            normal = trace.wy*scale[ids]
            length = _norm(normal)
            self.groups.append((ids, normal/length, length))
            expected = self.Wx[:, col, None]*normal
            actual = self.A[op.bottom][:, ids].toarray()
            tensor_error = max(tensor_error, _relative(actual-expected, _norm(actual)))
        if tensor_error > 128.*np.finfo(float).eps:
            raise ValueError('Equilibrated reaction trace tensor identity failed')
        self.factor_diagnostics = dict(
            horizontal_scaled_sigma_min=float(singular[-1]),
            horizontal_scaled_condition=float(singular[0]/singular[-1]),
            horizontal_rank_threshold=float(cutoff),
            equilibrated_tensor_relative=tensor_error,
            horizontal_columns=int(self.Wx.shape[1]))

    def recover(self, reaction, reference):
        op = self.op
        reaction = op.domain._nodal(reaction, 'Physical material constraint impulse')
        reference = np.abs(op.domain._nodal(reference, 'Physical impulse scale envelope'))
        support = np.zeros_like(reaction)
        balanced_scale = _norm(self.A.T @ reference)
        dual = np.asarray(self.A.T @ reaction)
        trace_residuals = []
        for component in range(2):
            if not np.any(op.fixed[:, component]):
                trace_residuals.append(0.)
                continue
            chi = np.array([np.dot(normal, dual[ids, component])/length
                            for ids, normal, length in self.groups])
            U, singular, Vt, inverse_norms = self.factor
            jb = U @ ((Vt @ (inverse_norms*chi))/singular)
            support[op.bottom, component] = jb
            trace_residuals.append(_relative(
                inverse_norms*(self.Wx.T @ jb-chi),
                _norm(inverse_norms*chi)))
        complement = reaction-support
        balanced_complement = self.A.T @ complement
        balanced_residual = _relative(balanced_complement, balanced_scale)
        # Use the actual supported material map W, including certified
        # symbolic trace zeros, for work conjugacy with GridMechanics.
        total_virtual = op.virtual_work_residual(reaction, reference)
        complement_virtual = op.virtual_work_residual(complement, reference)
        support_virtual = op.virtual_work_residual(support, reference)
        momentum = complement.sum(0)
        q = op.q
        torque = float(np.sum(q[:, 0]*complement[:, 1]-q[:, 1]*complement[:, 0]))
        momentum_scale = _norm(reference.sum(0))
        lever = max(float(np.max(np.linalg.norm(q, axis=1))), float(op.transfer.stencil.h))
        torque_scale = float(np.sum(np.abs(q[:, 0])*reference[:, 1]+
                                    np.abs(q[:, 1])*reference[:, 0]))
        momentum_limit = max(self.momentum_absolute_Ns, self.relative_tolerance*momentum_scale)
        torque_limit = max(self.momentum_absolute_Ns*lever, self.relative_tolerance*torque_scale)
        diag = dict(
            formulation='balanced_full_grid_reaction_split_v1',
            bottom_gauge='minimum Euclidean norm solution of Wx.T jb = chi',
            reaction_units='Ns', relative_tolerance=self.relative_tolerance,
            full_grid_balanced_residual=balanced_residual,
            full_grid_complement_norm=_norm(balanced_complement),
            full_grid_reference_norm=balanced_scale,
            total_virtual_work_residual=total_virtual,
            complement_virtual_work_residual=complement_virtual,
            support_virtual_work_residual=support_virtual,
            support_admissible_work_J=0.,
            support_work_reason='support acts only on exactly fixed velocity components',
            bottom_inverse_relative_by_component=trace_residuals,
            complement_impulse_Ns=momentum.tolist(),
            complement_angular_impulse_Nms=torque,
            complement_impulse_limit_Ns=momentum_limit,
            complement_angular_impulse_limit_Nms=torque_limit,
            support_impulse_Ns=support.sum(0).tolist(),
            support_angular_impulse_Nms=float(np.sum(q[:, 0]*support[:, 1]-q[:, 1]*support[:, 0])),
            complement_nodal_norm_Ns=_norm(complement),
            total_reaction_nodal_norm_Ns=_norm(reaction),
            support_nodal_norm_Ns=_norm(support),
            reference_nodal_envelope_norm_Ns=_norm(reference),
            no_residual_redistribution=True,
            **self.factor_diagnostics)
        if (not np.isfinite(support).all() or not np.isfinite(complement).all() or
                not np.isfinite(balanced_residual) or
                max(balanced_residual, total_virtual, complement_virtual, support_virtual) > self.relative_tolerance):
            raise ReactionFailure('MPM reaction does not split into bottom support and full-grid orthogonal complement', diag)
        if _norm(momentum) > momentum_limit or abs(torque) > torque_limit:
            raise ReactionFailure('MPM representation complement carries a nonzero resultant or torque', diag)
        if np.any(support[~op.fixed] != 0.):
            raise ReactionFailure('Recovered support acts on an unconstrained material component', diag)
        return support, complement, diag


def recover_reactions(op, reaction, reference):
    """Return (physical_bottom_impulse, representation_complement, diagnostics)."""
    return ReactionRecovery(op).recover(reaction, reference)
