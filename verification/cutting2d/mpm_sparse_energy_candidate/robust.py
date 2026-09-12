"""Numerical Gram-to-QR fallback in an unchanged certified column gauge.

The existing fast cache certifies the current range and retained column IDs.
QR is only a second way to solve in those same coordinates: no new rank
inference, pivot proposal, regularization, tolerance retry, or dropped row.
Every call first attempts the original Gram path. A QR factor, built lazily
after a numerical guard rejects, may be reused on later rejected calls for
this same immutable map. It is not a persistent trajectory state.
"""
import copy
import time

import numpy as np
from scipy import sparse

from cutting2d.energy import ProjectorFailure
from .fast_range import CertifiedFastRangeGramCache, GRAM_METHOD, UNSQUARED_METHOD
from .qr import SparseQRProjector
from .reduced import CertifiedRangeGramProjector


CURRENT_METHOD = 'retained_columns_Weyl_and_COMPLETE_current_dependency_reconstruction_v1'
NUMERICAL_FAILURES = (
    'Current full physical range-Gram checks failed:',
    'Nonfinite range-Gram projection',
    'Nonfinite range-Gram response',
    'MPM external momentum check failed:',
)


def _current_certificate(base):
    """Require the cache's complete proof, not QR success or random probes.

This validates the trusted in-memory factor contract. Checkpoint authenticity
and regeneration of the proof remain the existing cache reader's work.
"""
    proof = base.diagnostics.get('rank_certificate')
    if not isinstance(proof, dict):
        raise ProjectorFailure('QR fallback requires an existing complete range certificate')
    certificate = proof.get('current_range', proof.get('certificate'))
    methods = (GRAM_METHOD, UNSQUARED_METHOD, CURRENT_METHOD)
    if (not isinstance(certificate, dict) or certificate.get('certified') is not True or
            certificate.get('method') not in methods or
            certificate.get('rank') != base.rank or
            certificate.get('rows') != base.Z.shape[0] or
            certificate.get('columns') != base.Z.shape[1] or
            certificate.get('retained_separated') is not True or
            certificate.get('discarded_separated') is not True or
            not np.isfinite(certificate.get('sigma_r_lower', np.nan)) or
            certificate['sigma_r_lower'] <= 0.):
        raise ProjectorFailure('QR fallback requires a complete separated proof of this current range')
    return copy.deepcopy(certificate)


class RobustRangeGramProjector:
    """Wrap a certified current Gram factor without changing its basis.

Caller must retain the usual immutable-factor ownership: changing base.Z,
base.A, or its retained IDs after wrapping is unsupported. No proof is
inferred from a caller-supplied boolean or from the fallback's QR probes.
"""
    def __init__(self, base, *, external_Z=None):
        if not isinstance(base, CertifiedRangeGramProjector):
            raise TypeError('An existing certified current range-Gram factor is required')
        if not 0 < base.rtol <= 1e-11:
            raise ValueError('Physical residual tolerance may not exceed 1e-11')
        certificate = _current_certificate(base)
        self.base = base
        self.Z = base.Z
        self.A = base.A
        self.retained_columns = base.retained_columns
        self.rank = base.rank
        self.rtol = base.rtol
        self.frobenius = base.frobenius
        # Preserve the exact supplied CSR summation order used by mechanics.
        # The default supports direct wrapper use on an existing factor.
        self.external_Z = sparse.csr_matrix(base.Z if external_Z is None else external_Z, copy=True)
        if (self.external_Z.shape != self.Z.shape or
                not np.isfinite(self.external_Z.data).all() or
                (self.external_Z-self.Z).nnz != 0):
            raise ValueError('External momentum check must use the same physical Z')
        self._qr = None
        self.diagnostics = copy.deepcopy(base.diagnostics)
        self.diagnostics['backend'] = 'certified_same_columns_Gram_then_QR_v1'
        self.diagnostics['numerical_fallback'] = dict(
            policy='Gram first on every call; QR only after a numerical guard rejects',
            rank_source=certificate['method'], rank=self.rank,
            coordinates=self.Z.shape[1], dependent_coordinates=self.Z.shape[1]-self.rank,
            same_retained_ids=True, no_rank_retry=True, no_tolerance_retry=True,
            gram_successes=0, fallback_attempts=0, qr_successes=0,
            qr_failures=0, qr_constructions=0, qr_construction_seconds=0.,
            last_method=None, last_route=None, last_original_failure=None,
            first_original_failure=None, last_qr_failure=None,
            last_full_Z_guard=None, last_QR_guard=None, last_external_guard=None,
        )

    def _fallback_factor(self):
        if self._qr is None:
            started = time.perf_counter()
            # The cache has already proved A independent and the full range
            # complete. probes_only is a necessary numerical check, not the
            # proof. QR is forbidden to discard any of those proven columns.
            qr = SparseQRProjector(self.A, audit='probes_only', rtol=self.rtol)
            if (qr.rank != self.rank or qr.Z.shape != self.A.shape or
                    not np.array_equal(np.sort(qr.retained_columns), np.arange(self.rank))):
                raise ProjectorFailure('Fallback QR did not retain every certified independent column')
            self._qr = qr
            record = self.diagnostics['numerical_fallback']
            record['qr_constructions'] += 1
            record['qr_construction_seconds'] += time.perf_counter()-started
            record['QR_factor_diagnostics'] = copy.deepcopy(qr.diagnostics)
        return self._qr

    def _full_projection(self, y):
        qr = self._fallback_factor()
        _, reduced, qr_record = qr.project(y, return_coefficients=True)
        c = np.zeros((self.Z.shape[1], y.shape[1]))
        c[self.retained_columns] = reduced
        p = self.Z @ c
        largest = float(np.max(abs(y), initial=0.))
        if largest == 0.:
            normal = energy = 0.
        else:
            # Same full-Z normalization and identity as reduced.py.project.
            yn = y/largest; pn = p/largest; r = yn-pn
            scale = float(np.linalg.norm(yn))
            normal = float(np.linalg.norm(self.Z.T @ r))/(self.frobenius*scale)
            energy = abs(float(np.sum(yn*yn)-np.sum(pn*pn)-np.sum(r*r)))/scale**2
        record = dict(normal_relative=normal, energy_identity_relative=energy)
        self.base._guard(record)
        if not np.isfinite(c).all() or not np.isfinite(p).all():
            raise ProjectorFailure('Nonfinite full-Z fallback projection')
        return p, c, record, qr_record

    def _full_solve(self, b):
        qr = self._fallback_factor()
        reduced, _, qr_record = qr.solve_gram(b[self.retained_columns], return_response=True)
        c = np.zeros_like(b)
        c[self.retained_columns] = reduced
        p = self.Z @ c
        largest = float(np.max(abs(b), initial=0.))
        if largest == 0.:
            residual = energy = 0.
        else:
            # Reconstruct Zc; the internal QR response alone is insufficient.
            bn = b/largest
            residual = float(np.linalg.norm(self.Z.T @ (p/largest)-bn))/float(np.linalg.norm(bn))
            plargest = float(np.max(abs(p), initial=0.))
            if plargest == 0.:
                energy = 0.
            else:
                pn = p/plargest
                energy = abs(float(np.sum((b/plargest)*(c/plargest))-np.sum(pn*pn)))/float(np.sum(pn*pn))
        record = dict(full_Gram_residual_relative=residual, dual_energy_identity_relative=energy)
        self.base._guard(record)
        if not np.isfinite(c).all() or not np.isfinite(p).all():
            raise ProjectorFailure('Nonfinite full-Z fallback response')
        return c, p, record, qr_record

    def _external_guard(self, method, value, c):
        """Literal GridMechanics formulas, per physical RHS/component.

        No absolute load floor beyond the existing binary64 tiny, and no
        replacement of the externally checked state or right-hand side.
        """
        Z = self.external_Z
        relatives = []
        for column in range(c.shape[1]):
            supplied = value[:, column]
            mapped = Z @ c[:, column]
            if method == 'project':
                residual = Z.T @ (mapped-supplied)
                scale = max(float(np.linalg.norm(Z.T @ supplied)), np.finfo(float).tiny)
            else:
                residual = Z.T @ mapped-supplied
                scale = max(float(np.linalg.norm(supplied)), np.finfo(float).tiny)
            relatives.append(float(np.linalg.norm(residual)/scale))
        record = dict(method=method, per_rhs_relative=relatives,
            maximum_relative=max(relatives, default=0.))
        if not np.isfinite(relatives).all() or record['maximum_relative'] > self.rtol:
            raise ProjectorFailure('MPM external momentum check failed: '+str(record))
        return record

    def _evaluate(self, method, value):
        record = self.diagnostics['numerical_fallback']
        record['last_method'] = method
        try:
            if method == 'project':
                first, second, checked = self.base.project(value, return_coefficients=True)
            else:
                first, second, checked = self.base.solve_gram(value, return_response=True)
            c = second if method == 'project' else first
            external_checked = self._external_guard(method, value, c)
        except ProjectorFailure as original:
            if not str(original).startswith(NUMERICAL_FAILURES):
                raise
            original_text = str(original)
            record['fallback_attempts'] += 1
            record['last_original_failure'] = original_text
            if record['first_original_failure'] is None:
                record['first_original_failure'] = original_text
            try:
                operation = self._full_projection if method == 'project' else self._full_solve
                first, second, checked, qr_checked = operation(value)
                c = second if method == 'project' else first
                external_checked = self._external_guard(method, value, c)
            except (ProjectorFailure, ValueError, RuntimeError) as fallback:
                record['qr_failures'] += 1
                record['last_route'] = 'rejected_Gram_and_QR'
                record['last_qr_failure'] = type(fallback).__name__+': '+str(fallback)
                raise ProjectorFailure('Same-column Gram-to-QR fallback failed; original Gram: '+
                    original_text+'; QR/full-Z: '+record['last_qr_failure']) from fallback
            record['qr_successes'] += 1
            record['last_route'] = 'QR_same_certified_columns'
            record['last_QR_guard'] = qr_checked
        else:
            record['gram_successes'] += 1
            record['last_route'] = 'Gram'
        record['last_full_Z_guard'] = checked
        record['last_external_guard'] = external_checked
        return first, second, checked

    def project(self, y, *, return_coefficients=False):
        y, vector = self.base._input(y, self.Z.shape[0])
        p, c, record = self._evaluate('project', y)
        if vector:
            p = p[:, 0]; c = c[:, 0]
        return (p, c, record) if return_coefficients else (p, record)

    def solve_gram(self, b, *, return_response=False):
        b, vector = self.base._input(b, self.Z.shape[1])
        c, p, record = self._evaluate('solve_gram', b)
        if vector:
            c = c[:, 0]; p = p[:, 0]
        return (c, p, record) if return_response else (c, record)

    def project_coefficients(self, y):
        return self.project(y, return_coefficients=True)[1]

    def dual_coefficients(self, b):
        return self.solve_gram(b)[0]


class CertifiedRobustRangeGramCache(CertifiedFastRangeGramCache):
    """Same complete certificate and snapshot mathematics; new solve backend.

The invocation/backend must distinguish this class from the historical fast
cache. A cached QR factor is deliberately not checkpointed: it is a lazy
factorization of the current A, while the immutable range cache is unchanged.
"""
    def factor(self, Z, coordinate_identity):
        return RobustRangeGramProjector(super().factor(Z, coordinate_identity), external_Z=Z)
