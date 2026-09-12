"""Transactional explicit MPM on a reset B2 grid with shared domains.

Persistent material Q4 geometry/quadrature and constitutive/thermal routines
are reused from the reference components. Every momentum and contact solve
is restricted to the current background space; no direct material dynamics
solve is substituted for the grid equation.
"""
import copy
import numpy as np
from cutting2d.cases import make_tool
from verification.cutting2d.direct_q4_candidate.solver import DirectCutting
from verification.cutting2d.direct_q4_candidate.thermal_subcycling import advance as advance_thermal
from .mechanics import GridMechanics
from .contact import solve_contact


class MPMCutting(DirectCutting):
    def __init__(self,case,*,grid_h,material_h=None,origin=None,backend='dense',threads=1,
                 geometry_backend='cached',constitutive_options=None):
        self.grid_h=float(grid_h);self.material_h=float(material_h if material_h is not None else grid_h/2)
        self.origin=np.zeros(2) if origin is None else np.asarray(origin,float).copy();self.backend=backend
        if self.origin.shape!=(2,) or not np.isfinite(self.origin).all():raise ValueError('Finite background grid origin required')
        super().__init__(case,h=self.material_h,threads=threads,geometry_backend=geometry_backend,
            mesh_profile='uniform',constitutive_options=constitutive_options)
        self.h=self.grid_h
        self.penetration_tolerance=min(case['acceptance']['penetration_absolute_m'],self.grid_h*case['acceptance']['penetration_over_h'])
        self.bottom_trace=None;self._rank_caches={};self._rank_trial_backup=None
        self.mesh_diagnostics.update(background_h_m=self.grid_h,material_h_m=self.material_h,
            formulation='MPM_shared_domain_B2_four_persistent_GP_v1')

    def grid_operator(self):
        op=GridMechanics(self.model,self.q,self.grid_h,self.fixed,origin=self.origin,
            bottom_trace=self.bottom_trace,backend=self.backend,rank_caches=self._rank_caches)
        self.bottom_trace=op.trace
        return op

    def _begin_operator_trial(self):
        # References/certificates are immutable; factor() replaces them rather
        # than mutating them. Copy only cache handles, counters and diagnostics.
        self._rank_trial_backup={key:copy.copy(cache) for key,cache in self._rank_caches.items()}

    def commit(self,trial):
        record=super().commit(trial)
        self._rank_trial_backup=None
        return record

    def reject(self):
        super().reject()
        if self._rank_trial_backup is not None:
            self._rank_caches=self._rank_trial_backup;self._rank_trial_backup=None

    def snapshot_operator_cache(self):
        # A checkpoint represents committed material and committed proofs.
        caches=self._rank_caches if self._rank_trial_backup is None else self._rank_trial_backup
        return {key:caches[key].snapshot() for key in sorted(caches)}

    def restore_operator_cache(self,payload):
        if not isinstance(payload,dict) or set(payload)-{'free','supported'}:
            raise ValueError('Invalid MPM operator verification cache schema')
        if self.backend not in ('certified_gram','certified_range','certified_range_fast','certified_range_robust') and payload:
            raise ValueError('MPM operator verification cache requires certified Gram')
        restored={}
        if payload:
            if self.backend=='certified_gram':
                from verification.cutting2d.mpm_sparse_energy_candidate.accelerated import CertifiedGramCache as Cache
            elif self.backend=='certified_range':
                from verification.cutting2d.mpm_sparse_energy_candidate.reduced import CertifiedRangeGramCache as Cache
            elif self.backend=='certified_range_fast':
                from verification.cutting2d.mpm_sparse_energy_candidate.fast_range import CertifiedFastRangeGramCache as Cache
            else:
                from verification.cutting2d.mpm_sparse_energy_candidate.robust import CertifiedRobustRangeGramCache as Cache
            for key in sorted(payload):restored[key]=Cache.restore(payload[key])
        # Publish only after every proof passes. No current Gram/LU or
        # particle/grid coefficients are carried through a restart.
        self._rank_caches=restored;self._rank_trial_backup=None

    def trial(self,dt):
        self.reject();self._begin_operator_trial();m=self.model;old=self.material.state
        if not np.isfinite(dt) or dt<=0.:raise ValueError('Positive finite MPM timestep required')
        from .reactions import recover_reactions
        op=self.grid_operator()
        self.conductor.update_geometry(self.q);diffusion=advance_thermal(self.conductor,old.T,dt)
        projected,reset=op.project_velocity(self.v)
        force=m.force_from_piola(old.first_piola)
        free=projected+dt*op.solve(force)
        kick_constraint=m.multiply(free-self.v)-dt*force
        support_kick,complement_kick,kick_record=recover_reactions(op,kick_constraint,
            np.asarray(abs(m.M)@(abs(free)+abs(self.v)))+abs(dt*force))
        points,_=self.geometry.contact_candidates(self.q,make_tool(self.config,self.time),gap_tolerance=self.penetration_tolerance)
        velocity,contact=solve_contact(op,self.q,points,free,self.tool_velocity,dt,self.fixed,
            mu=self.case['thermal']['friction_coefficient'])
        support_contact,complement_contact,contact_record=recover_reactions(op,contact['constraint_reaction'],
            np.asarray(abs(m.M)@(abs(velocity)+abs(free)))+abs(contact['nodal_contact_impulse']))
        contact['support_reaction']=support_contact
        result=self.finish_trial(dt,velocity,contact,diffusion,support_kick)
        result.diagnostics.update(integrator='MPM_symplectic_euler_shared_domain_v1',
            contact_solver_resolution=contact['resolution'],
            mpm=op.diagnostics,mpm_reset=reset,mpm_kick_reaction=kick_record,mpm_contact_reaction=contact_record,
            mpm_complement_impulse_Ns=(complement_kick+complement_contact).sum(0))
        # Separate the orthogonal reset loss from integration and contact.
        # The original physical total energy defect is left unchanged.
        result.diagnostics['mechanical_integration_defect_J']+=reset['reset_kinetic_loss_J']
        return result
