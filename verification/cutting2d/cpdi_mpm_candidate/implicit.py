"""Backward Euler in the frozen pre-advection MPM background space.

The final residual uses physical inertia and nominal finite-JC stress.
The tangent supplies a Newton iteration metric only. Constitutive histories
commit once, after geometry, contact, heat and equilibrium checks all pass.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from cutting2d.cases import make_tool
from verification.cutting2d.direct_q4_candidate.implicit import material_stiffness
from verification.cutting2d.direct_q4_candidate.solver import PreparedMaterialTrial
from verification.cutting2d.direct_q4_candidate.coupling import scatter_heat
from verification.cutting2d.direct_q4_candidate.thermal_subcycling import advance as advance_thermal
from .solver import MPMCutting
from .contact import solve_contact


class GridNewtonMetric:
    def __init__(self,op,A):
        self.op=op;self.A=A;self.fixed=op.fixed;self.shape=op.fixed.shape
        rows=[];cols=[];values=[];offset=0
        for a,data in enumerate(op.components):
            factor=data['factor']
            if hasattr(factor,'retained_columns'):basis=data['W'][:,factor.retained_columns].tocoo()
            elif hasattr(factor,'V'):basis=sparse.coo_matrix(data['W']@factor.V)
            else:raise ValueError('MPM Newton requires an independently checked full-rank coordinate basis')
            rows.extend(2*basis.row+a);cols.extend(basis.col+offset);values.extend(basis.data);offset+=basis.shape[1]
        self.lift=sparse.coo_matrix((values,(rows,cols)),shape=(op.nodes*2,offset)).tocsr()
        self.reduced=(self.lift.T@A@self.lift).tocsc();self.factor=splu(self.reduced)

    def solve(self,force,fixed):
        if not np.array_equal(fixed,self.fixed):raise ValueError('MPM Newton material trace mismatch')
        rhs=self.lift.T@np.asarray(force).ravel();c=self.factor.solve(np.asarray(rhs));answer=self.lift@c
        residual=self.reduced@c-rhs
        if np.linalg.norm(residual)>1e-10*max(float(np.linalg.norm(rhs)),np.finfo(float).tiny):
            raise ValueError('MPM Newton linear background equation failed')
        return np.asarray(answer).reshape(self.shape)

    def virtual_work_residual(self,reaction,reference):
        top=np.linalg.norm(self.lift.T@np.asarray(reaction).ravel())
        scale=np.linalg.norm(abs(self.lift).T@abs(np.asarray(reference).ravel()))
        return float(top/max(scale,np.finfo(float).tiny))

    def multiply(self,v):return np.asarray(self.A@np.asarray(v).ravel()).reshape(self.shape)
    def reaction_envelope(self,new,old,load):
        value=abs(self.A)@(abs(np.asarray(new).ravel())+abs(np.asarray(old).ravel()))
        return np.asarray(value).reshape(self.shape)+abs(load)
    def kinetic(self,v):return .5*float(np.sum(v*self.multiply(v)))
    def momentum(self,v):return self.multiply(v).sum(0)


class ImplicitMPMCutting(MPMCutting):
    def trial(self,dt):
        self.reject();self._begin_operator_trial();m=self.model;old=self.material.state
        if not np.isfinite(dt) or dt<=0.:raise ValueError('Positive finite MPM timestep required')
        from .reactions import recover_reactions
        op=self.grid_operator();projected,reset=op.project_velocity(self.v)
        self.conductor.update_geometry(self.q);diffusion=advance_thermal(self.conductor,old.T,dt)
        points,_=self.geometry.contact_candidates(self.q,make_tool(self.config,self.time),gap_tolerance=self.penetration_tolerance)
        velocity=projected;heat=diffusion.heat_J.copy();fraction=self.case['thermal']['friction_heat_workpiece_fraction']
        M=sparse.kron(m.M,sparse.eye(2,format='csr'),format='csr')
        convergence=[];metric=None;tangent_builds=0;maximum_asymmetry=0.
        prepared=None;nominal_reuses=0;nominal_evaluations=0
        def nominal(F,source):
            nonlocal prepared,nominal_reuses,nominal_evaluations
            if prepared is not None and prepared.matches(self.material,F,dt,source,m.volume0):
                nominal_reuses+=1;return prepared.response
            response=self.material.trial(F,dt,source,m.volume0);nominal_evaluations+=1
            prepared=PreparedMaterialTrial.capture(self.material,response,dt,source,m.volume0)
            return response
        F=None
        for iteration in range(30):
            if F is None:_,_,_,F,_=self.position_trial(dt,velocity)
            rebuild=(metric is None or (iteration>=2 and convergence[-1]['physical_velocity_residual']>
                .5*convergence[-2]['physical_velocity_residual']) or iteration%6==0)
            if rebuild:
                prepared=None;response=self.material.trial_with_tangent(F,dt,heat,m.volume0)
                stiffness=material_stiffness(m,response.tangent)
                maximum_asymmetry=max(maximum_asymmetry,float(sparse.linalg.norm(stiffness-stiffness.T))/
                    max(float(sparse.linalg.norm(stiffness)),np.finfo(float).tiny))
                metric=GridNewtonMetric(op,M+dt*dt*.5*(stiffness+stiffness.T));tangent_builds+=1
            else:response=nominal(F,heat)
            force=m.force_from_piola(response.first_piola)
            residual=m.multiply(velocity-self.v)-dt*force
            free=velocity-metric.solve(residual,self.fixed)
            candidate,contact=solve_contact(metric,self.q,points,free,self.tool_velocity,dt,self.fixed,
                mu=self.case['thermal']['friction_coefficient'])
            new_heat,_=scatter_heat(diffusion.heat_J,contact['events'],fraction,contact['friction_heat'])
            _,_,_,Fnew,_=self.position_trial(dt,candidate)
            final=nominal(Fnew,new_heat);final_force=m.force_from_piola(final.first_piola)
            physical_residual=m.multiply(candidate-self.v)-dt*final_force-contact['nodal_contact_impulse']
            error_velocity=op.solve(physical_residual)
            velocity_residual=float(np.max(abs(error_velocity),initial=0.))
            change=float(np.max(abs(candidate-velocity),initial=0.))
            convergence.append(dict(iteration=iteration+1,physical_velocity_residual=velocity_residual,
                velocity_change=change,heat_change_J=float(np.max(abs(new_heat-heat),initial=0.)),
                contact_solver_resolution=contact['resolution'],contact_PGS_sweeps=contact['iterations']))
            velocity=candidate;heat=new_heat;F=Fnew
            if velocity_residual<=1e-9 and change<=1e-8:
                scale=np.asarray(abs(m.M)@(abs(candidate)+abs(self.v)))+abs(dt*final_force)+abs(contact['nodal_contact_impulse'])
                equilibrium_error=m.multiply(error_velocity)
                # Separate the admitted finite Newton residual from the
                # dual constraint subspace. It is reported as error, never
                # absorbed into support or used to change the state.
                support,complement,reaction_record=recover_reactions(op,physical_residual-equilibrium_error,scale)
                contact['support_reaction']=np.zeros_like(support)
                delta=velocity-self.v;contact['contact_projection_loss']=0.
                contact['energy_identity']=(m.kinetic(velocity)-m.kinetic(self.v)-dt*np.sum(velocity*final_force)
                    -contact['tool_work']+contact['friction_heat']+contact['gap_work']+m.kinetic(delta)-contact['friction_roundoff_defect'])
                result=self.finish_trial(dt,velocity,contact,diffusion,support,prepared_material=prepared)
                result.diagnostics.update(integrator='MPM_backward_euler_shared_domain_v1',newton=convergence,
                    contact_solver_resolution=contact['resolution'],
                    newton_tangent_builds=tangent_builds,tangent_max_relative_asymmetry=maximum_asymmetry,
                    nominal_material_evaluations=nominal_evaluations,nominal_material_reuses=nominal_reuses+1,
                    mechanical_nonlinear_velocity_residual=velocity_residual,
                    backward_euler_kinetic_dissipation_J=m.kinetic(delta),mpm=op.diagnostics,mpm_reset=reset,
                    mpm_reaction=reaction_record,mpm_complement_impulse_Ns=complement.sum(0),
                    equilibrium_error_impulse_Ns=equilibrium_error.sum(0),
                    equilibrium_error_angular_impulse_Nms=float(np.sum(
                        self.q[:,0]*equilibrium_error[:,1]-self.q[:,1]*equilibrium_error[:,0])),
                    equilibrium_error_work_J=float(np.sum(velocity*equilibrium_error)))
                result.diagnostics['mechanical_integration_defect_J']+=reset['reset_kinetic_loss_J']
                return result
            if iteration>=5 and change<=1e-8:
                recent=min(row['physical_velocity_residual'] for row in convergence[-3:])
                previous=min(row['physical_velocity_residual'] for row in convergence[-6:-3])
                if recent>.8*previous:
                    self.reject();raise ValueError('MPM backward Euler residual stagnated above unchanged tolerance: '+str(convergence[-1]))
        self.reject();raise ValueError('MPM backward Euler Newton failed: '+str(convergence[-1]))
