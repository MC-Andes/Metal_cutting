"""Backward Euler candidate with exact nonlinear residual and modified Newton.

The symmetrized finite-difference tangent is an iteration matrix only. The
accepted equation uses the unchanged nominal finite-JC first Piola stress.
Its dissipation is reported as numerical energy error, never thermal input.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from cutting2d.cases import make_tool
from .solver import DirectCutting,PreparedMaterialTrial
from .contact import solve_contact
from .coupling import scatter_heat
from .thermal_subcycling import advance as advance_thermal


def material_stiffness(model,tangent):
    G=model.G_ref;V=model.volume0
    local=np.einsum('q,qaj,qijkl,qbl->qaibk',V,G,tangent,G,optimize=True)
    dofs=(2*model.ids[:,:,None]+np.arange(2)).reshape(-1,8)
    rows=np.repeat(dofs,8,axis=1).ravel();columns=np.tile(dofs,(1,8)).ravel()
    return sparse.coo_matrix((local.reshape(-1), (rows,columns)),shape=(2*model.nodes,2*model.nodes)).tocsr()


class NewtonMetric:
    """Iteration mobility, explicitly distinct from physical inertia."""
    def __init__(self,A,fixed):
        self.A=A;self.fixed=fixed;self.free=np.flatnonzero(~fixed.ravel());self.shape=fixed.shape
        self.factor=splu(A[self.free][:,self.free].tocsc())
    def solve(self,force,fixed):
        if not np.array_equal(fixed,self.fixed):raise ValueError('Newton support mismatch')
        answer=np.zeros(force.size);answer[self.free]=self.factor.solve(force.ravel()[self.free])
        residual=(self.A@answer-force.ravel())[self.free]
        if np.linalg.norm(residual)>1e-10*max(np.linalg.norm(force.ravel()[self.free]),1e-30):
            raise ValueError('Newton linear solve failed its residual')
        return answer.reshape(self.shape)
    def multiply(self,v):return (self.A@v.ravel()).reshape(self.shape)
    def kinetic(self,v):return .5*float(v.ravel()@(self.A@v.ravel()))
    def momentum(self,v):return self.multiply(v).sum(0)


class ImplicitCutting(DirectCutting):
    def trial(self,dt):
        self.reject();m=self.model;old=self.material.state
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite timestep required')
        self.conductor.update_geometry(self.q);diffusion=advance_thermal(self.conductor,old.T,dt)
        points,_=self.geometry.contact_candidates(self.q,make_tool(self.config,self.time),gap_tolerance=self.penetration_tolerance)
        velocity=self.v.copy();heat=diffusion.heat_J.copy();fraction=self.case['thermal']['friction_heat_workpiece_fraction']
        M=sparse.kron(m.M,sparse.eye(2,format='csr'),format='csr')
        convergence=[];contact=None;maximum_asymmetry=0.;metric=None;tangent_builds=0
        prepared=None;nominal_reuses=0;nominal_evaluations=0
        def nominal(F,heat):
            nonlocal prepared,nominal_reuses,nominal_evaluations
            if prepared is not None and prepared.matches(self.material,F,dt,heat,m.volume0):
                nominal_reuses+=1
                return prepared.response
            response=self.material.trial(F,dt,heat,m.volume0);nominal_evaluations+=1
            prepared=PreparedMaterialTrial.capture(self.material,response,dt,heat,m.volume0)
            return response
        for iteration in range(30):
            _,_,_,F,_=self.position_trial(dt,velocity)
            rebuild=(metric is None or (iteration>=2 and convergence[-1]['physical_velocity_residual']>
                .5*convergence[-2]['physical_velocity_residual']) or iteration%6==0)
            if rebuild:
                prepared=None
                response=self.material.trial_with_tangent(F,dt,heat,m.volume0)
                stiffness=material_stiffness(m,response.tangent)
                maximum_asymmetry=max(maximum_asymmetry,float(sparse.linalg.norm(stiffness-stiffness.T))/max(float(sparse.linalg.norm(stiffness)),1e-30))
                A=M+dt*dt*.5*(stiffness+stiffness.T)
                metric=NewtonMetric(A,self.fixed);tangent_builds+=1
            else:response=nominal(F,heat)
            force=m.force_from_piola(response.first_piola)
            residual=m.multiply(velocity-self.v)-dt*force
            free=velocity-metric.solve(residual,self.fixed)
            candidate,linear_contact=solve_contact(metric,self.q,points,free,self.tool_velocity,dt,self.fixed,
                mu=self.case['thermal']['friction_coefficient'])
            # Friction is recomputed from this candidate's actual endpoint
            # relative velocity and physical impulse, then enters the same JC
            # trial as conduction. No Newton iteration commits a history.
            new_heat,_=scatter_heat(diffusion.heat_J,linear_contact['events'],fraction,linear_contact['friction_heat'])
            _,_,_,Fnew,_=self.position_trial(dt,candidate)
            final=nominal(Fnew,new_heat)
            final_force=m.force_from_piola(final.first_piola)
            physical_residual=m.multiply(candidate-self.v)-dt*final_force-linear_contact['nodal_contact_impulse']
            free_residual=physical_residual.copy();free_residual[self.fixed]=0.
            # M^-1 norm converts the exact physical momentum equation to a
            # velocity residual without accepting relative cancellation.
            velocity_residual=float(np.max(abs(m.solve(free_residual,self.fixed)),initial=0.))
            change=float(np.max(abs(candidate-velocity),initial=0.))
            convergence.append(dict(iteration=iteration+1,physical_velocity_residual=velocity_residual,
                velocity_change=change,heat_change_J=float(np.max(abs(new_heat-heat),initial=0.))))
            velocity=candidate;heat=new_heat;contact=linear_contact
            if velocity_residual<=1e-9 and change<=1e-8:
                # Only the fixed rows are external support. Free-row residuals
                # remain an error and cannot be hidden in the reaction ledger.
                support=physical_residual.copy();support[~self.fixed]=0.
                angular=lambda f:float(np.sum(self.q[:,0]*f[:,1]-self.q[:,1]*f[:,0]))
                contact['support_reaction']=np.zeros_like(support)
                contact['support_impulse']=np.zeros(2)
                contact['support_angular_impulse']=0.
                # Report the actual nonlinear BE kinetic identity. The Newton
                # mobility's quadratic energy is not a physical energy.
                delta=velocity-self.v
                contact['contact_projection_loss']=0.
                contact['energy_identity']=(m.kinetic(velocity)-m.kinetic(self.v)-dt*np.sum(velocity*final_force)
                    -contact['tool_work']+contact['friction_heat']+contact['gap_work']+m.kinetic(delta)-contact['friction_roundoff_defect'])
                result=self.finish_trial(dt,velocity,contact,diffusion,support,prepared_material=prepared)
                result.diagnostics.update(integrator='backward_euler_v1',newton=convergence,
                    newton_tangent_builds=tangent_builds,
                    nominal_material_evaluations=nominal_evaluations,nominal_material_reuses=nominal_reuses+1,
                    tangent_max_relative_asymmetry=maximum_asymmetry,
                    mechanical_nonlinear_velocity_residual=velocity_residual,
                    backward_euler_kinetic_dissipation_J=m.kinetic(delta),
                    support_torque_at_old_coordinates_Nms=angular(support))
                return result
            if iteration>=5 and change<=1e-8:
                recent=min(row['physical_velocity_residual'] for row in convergence[-3:])
                previous=min(row['physical_velocity_residual'] for row in convergence[-6:-3])
                if recent>.8*previous:
                    self.reject()
                    raise ValueError('Backward Euler residual stagnated above the unchanged tolerance: '+str(convergence[-1]))
        self.reject();raise ValueError('Backward Euler Newton iteration did not converge: '+str(convergence[-1]))
