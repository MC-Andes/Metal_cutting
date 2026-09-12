"""Transactional explicit reference for direct material Q4 cutting.

This candidate is not an accepted developed-cutting backend. No source,
velocity, geometry or mass is modified to manufacture acceptance.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from cutting2d.cases import make_tool
from .mechanics import DirectQ4,add_compensated
from .constitutive import ConstitutiveBatch
from .thermal import DirectQ4Thermal
from .thermal_subcycling import advance as advance_thermal
from .contact import solve_contact
from .coupling import scatter_heat


def rectangular_mesh(length,height,h):
    nx,ny=[int(round(value/h)) for value in (length,height)]
    if min(nx,ny)<1 or not np.allclose([nx*h,ny*h],[length,height],rtol=1e-12,atol=0.):
        raise ValueError('The specified mesh size must divide both physical dimensions')
    X=np.array([[x,y] for x in np.linspace(0,length,nx+1) for y in np.linspace(0,height,ny+1)])
    cells=np.array([[i*(ny+1)+j,(i+1)*(ny+1)+j,(i+1)*(ny+1)+j+1,i*(ny+1)+j+1]
                    for i in range(nx) for j in range(ny)])
    return X,cells


def internal_energy(response,volume):
    return float(sum(np.dot(response.energy_densities[key],volume)
                     for key in ('elastic_energy','stored_energy','thermal_energy')))


@dataclass(frozen=True)
class Trial:
    q: np.ndarray
    v: np.ndarray
    displacement_hi: np.ndarray
    displacement_lo: np.ndarray
    material: object
    time: float
    clock_compensation: float
    diagnostics: dict
    owner: object


@dataclass(frozen=True)
class PreparedMaterialTrial:
    """One pending nominal response tied to exact inputs and committed history."""
    response: object
    history: object
    dt: float
    heat_J: np.ndarray
    volume0: np.ndarray

    @classmethod
    def capture(cls,batch,response,dt,heat_J,volume0):
        heat=np.array(heat_J,copy=True);volume=np.array(volume0,copy=True)
        heat.setflags(write=False);volume.setflags(write=False)
        return cls(response,batch.state,float(dt),heat,volume)

    def matches(self,batch,F,dt,heat_J,volume0):
        def same_bits(a,b):
            a=np.asarray(a);b=np.asarray(b)
            return (a.dtype==b.dtype==np.dtype(np.float64) and a.shape==b.shape
                    and np.array_equal(a.view(np.uint64),b.view(np.uint64)))
        return (self.response._batch is batch and batch.state is self.history
                and batch._pending is self.response and float(dt)==self.dt
                and same_bits(F,self.response.F) and same_bits(heat_J,self.heat_J)
                and same_bits(volume0,self.volume0))


class DirectCutting:
    def __init__(self,case,*,h=None,threads=1,geometry_backend='cached',mesh_profile='uniform',constitutive_options=None):
        self.case=case;g=case['geometry'];p=case['material']['properties'];self.h=float(h or case['solver']['h'])
        self.config=SimpleNamespace(geometry=SimpleNamespace(**g))
        if mesh_profile=='uniform':
            X,cells=rectangular_mesh(g['length'],g['height'],self.h)
            self.mesh_diagnostics=dict(profile='uniform',h_m=self.h,nodes=len(X),cells=len(cells))
        elif mesh_profile=='graded':
            from .mesh import tensor_graded_mesh
            X,cells,self.mesh_diagnostics=tensor_graded_mesh(g['length'],g['height'],self.h)
            self.mesh_diagnostics['profile']='graded'
        else:raise ValueError('Unknown mesh profile')
        self.model=DirectQ4(X,cells,p['density_kg_m3'],g['width'])
        if geometry_backend=='cached':
            from .geometry import MeshGeometry
            self.geometry=MeshGeometry(X,cells)
        elif geometry_backend=='historical':
            from verification.cutting2d.q4_contact_candidate.geometry import contact_candidates
            class Historical:
                def contact_candidates(_,q,tool,gap_tolerance):return contact_candidates(q,cells,tool,gap_tolerance)
                def validate(_,q,tool,penetration_tolerance):return contact_candidates(q,cells,tool,penetration_tolerance)[1]
            self.geometry=Historical()
        else:raise ValueError('Unknown explicitly selected geometry backend')
        self.geometry_backend=geometry_backend
        options={'relative_tolerance':1e-13} if constitutive_options is None else dict(constitutive_options)
        self.material=ConstitutiveBatch(case['material'],self.model.point_count,p['initial_temperature_k'],threads=threads,options=options)
        self.conductor=DirectQ4Thermal(X,cells,g['width'],p['density_kg_m3'],p['specific_heat_j_kg_k'],
            p['thermal_conductivity_w_m_k'] if case['thermal']['conduction'] else 0.,
            mass_gp=self.model.mass,reference_volume_gp=self.model.volume0)
        bottom=X[:,1]==0.;self.fixed=np.column_stack((bottom & case['boundary']['fix_bottom_x'],bottom & case['boundary']['fix_bottom_y']))
        self.q=X.copy();self.v=np.zeros_like(X);self.time=0.;self.clock_compensation=0.;self.step=0;self.pending=None
        self.displacement_hi=np.zeros_like(X);self.displacement_lo=np.zeros_like(X)
        self.tool_velocity=np.array([g['tool_speed'],0.]);self.penetration_tolerance=min(
            case['acceptance']['penetration_absolute_m'],self.h*case['acceptance']['penetration_over_h'])
        self.initial_energy=internal_energy(self.material.state,self.model.volume0)
        self.geometry.validate(self.q,make_tool(self.config,0.),penetration_tolerance=self.penetration_tolerance)

    def trial(self,dt):
        self.reject();m=self.model;old=self.material.state
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite trial timestep required')
        self.conductor.update_geometry(self.q)
        diffusion=advance_thermal(self.conductor,old.T,dt)
        force=m.force_from_piola(old.first_piola)
        free=self.v+dt*m.solve(force,self.fixed)
        reaction_kick=m.multiply(free-self.v)-dt*force
        points,geometry0=self.geometry.contact_candidates(self.q,make_tool(self.config,self.time),gap_tolerance=self.penetration_tolerance)
        velocity,contact=solve_contact(m,self.q,points,free,self.tool_velocity,dt,self.fixed,
            mu=self.case['thermal']['friction_coefficient'])
        return self.finish_trial(dt,velocity,contact,diffusion,reaction_kick)

    def finish_trial(self,dt,velocity,contact,diffusion,reaction_kick,*,prepared_material=None):
        m=self.model;old=self.material.state
        q,hi,lo,F,corners=self.position_trial(dt,velocity)
        adjusted=dt-self.clock_compensation;clock=self.time+adjusted;compensation=(clock-self.time)-adjusted
        if clock<=self.time:raise ValueError('Trial cannot advance represented physical clock')
        geometry1=self.geometry.validate(q,make_tool(self.config,clock),penetration_tolerance=self.penetration_tolerance)
        geometry1['discarded_separation_cell_count']=len(geometry1.get('discarded_separation_cells',()))
        fraction=self.case['thermal']['friction_heat_workpiece_fraction']
        heat,heat_accounting=scatter_heat(diffusion.heat_J,contact['events'],fraction,contact['friction_heat'])
        if prepared_material is None:
            response=self.material.trial(F,dt,heat,m.volume0)
        elif isinstance(prepared_material,PreparedMaterialTrial) and prepared_material.matches(self.material,F,dt,heat,m.volume0):
            response=prepared_material.response
        else:
            raise ValueError('Prepared material response differs from final inputs, history or pending owner')
        # No local thermal source is applied twice: diffusion and interface heat
        # enter once, with plastic heating computed exclusively by finite JC.
        if np.any(response.T<=0) or np.any(response.T>=self.case['material']['properties']['melt_temperature_k']):
            raise ValueError('Constitutive temperature is outside the solid domain')
        corner_temperature=self.conductor.corner_temperatures(response.T)
        if np.min(corner_temperature)<=0:
            raise ValueError('Q4 thermal reconstruction is nonpositive after physical sources')
        if np.any(q[self.fixed]!=m.X[self.fixed]) or np.any(velocity[self.fixed]!=0):
            raise ValueError('Stationary material support moved')
        if np.any(hi[self.fixed]!=0) or np.any(lo[self.fixed]!=0):
            raise ValueError('Canonical material support displacement moved')
        K0=m.kinetic(self.v);K1=m.kinetic(velocity)
        U0=internal_energy(old,m.volume0);U1=internal_energy(response,m.volume0)
        support=reaction_kick+contact['support_reaction'];tool_heat=(1-fraction)*contact['friction_heat']
        total_defect=(K1-K0)+(U1-U0)-contact['tool_work']+tool_heat
        angular=lambda f:float(np.sum(self.q[:,0]*f[:,1]-self.q[:,1]*f[:,0]))
        support_impulse=support.sum(0);momentum_defect=m.momentum(velocity)-m.momentum(self.v)-contact['tool_impulse']-support_impulse
        angular_defect=m.angular_momentum(q,velocity)-m.angular_momentum(self.q,self.v)-contact['tool_angular_impulse']-angular(support)
        scale=max(float(np.linalg.norm(contact['tool_impulse'])),float(np.linalg.norm(support_impulse)),1e-14)
        if np.linalg.norm(momentum_defect)>max(1e-14,self.case['acceptance']['momentum_relative']*scale):
            raise ValueError('Physical momentum balance failed')
        energies=response.energies_J(m.volume0)
        diagnostics=dict(integrator='symplectic_euler_v1',step=self.step+1,time_s=clock,dt_s=dt,tool_advance_m=clock*self.tool_velocity[0],
            kinetic_J=K1,internal_J=U1,total_energy_J=K1+U1,energy_defect_J=total_defect,
            tool_work_J=contact['tool_work'],tool_heat_J=tool_heat,friction_heat_J=contact['friction_heat'],
            conduction_net_J=float(diffusion.heat_J.sum()),external_heat_J=float(heat.sum()),
            tool_impulse_Ns=contact['tool_impulse'],support_impulse_Ns=support_impulse,
            momentum_Ns=m.momentum(velocity),momentum_defect_Ns=momentum_defect,
            angular_defect_Nms=angular_defect,contact_energy_identity_J=contact['energy_identity'],
            contact_projection_loss_J=contact['contact_projection_loss'],gap_work_J=contact['gap_work'],
            friction_roundoff_J=contact['friction_roundoff_defect'],
            mechanical_integration_defect_J=total_defect+contact['contact_projection_loss']+contact['gap_work']-contact['friction_roundoff_defect'],
            min_J=float(response.J.min()),min_corner_J_ratio=float(np.min(corners/m.reference_corner_jacobians)),
            alpha_max=float(response.alpha.max()),T_min_K=float(response.T.min()),T_max_K=float(response.T.max()),
            T_reconstructed_corner_min_K=float(np.min(corner_temperature)),
            max_speed_m_s=float(np.max(np.linalg.norm(velocity,axis=1))),
            energy_components_J={key:float(value.sum()) for key,value in energies.items()},
            contact_events=len(contact['events']),contact_velocity_residual=contact['velocity_residual'],
            contact_constraints=contact['constraints'],contact_mass_response_rows=contact['mass_response_rows'],
            contact_iterations=contact['iterations'],geometry={key:value for key,value in geometry1.items()
                if key not in ('boundary_edges','boundary_owners','boundary_loop','corner_jacobian_minima',
                               'per_cell_separation_values','per_cell_is_lower_bound','discarded_separation_cells')},
            thermal=diffusion.diagnostics,heat_source_accounting=heat_accounting,
            position_representation='material_two_float_expansion_v1',
            geometry_roundoff_bounds={key:value for key,value in m.diagnostics.items()
                if key.startswith('maximum_displacement_') or key.startswith('maximum_display_position_')})
        self.pending=Trial(q,velocity,hi,lo,response,clock,compensation,diagnostics,self)
        return self.pending

    def position_trial(self,dt,velocity):
        hi,lo=add_compensated(self.displacement_hi,self.displacement_lo,dt*velocity)
        F,corners=self.model.deformation_displacement(hi,lo)
        q=self.model.display_positions(hi,lo)
        return q,hi,lo,F,corners

    def commit(self,trial):
        if trial is not self.pending or trial.owner is not self:raise ValueError('Only the current successful trial can commit')
        self.material.commit(trial.material);self.q=trial.q;self.v=trial.v
        self.displacement_hi=trial.displacement_hi;self.displacement_lo=trial.displacement_lo
        self.time=trial.time;self.clock_compensation=trial.clock_compensation;self.step+=1;self.pending=None
        return trial.diagnostics

    def reject(self):
        self.material.reject();self.pending=None
