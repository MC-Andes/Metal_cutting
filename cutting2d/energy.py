"""Experimental APIC dynamics in particle energy coordinates, without Gram inversion.

This opt-in path evaluates the same MLS equations through an orthogonal
projection of the unsquared energy map. Dense SVD currently limits case size.
"""
from __future__ import annotations
import hashlib,time
import numpy as np
from scipy import sparse


class ProjectorFailure(ValueError):pass


class EnergyProjector:
 def __init__(self,stencil,mass,rtol=1e-10):
  start=time.perf_counter();mass=np.asarray(mass,float)
  if mass.shape!=(len(stencil.positions),) or not np.isfinite(mass).all() or np.any(mass<=0):
   raise ValueError('positive finite particle mass required')
  if not np.isfinite(rtol) or not 0<rtol<1:raise ValueError('rtol must be in (0,1)')
  self.stencil=stencil;self.particle_mass=mass.copy();self.root_mass=np.sqrt(mass);self.particles=len(mass)
  self.nodes=len(stencil.nodes);self.shape=(3*self.particles,2);self.rtol=float(rtol)
  if self.nodes>2048:raise ProjectorFailure('Dense prototype limited to 2048 grid nodes; no sparse projector has been accredited')
  if (np.max(abs(stencil.weights.sum(1)-1),initial=0)>2e-12 or
      np.max(abs(np.einsum('pi,pia->pa',stencil.weights,stencil.offsets)),initial=0)>2e-12*stencil.h or
      not np.allclose(np.einsum('pi,pia,pib->pab',stencil.weights,stencil.offsets,stencil.offsets),np.eye(2)*stencil.h**2/4,rtol=2e-12,atol=2e-12*stencil.h**2)):
   raise ValueError('Complete B2 moment identities required')
  rows=np.repeat(np.arange(self.particles),9);cols=stencil.indices.ravel()
  def matrix(values):return sparse.coo_matrix((values.ravel(),(rows,cols)),shape=(self.particles,self.nodes)).tocsr()
  W=matrix(stencil.weights);root=sparse.diags(self.root_mass)
  self.H=sparse.vstack([root@W]+[root@matrix(stencil.weights*stencil.offsets[:,:,a]*2/stencil.h) for a in range(2)]).tocsr()
  self.grid_mass=np.asarray(W.T@mass).ravel();self.mass=self.grid_mass;self.active=self.grid_mass>0
  self.Z=(self.H[:,self.active]@sparse.diags(1/np.sqrt(self.grid_mass[self.active]))).tocsr()
  self._cache={}
  self.diagnostics={'prototype':'particle_energy_svd_projector','particles':self.particles,'nodes':self.nodes,
   'active_nodes':int(self.active.sum()),'no_gram_inverse':True,'no_mass_regularization':True,
   'coordinate_order':'[sqrt(m) v; sqrt(m) h/2 A[:,0]; sqrt(m) h/2 A[:,1]], each block P x2',
   'factors':[],'applications':0,'maximum_orthogonality_error':0.}
  self._factor(np.ones(self.nodes,bool))
  self.diagnostics['construction_seconds']=time.perf_counter()-start

 def _factor(self,free):
  active_free=np.asarray(free,bool)[self.active];key=active_free.tobytes()
  if key in self._cache:return self._cache[key]
  start=time.perf_counter();Z=self.Z[:,active_free].toarray()
  if not Z.shape[1]:
   U=np.empty((self.shape[0],0));record={'rank':0,'columns':0,'sigma_min_retained':None,'sigma_max':None,
    'singular_value_cutoff':None,'svd_reconstruction_relative_residual':0.,'subspace_sensitivity_indicator':None}
  else:
   U,s,Vh=np.linalg.svd(Z,full_matrices=False)
   cutoff=max(Z.shape)*np.finfo(float).eps*s[0];rank=int(np.sum(s>cutoff))
   reconstruction=float(np.linalg.norm(Z-(U*s)@Vh,ord='fro')/np.linalg.norm(Z,ord='fro'))
   discarded=float(s[rank]) if rank<len(s) else 0.
   gap=float(s[rank-1]-discarded) if rank else 0.
   # Diagnostic, not a rigorous error certificate: actual reconstruction error
   # divided by the retained/discarded singular gap indicates sensitivity.
   indicator=reconstruction*float(np.linalg.norm(Z,ord='fro'))/gap if gap>0 else None
   record={'rank':rank,'columns':Z.shape[1],'sigma_min_retained':float(s[rank-1]) if rank else None,
    'sigma_max':float(s[0]),'singular_value_cutoff':float(cutoff),'sigma_first_discarded':discarded,
    'svd_reconstruction_relative_residual':reconstruction,'subspace_sensitivity_indicator':indicator,
    'gram_condition_on_range':float((s[0]/s[rank-1])**2) if rank else None}
   U=U[:,:rank]
  orthogonal=float(np.linalg.norm(U.T@U-np.eye(U.shape[1]),ord='fro'))
  if orthogonal>self.rtol:raise ProjectorFailure(f'SVD range vectors fail orthogonality: {orthogonal:g}')
  # Keep the original orthogonality calculation and acceptance above. Then
  # store the unchanged vectors contiguously by column for repeated GEMV.
  # The copy also releases unretained vectors in a rank-deficient factor.
  U=np.asfortranarray(U)
  self.diagnostics['maximum_orthogonality_error']=max(self.diagnostics['maximum_orthogonality_error'],orthogonal)
  record.update(free_sha256=hashlib.sha256(key).hexdigest(),seconds=time.perf_counter()-start,orthogonality_frobenius=orthogonal)
  factor={'U':U,'diagnostics':record};self._cache[key]=factor;self.diagnostics['factors'].append(record)
  return factor

 def _energy(self,y):
  y=np.asarray(y,float)
  if y.shape!=self.shape or not np.isfinite(y).all():raise ValueError(f'Energy array must be finite {self.shape}')
  return y

 def apply(self,y,fixed=None):
  """P y (or P_free y): Euclidean orthogonal projection, without division by s."""
  y=self._energy(y)
  if fixed is None:fixed=np.zeros((self.nodes,2),bool)
  fixed=np.asarray(fixed)
  if fixed.shape!=(self.nodes,2) or fixed.dtype!=bool:raise ValueError('fixed requires bool (nodes,2)')
  result=np.empty_like(y)
  for a in range(2):
   U=self._factor(~fixed[:,a])['U'];result[:,a]=U@(U.T@y[:,a])
  self.diagnostics['applications']+=1
  return result

 def pack(self,velocity,affine):
  velocity=np.asarray(velocity,float);affine=np.asarray(affine,float)
  if velocity.shape!=(self.particles,2) or affine.shape!=(self.particles,2,2):raise ValueError('particle velocity/affine shape mismatch')
  y=np.vstack([self.root_mass[:,None]*velocity]+[self.root_mass[:,None]*self.stencil.h/2*affine[:,:,a] for a in range(2)])
  return self._energy(y)

 def particle_state(self,y):
  y=self._energy(y);P=self.particles
  velocity=y[:P]/self.root_mass[:,None]
  affine=np.stack([y[P:2*P]/self.root_mass[:,None]*2/self.stencil.h,
                   y[2*P:]/self.root_mass[:,None]*2/self.stencil.h],axis=2)
  return velocity,affine

 def kinetic(self,y):
  y=self._energy(y);return .5*float(np.sum(y*y))

 def force_density(self,current_volume,stress,acceleration=(0.,0.)):
  """Energy-coordinate force g with H.T g = MLS internal plus body force."""
  stress=np.asarray(stress,float);volume=np.asarray(current_volume,float);acceleration=np.asarray(acceleration,float)
  if stress.shape not in ((self.particles,2,2),(self.particles,3,3)) or volume.shape!=(self.particles,) or acceleration.shape!=(2,):
   raise ValueError('Force data shapes invalid')
  if np.any(volume<=0) or not all(np.isfinite(a).all() for a in (stress,volume,acceleration)):
   raise ValueError('Positive volumes and finite force data required')
  return np.vstack([self.root_mass[:,None]*acceleration]+[
   -2/self.stencil.h*volume[:,None]/self.root_mass[:,None]*stress[:,:2,a] for a in range(2)])

 def nodal_impulse(self,dy):
  """Stable H.T dy; no recovery of nodal velocity coefficients."""
  return np.asarray(self.H.T@self._energy(dy))

 def momentum(self,y):
  """Linear momentum and angular momentum of the APIC representation."""
  velocity,A=self.particle_state(y);x=self.stencil.positions
  linear=np.sum(self.particle_mass[:,None]*velocity,axis=0)
  angular=float(np.sum(self.particle_mass*(x[:,0]*velocity[:,1]-x[:,1]*velocity[:,0]+self.stencil.h**2/4*(A[:,1,0]-A[:,0,1]))))
  return linear,angular
