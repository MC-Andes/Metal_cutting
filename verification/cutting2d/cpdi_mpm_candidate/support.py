"""Exact stationary bottom trace in the background velocity space.

For a flat material bottom, C = Wx tensor wy. If the sampled horizontal
B2 map Wx is injective, ker(C) is obtained by eliminating one vertical
combination independently at each horizontal grid address. No grid band is
clamped. The one-dimensional rank check must pass before this reduction is
used; it makes no claim about the rank of the interior mass operator.
"""
import numpy as np
from scipy import sparse
from cutting2d.transfer import build_stencil


class BottomTrace:
    def __init__(self, q_bottom, h, origin=None):
        q=np.asarray(q_bottom,float)
        if q.ndim!=2 or q.shape[1]!=2 or len(q)<3 or not np.isfinite(q).all():
            raise ValueError('Finite sampled bottom trace required')
        if np.any(q[:,1]!=q[0,1]) or len(np.unique(q[:,0]))!=len(q):
            raise ValueError('Bottom trace must be horizontal with distinct material corners')
        st=build_stencil(q,h,origin);self.q=q.copy();self.h=st.h;self.origin=st.origin.copy()
        lattice=st.lattice_nodes;xs=np.unique(lattice[:,0]);ys=np.unique(lattice[:,1])
        if len(ys)!=3:raise ValueError('Bottom B2 trace must use three lattice rows')
        C=sparse.coo_matrix((st.weights.ravel(),(np.repeat(np.arange(len(q)),9),st.indices.ravel())),shape=(len(q),len(lattice))).tocsr()
        # Summation along each tensor direction recovers its partitioned map.
        Wx=np.column_stack([np.asarray(C[:,lattice[:,0]==ix].sum(1)).ravel() for ix in xs])
        wy=np.array([float(C[0,lattice[:,1]==iy].sum()) for iy in ys])
        active_x=np.any(Wx!=0.,axis=0);Wx=Wx[:,active_x];xs=xs[active_x]
        norms=np.linalg.norm(Wx,axis=0)
        singular=np.linalg.svd(Wx/norms,compute_uv=False)
        threshold=max(Wx.shape)*np.finfo(float).eps*singular[0]
        if Wx.shape[0]<Wx.shape[1] or singular[-1]<=32*threshold:
            raise ValueError('Horizontal bottom trace rank is not certified; local elimination would overconstrain')
        reconstruction=np.zeros(C.shape)
        lookup={tuple(a):i for i,a in enumerate(lattice)}
        for col,ix in enumerate(xs):
            for j,iy in enumerate(ys):reconstruction[:,lookup[(ix,iy)]]=Wx[:,col]*wy[j]
        tensor_error=float(np.max(abs(C.toarray()-reconstruction),initial=0.))
        if tensor_error>64*np.finfo(float).eps:raise ValueError('Bottom trace tensor factorization failed')
        # Householder maps the normalized constraint to the first coordinate.
        normal=wy/np.linalg.norm(wy);v=normal.copy();v[0]+=1.
        Q=np.eye(3)-2*np.outer(v,v)/np.dot(v,v);null=Q[:,1:]
        self.xs=xs;self.ys=ys;self.wy=wy;self.local_null=null
        self.diagnostics=dict(representation='local_tensor_bottom_nullspace_v1',
            material_constraints=len(q),independent_constraints=len(xs),
            horizontal_scaled_sigma_min=float(singular[-1]),horizontal_rank_threshold=float(threshold),
            tensor_factorization_error=tensor_error,no_background_band_clamping=True,
            rank_scope='One-dimensional stationary bottom trace only; not interior energy map')

    def reduction(self, transfer, bottom, coordinate_scale=None):
        bottom=np.asarray(bottom)
        if (bottom.dtype.kind not in 'iu' or bottom.ndim!=1 or
            not np.array_equal(transfer.q[bottom],self.q) or transfer.stencil.h!=self.h or
            not np.array_equal(transfer.stencil.origin,self.origin)):
            raise ValueError('Bottom support cache differs from current material trace or grid')
        nodes=transfer.stencil.lattice_nodes;lookup={tuple(a):i for i,a in enumerate(nodes)}
        scale=np.ones(len(nodes)) if coordinate_scale is None else np.asarray(coordinate_scale,float)
        if scale.shape!=(len(nodes),) or not np.isfinite(scale).all() or np.any(scale<=0.):
            raise ValueError('Positive invertible background coordinate scale required')
        rows=[];cols=[];data=[];used=set();column=0
        for ix in self.xs:
            try:ids=[lookup[(ix,iy)] for iy in self.ys]
            except KeyError as error:raise ValueError('Background grid lost a bottom support slot') from error
            used.update(ids)
            normal=self.wy*scale[ids];normal/=np.linalg.norm(normal)
            vector=normal.copy();vector[0]+=1.
            local=(np.eye(3)-2*np.outer(vector,vector)/np.dot(vector,vector))[:,1:]
            for j in range(2):
                rows.extend(ids);cols.extend([column]*3);data.extend(scale[ids]*local[:,j]);column+=1
        for i in range(len(nodes)):
            if i not in used:rows.append(i);cols.append(column);data.append(scale[i]);column+=1
        L=sparse.coo_matrix((data,(rows,cols)),shape=(len(nodes),column)).tocsr()
        trace=(transfer.R[bottom]@L).tocsr()
        error=float(np.max(abs(trace.data),initial=0.))
        trace_scale=float(np.max(abs((transfer.R[bottom]@sparse.diags(scale)).data),initial=0.))
        relative=error/max(trace_scale,np.finfo(float).tiny)
        if relative>128*np.finfo(float).eps:raise ValueError('Reduced background field violates stationary material trace')
        B=(transfer.R@L).tocsr();B.sort_indices()
        # These rows are symbolic zeros of the certified constraint. Store
        # them as such, rather than evaluating a cancelling sum each step.
        # No accepted velocity, geometry or displacement is repaired here.
        for row in bottom:B.data[B.indptr[row]:B.indptr[row+1]]=0.
        B.eliminate_zeros()
        return L,B,dict(self.diagnostics,raw_bottom_null_error=error,
            scaled_bottom_null_relative=relative,reduced_grid_coordinates=column)
