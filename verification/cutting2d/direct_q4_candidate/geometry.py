"""Cached reference topology, full material geometry checks and Q4 contact.

No all-pairs dense arrays are formed. Topology is immutable; geometry is
rechecked after every distinct trial position array. Contact rows retain the
historical exact boundary construction. Tool broad phase gives conservative
separation bounds, never a penetration repair.
"""
from __future__ import annotations
import copy
import time
import numpy as np
from verification.cutting2d.q4_contact_candidate.geometry import (
    GeometryFailure, area, cross, clip_polygon, arc_parameter, support)
from verification.cutting2d.q4_mesh_validation_scaling_candidate.validator import (
    validate_mesh_sparse, unresolved_vertices, overlap_pairs)
from verification.cutting2d.q4_geometry_scaling_candidate.bounded_geometry import bounded_separations
from .mechanics import CORNER_SIGNS, shape_functions, det2, readonly


class MeshGeometry:
    def __init__(self, X, cells, *, exhaustive_overlap=False):
        if type(exhaustive_overlap) is not bool:
            raise ValueError('exhaustive_overlap must be bool')
        self.exhaustive_overlap=exhaustive_overlap
        X=np.asarray(X,float);cells=np.asarray(cells)
        mesh=validate_mesh_sparse(X,cells)
        self.X=readonly(X);self.cells=readonly(cells,np.int64);self.nodes=len(X)
        self.boundary_edges=readonly(mesh['boundary_edges'],np.int64)
        self.boundary_owners=readonly(mesh['boundary_owners'],np.int64)
        self.boundary_loop=readonly(mesh['boundary_loop'],np.int64)
        self._corner_D=shape_functions(CORNER_SIGNS)[1]
        self._gauss_D=shape_functions(CORNER_SIGNS/np.sqrt(3.))[1]
        # Convex cells with opposite orientations along a common material edge
        # occupy opposite half-planes: their interiors cannot overlap.
        edges={}
        for cell,ids in enumerate(self.cells):
            for a,b in zip(ids,np.roll(ids,-1)):
                edges.setdefault(tuple(sorted((int(a),int(b)))),[]).append(cell)
        self._edge_neighbors={tuple(sorted(owners)) for owners in edges.values() if len(owners)==2}
        self._last_q=None;self._last_mesh=None;self._last_exhaustive=None
        self.stats=dict(validations=0,cache_hits=0,contact_calls=0)
        self._mesh(X)

    def _positions(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.nodes,2) or not np.isfinite(q).all():
            raise GeometryFailure('Finite current coordinates of the fixed reference topology required')
        return q

    def _simple_boundary(self,q,tolerance):
        segments=q[self.boundary_edges]
        lo=np.nextafter(segments.min(1)-tolerance,-np.inf)
        hi=np.nextafter(segments.max(1)+tolerance,np.inf)
        for i,j in overlap_pairs(lo,hi):
            if np.intersect1d(self.boundary_edges[i],self.boundary_edges[j]).size:
                continue
            a,b=segments[i];c,d=segments[j]
            o1=cross(b-a,c-a);o2=cross(b-a,d-a)
            o3=cross(d-c,a-c);o4=cross(d-c,b-c)
            margin=tolerance*max(np.linalg.norm(b-a),np.linalg.norm(d-c))
            side1=(o1>margin and o2>margin) or (o1<-margin and o2<-margin)
            side2=(o3>margin and o4>margin) or (o3<-margin and o4<-margin)
            if not side1 and not side2:
                raise GeometryFailure('Nonadjacent material boundary segments intersect or are unresolved')

    def _mesh(self,q,exhaustive_overlap=None):
        q=self._positions(q)
        exhaustive=self.exhaustive_overlap if exhaustive_overlap is None else exhaustive_overlap
        if type(exhaustive) is not bool:
            raise ValueError('exhaustive_overlap must be bool')
        if self._last_q is not None and self._last_exhaustive==exhaustive and np.array_equal(q,self._last_q):
            self.stats['cache_hits']+=1
            return copy.deepcopy(self._last_mesh)
        started=time.perf_counter();self.stats['validations']+=1
        extent=max(float(np.max(np.ptp(q,axis=0))),np.finfo(float).tiny)
        position_tol=128*np.finfo(float).eps*max(extent,float(np.max(abs(q))))
        if unresolved_vertices(q,position_tol):
            raise GeometryFailure('Distinct vertex IDs have coincident or unresolved positions')
        polygons=q[self.cells];centered=polygons-polygons.mean(1)[:,None,:]
        corners=det2(np.einsum('eai,gaj->egij',centered,self._corner_D))
        lengths=np.linalg.norm(np.roll(polygons,-1,axis=1)-polygons,axis=2).max(1)
        minima=corners.min(1);threshold=128*np.finfo(float).eps*lengths*lengths
        invalid=np.flatnonzero(minima<=threshold)
        if len(invalid):
            e=int(invalid[0]);raise GeometryFailure(f'Q4 {e} has nonpositive or unresolved corner Jacobian: {minima[e]:.17g}')
        gauss=det2(np.einsum('eai,gaj->egij',centered,self._gauss_D))
        areas=.5*np.sum(cross(centered,np.roll(centered,-1,axis=1)),axis=1)
        overlap_tol=128*np.finfo(float).eps*extent*extent
        self._simple_boundary(q,position_tol)
        # Every positive convex cell contributes winding number 0 or 1. Its
        # oriented edge chain cancels every shared edge exactly, leaving the
        # one simple CCW outer boundary: summed cell winding is therefore 0
        # outside and 1 inside. Two cell interiors cannot cover the same point.
        # The complete reference disk/connectivity was certified at init;
        # strict orientation and boundary simplicity are rechecked above.
        # Optional exhaustive clipping provides a separately computed audit.
        pairs=[];maximum=0.;clipped=0;shared=0;zero_box=0
        if exhaustive:
            pairs=overlap_pairs(polygons.min(1),polygons.max(1))
            lo=polygons.min(1);hi=polygons.max(1)
            for i,j in pairs:
                if tuple(sorted((i,j))) in self._edge_neighbors:
                    shared+=1;continue
                # A rectangle intersection with zero width cannot contain a
                # positive polygon intersection area (represented coordinates).
                if np.any(np.minimum(hi[i],hi[j])<=np.maximum(lo[i],lo[j])):
                    zero_box+=1;continue
                intersection=clip_polygon(polygons[i],polygons[j]);clipped+=1
                overlap=area(intersection) if len(intersection)>=3 else 0.;maximum=max(maximum,overlap)
                if overlap>overlap_tol:
                    raise GeometryFailure(f'Positive overlap between Q4 {i} and {j}')
        boundary_area=area(q[self.boundary_loop]);total=float(sum(areas));closure=abs(boundary_area-total)
        if boundary_area<=0 or closure>8*overlap_tol:
            raise GeometryFailure('External area does not match tessellation')
        result=dict(boundary_edges=self.boundary_edges.copy(),boundary_owners=self.boundary_owners.copy(),
            boundary_loop=self.boundary_loop.copy(),minimum_corner_jacobian=float(minima.min()),
            minimum_Gauss_jacobian=float(gauss.min()),corner_jacobian_minima=minima,
            total_area=total,boundary_area=boundary_area,area_closure_error=closure,
            maximum_pairwise_overlap_area=maximum,position_roundoff_tolerance=position_tol,
            overlap_roundoff_tolerance=overlap_tol,
            orientation_proof='Bilinear 2D Q4 determinant is affine on the reference square; extrema at corners.',
            boundary_simple_checked=True,reference_topology_cached=True,overlap_candidate_pairs=len(pairs),
            overlap_clipping_evaluations=clipped,shared_edge_halfplane_certificates=shared,
            zero_area_AABB_certificates=zero_box,
            overlap_check='exhaustive_broadphase_clipping' if exhaustive else 'positive_cell_winding_certificate',
            overlap_certificate_conditions='Fixed conforming disk; strict convex CCW cells; one simple CCW boundary; exact cancellation of shared directed edges',
            mesh_validation_seconds=time.perf_counter()-started)
        self._last_q=q.copy();self._last_mesh=copy.deepcopy(result);self._last_exhaustive=exhaustive
        return result

    def validate(self,q,tool=None,penetration_tolerance=1e-12,*,exhaustive_overlap=None):
        if not np.isfinite(penetration_tolerance) or penetration_tolerance<0:
            raise ValueError('Nonnegative finite penetration tolerance required')
        q=self._positions(q);mesh=self._mesh(q,exhaustive_overlap)
        if tool is not None:
            values,broadphase=bounded_separations(q,self.cells,tool,mesh['position_roundoff_tolerance'],penetration_tolerance)
            if np.min(values)<-penetration_tolerance:
                raise GeometryFailure('A material Q4 overlaps the rounded tool')
            mesh.update(broadphase,minimum_tool_separation=float(values.min()))
        return mesh

    def contact_candidates(self,q,tool,gap_tolerance=1e-12):
        self.stats['contact_calls']+=1
        mesh=self.validate(q,tool,gap_tolerance)
        return _boundary_candidates(self._positions(q),self.cells,tool,mesh,gap_tolerance)


# The row construction below follows the historical
# q4_contact_candidate.geometry.contact_candidates after its validation and
# separation prefix. Only merge lookup is filtered; its predicates and order
# retain every external edge, row, owner and point ID.
def _boundary_candidates(q,cells,tool,mesh,gap_tolerance):
    tolerance=mesh['position_roundoff_tolerance'];angular=64*np.finfo(float).eps*2*np.pi
    result=[];rows_by_node={};C=np.asarray(tool.center);radius=tool.radius
    def add(a,b,s,normal,gap,feature,tool_point):
        if gap<-gap_tolerance:return
        s=float(np.clip(s,0.,1.));p=(1-s)*q[a]+s*q[b];n=np.asarray(normal);n=n/np.linalg.norm(n)
        coeff=np.zeros(len(q));coeff[a]=1-s;coeff[b]+=s
        # Geometric merging cannot combine distinct velocity traces.
        owner={'owner_cell':int(owner_cell),'edge_id':int(edge_id),'edge':(int(a),int(b)),
               'edge_parameter':s,'local_edge':int(local_edge)}
        # A convex edge trace has at most two nonzero coefficients summing
        # to one. Disjoint supports differ by at least 1/2 in max norm, so
        # cannot satisfy the unchanged 1e-13 merge test. Retain original row
        # order among eligible traces to preserve the historical first match.
        nodes=tuple(node for node in (a,b) if coeff[node]!=0.)
        eligible=sorted({row for node in nodes for row in rows_by_node.get(node,())})
        for row in eligible:
            r=result[row]
            if np.linalg.norm(p-r['position'])<=tolerance and abs(cross(n,r['normal']))<=angular and n@r['normal']>0 and np.max(abs(coeff-r['coefficients']))<1e-13:
                if not any(o['edge_id']==owner['edge_id'] for o in r['trace_owners']):r['trace_owners'].append(owner)
                canonical=min(r['trace_owners'],key=lambda o:(o['owner_cell'],o['edge_id']))
                r.update(canonical);r['parameter']=canonical['edge_parameter']
                return
        for node in nodes:rows_by_node.setdefault(node,[]).append(len(result))
        result.append({'edge':(int(a),int(b)),'parameter':s,'position':p,'normal':n,'gap':float(gap),
                       'tool_point':np.asarray(tool_point),'feature':feature,'coefficients':coeff,
                       **owner,'trace_owners':[owner]})
    for edge_id,((a,b),owner_cell) in enumerate(zip(mesh['boundary_edges'],mesh['boundary_owners'])):
        local_edge=int(np.flatnonzero(cells[owner_cell]==a)[0])
        p=q[a];d=q[b]-p;length2=float(d@d)
        # The normal projection onto each finite face is an interval in s.
        for j,(_,n,t,length,origin) in enumerate(tool.faces[:3]):
            n=np.asarray(n);t=np.asarray(t);origin=np.asarray(origin);z0=float((p-origin)@t);dz=float(d@t)
            if abs(dz)<=tolerance:
                interval=(0.,1.) if -tolerance<=z0<=length+tolerance else None
            else:
                endpoints=sorted((-z0/dz,(length-z0)/dz));lo=max(0.,endpoints[0]);hi=min(1.,endpoints[1]);interval=(lo,hi) if lo<=hi else None
            if interval is not None:
                for s in interval:
                    point=p+s*d;gap=float((point-origin)@n);rigid=point-gap*n
                    add(a,b,s,n,gap,'face_'+str(j),rigid)
        # The radial gap along a segment has its minimum at the circle-center
        # projection or at an endpoint. Each retained point has its own normal.
        for s in (0.,1.,float(np.clip((C-p)@d/length2,0.,1.))):
            point=p+s*d;delta=point-C;distance=np.linalg.norm(delta)
            if distance==0:continue
            n=delta/distance;parameter=arc_parameter(tool,n)
            if angular<parameter<tool.arc_sweep-angular:
                add(a,b,s,n,distance-radius,'arc_interior',C+radius*n)
            elif min(parameter,abs(parameter-2*np.pi),abs(parameter-tool.arc_sweep))<=angular:
                add(a,b,s,n,distance-radius,'arc_transition',C+radius*n)
        # Nonsmooth posterior vertex projections are included geometrically,
        # but active impulse there is deliberately outside this prototype.
        for j,vertex in enumerate(tool.vertices[1:]):
            vertex=np.asarray(vertex);s=float(np.clip((vertex-p)@d/length2,0.,1.));point=p+s*d;delta=point-vertex;distance=np.linalg.norm(delta)
            if distance>tolerance:
                n=delta/distance
                if support(tool,n)-vertex@n<=tolerance:add(a,b,s,n,distance,'posterior_vertex_'+str(j),vertex)
    if not result:raise GeometryFailure('No external closest-feature candidates')
    for point_id,p in enumerate(result):p['point_id']=point_id
    return result,dict(mesh,minimum_tool_separation=mesh['minimum_tool_separation'],candidate_count=len(result),
                       angular_roundoff_tolerance=angular,finite_step_admissibility='not implied; revalidate moved mesh and all tool separations')
