"""Verification-only conservative broad phase; original row construction preserved.

Only the Q4/tool separation loop differs from frozen geometry.py.
Every call still validates the complete mesh and constructs every boundary row.
"""
import numpy as np
from verification.cutting2d.q4_contact_candidate.geometry import (
    GeometryFailure, validate_mesh, separation, arc_parameter, support, cross,
)

FROZEN_GEOMETRY_SHA256 = "408b3c1ae23d249826d6a36752ab08c3fe1f52dccc62072a90414ba32438fb04"


def bounded_separations(q, cells, tool, position_tolerance, gap_tolerance):
    """Positive disjoint-AABB bounds replace only distant separation evaluations.

    The box contains all original vertices AND the entire fillet circle. A
    Q4 lies in the convex hull of its four vertices. Outward expansion prevents
    a marginal floating-point box comparison from being used as a certificate.
    Returned distant values are lower bounds, never exact distances.
    """
    vertices = np.asarray(tool.vertices, float)
    center = np.asarray(tool.center, float)
    cloud = np.vstack((vertices, center - tool.radius, center + tool.radius))
    scale = max(float(np.max(abs(q))), float(np.max(abs(cloud))),
                float(np.max(np.ptp(q, axis=0))), float(tool.radius), np.finfo(float).tiny)
    padding = 512 * np.finfo(float).eps * scale
    tool_lo = np.nextafter(cloud.min(0) - padding, -np.inf)
    tool_hi = np.nextafter(cloud.max(0) + padding, np.inf)
    polys = q[cells]
    lo = np.nextafter(polys.min(1), -np.inf)
    hi = np.nextafter(polys.max(1), np.inf)
    axis_gap = np.maximum(lo - tool_hi, tool_lo - hi)
    lower_bound = np.nextafter(axis_gap.max(1), -np.inf)
    discarded = lower_bound > max(position_tolerance, gap_tolerance)
    values = lower_bound.copy()
    for k in np.flatnonzero(~discarded):
        values[k] = separation(polys[k], tool)[0]
    return values, {
        'tool_outer_box_lower': tool_lo,
        'tool_outer_box_upper': tool_hi,
        'tool_outer_box_roundoff_padding': padding,
        'discarded_separation_cells': np.flatnonzero(discarded),
        'separation_exact_evaluations': int(np.count_nonzero(~discarded)),
        'separation_bound_evaluations': int(np.count_nonzero(discarded)),
        'per_cell_separation_values': values,
        'per_cell_is_lower_bound': discarded,
        'minimum_tool_separation_kind': 'conservative_lower_bound' if discarded.any() else 'frozen_separation',
        'minimum_tool_separation_is_lower_bound': bool(discarded.any()),
        'broadphase_scope': 'All mesh validation and external boundary candidates are retained; only distant Q4 separation calls are skipped.',
    }


def contact_candidates_bounded(q,cells,tool,gap_tolerance=1e-12):
    """Real endpoint/face-overlap/arc-closest constraints on external segments.

    Flat faces retain their complete overlap interval endpoints and own gaps.
    Arc points use their own radial normals; no common tangent-plane barrier.
    The nonlinear finite-step gap must still be checked after moving vertices.
    """
    q=np.asarray(q,float);mesh=validate_mesh(q,cells)
    separations, broadphase = bounded_separations(q, cells, tool, mesh['position_roundoff_tolerance'], gap_tolerance)
    mesh = dict(mesh, **broadphase)
    if min(separations)<-gap_tolerance:raise GeometryFailure('A material Q4 overlaps the rounded tool')
    tolerance=mesh['position_roundoff_tolerance'];angular=64*np.finfo(float).eps*2*np.pi
    result=[];C=np.asarray(tool.center);radius=tool.radius
    def add(a,b,s,normal,gap,feature,tool_point):
        if gap<-gap_tolerance:return
        s=float(np.clip(s,0.,1.));p=(1-s)*q[a]+s*q[b];n=np.asarray(normal);n=n/np.linalg.norm(n)
        coeff=np.zeros(len(q));coeff[a]=1-s;coeff[b]+=s
        # Geometric merging cannot combine distinct velocity traces.
        owner={'owner_cell':int(owner_cell),'edge_id':int(edge_id),'edge':(int(a),int(b)),
               'edge_parameter':s,'local_edge':int(local_edge)}
        for r in result:
            if np.linalg.norm(p-r['position'])<=tolerance and abs(cross(n,r['normal']))<=angular and n@r['normal']>0 and np.max(abs(coeff-r['coefficients']))<1e-13:
                if not any(o['edge_id']==owner['edge_id'] for o in r['trace_owners']):r['trace_owners'].append(owner)
                canonical=min(r['trace_owners'],key=lambda o:(o['owner_cell'],o['edge_id']))
                r.update(canonical);r['parameter']=canonical['edge_parameter']
                return
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
    return result,dict(mesh,minimum_tool_separation=float(min(separations)),candidate_count=len(result),
                       angular_roundoff_tolerance=angular,finite_step_admissibility='not implied; revalidate moved mesh and all tool separations')
