"""Verification-only conforming Q4 geometry and real boundary trace points.

Supported mesh: one intact, connected, hole-free body of strictly oriented Q4s.
No APIC centres, affine particle domains or production contact functions.
"""
import numpy as np


class GeometryFailure(ValueError):pass


def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def area(poly):
    p=np.asarray(poly)-np.mean(poly,axis=0)
    return float(.5*np.sum(cross(p,np.roll(p,-1,axis=0))))


def jacobians(q,coordinates):
    sign=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    values=[];centered=q-q.mean(0)
    for xi,eta in coordinates:
        D=np.c_[sign[:,0]*(1+sign[:,1]*eta),sign[:,1]*(1+sign[:,0]*xi)]/4
        values.append(float(np.linalg.det(centered.T@D)))
    return np.array(values)


def clip_polygon(subject,clip):
    p=np.asarray(subject)
    for a,b in zip(clip,np.roll(clip,-1,axis=0)):
        if not len(p):break
        result=[];old=p[-1];fo=cross(b-a,old-a)
        for new in p:
            fn=cross(b-a,new-a)
            if (fo>=0)!=(fn>=0):result.append(old+fo/(fo-fn)*(new-old))
            if fn>=0:result.append(new)
            old,fo=new,fn
        p=np.asarray(result).reshape(-1,2)
    return p


def validate_mesh(q,cells):
    q=np.asarray(q,float);cells=np.asarray(cells)
    if q.ndim!=2 or q.shape[1]!=2 or not np.isfinite(q).all():raise GeometryFailure('Finite N by 2 vertices required')
    if cells.ndim!=2 or cells.shape[1]!=4 or cells.dtype.kind not in 'iu':raise GeometryFailure('Integer E by 4 connectivity required')
    if not len(cells) or cells.min()<0 or cells.max()>=len(q):raise GeometryFailure('Invalid or empty connectivity')
    if len(np.unique(cells))!=len(q):raise GeometryFailure('Unused vertices excluded')
    extent=max(float(np.max(np.ptp(q,axis=0))),np.finfo(float).tiny)
    position_tol=128*np.finfo(float).eps*max(extent,float(np.max(abs(q))))
    distance2=np.sum((q[:,None]-q[None,:])**2,axis=2);np.fill_diagonal(distance2,np.inf)
    if np.any(distance2<=position_tol**2):
        raise GeometryFailure('Distinct vertex IDs have coincident or unresolved positions')
    corners=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    gauss=corners/np.sqrt(3);minimum=[];minimum_gauss=[];edges={};areas=[];seen=set()
    for e,ids in enumerate(cells):
        if len(set(ids))!=4:raise GeometryFailure('Repeated vertex in a Q4')
        canonical=tuple(sorted(ids))
        if canonical in seen:raise GeometryFailure('Duplicate Q4 domain')
        seen.add(canonical);poly=q[ids];d=jacobians(poly,corners);L=max(np.linalg.norm(np.roll(poly,-1,axis=0)-poly,axis=1))
        tolerance=128*np.finfo(float).eps*L*L
        if d.min()<=tolerance:raise GeometryFailure(f'Q4 {e} has nonpositive or unresolved corner Jacobian: {d.min():.17g}')
        minimum.append(float(d.min()));minimum_gauss.append(float(jacobians(poly,gauss).min()));areas.append(area(poly))
        for a,b in zip(ids,np.roll(ids,-1)):
            key=tuple(sorted((int(a),int(b))));edges.setdefault(key,[]).append((e,int(a),int(b)))
    neighbors=[set() for _ in cells];boundary=[]
    for key,owners in edges.items():
        if len(owners)>2:raise GeometryFailure('Nonmanifold edge shared by more than two domains')
        if len(owners)==2:
            p,a,b=owners[0];r,c,d=owners[1]
            if (a,b)!=(d,c):raise GeometryFailure('Shared edge orientations do not oppose')
            neighbors[p].add(r);neighbors[r].add(p)
        else:boundary.append(owners[0])
    visited={0};todo=[0]
    while todo:
        for e in neighbors[todo.pop()]-visited:visited.add(e);todo.append(e)
    if len(visited)!=len(cells):raise GeometryFailure('Disconnected material body excluded')
    outgoing={};incoming={}
    for e,a,b in boundary:
        if a in outgoing or b in incoming:raise GeometryFailure('Branched/nonmanifold boundary vertex')
        outgoing[a]=b;incoming[b]=a
    if set(outgoing)!=set(incoming):raise GeometryFailure('Open material boundary')
    start=min(outgoing);loop=[start];node=outgoing[start]
    while node!=start:
        if node in loop:raise GeometryFailure('Invalid boundary cycle')
        loop.append(node);node=outgoing[node]
    if len(loop)!=len(boundary):raise GeometryFailure('Holes or multiple boundary cycles excluded')
    # Pairwise convex clipping detects both crossings and containment. This is
    # a bounded reference validator, not an optimized production broad phase.
    overlap_tolerance=128*np.finfo(float).eps*extent*extent
    maximum_overlap=0.
    polygons=q[cells];lo=polygons.min(1);hi=polygons.max(1)
    possible=np.all(hi[:,None]>=lo[None,:],axis=2)&np.all(hi[None,:]>=lo[:,None],axis=2)
    for i,j in zip(*np.where(np.tril(possible,-1))):
        intersection=clip_polygon(polygons[i],polygons[j])
        overlap=area(intersection) if len(intersection)>=3 else 0.;maximum_overlap=max(maximum_overlap,overlap)
        if overlap>overlap_tolerance:raise GeometryFailure(f'Positive overlap between Q4 {i} and {j}')
    boundary_area=area(q[loop]);area_error=abs(boundary_area-sum(areas))
    if boundary_area<=0 or area_error>8*overlap_tolerance:raise GeometryFailure('External area does not match tessellation')
    return {'boundary_edges':np.array([(a,b) for e,a,b in boundary],int),'boundary_owners':np.array([e for e,a,b in boundary]),
        'boundary_loop':np.array(loop,int),'minimum_corner_jacobian':min(minimum),'minimum_Gauss_jacobian':min(minimum_gauss),
        'corner_jacobian_minima':np.array(minimum),'total_area':sum(areas),'boundary_area':boundary_area,
        'area_closure_error':area_error,'maximum_pairwise_overlap_area':maximum_overlap,
        'position_roundoff_tolerance':position_tol,'overlap_roundoff_tolerance':overlap_tolerance,
        'orientation_proof':'det(dq/d(xi,eta))=a+b xi+c eta for a bilinear 2D Q4; its extrema occur at the four reference corners.'}


def arc_parameter(tool,n):
    return (tool.arc_sign*(np.arctan2(n[1],n[0])-tool.arc_start))%(2*np.pi)


def support(tool,n):
    points=np.array([*tool.vertices[1:],*tool.tangent_points]);value=float(np.max(points@n))
    if arc_parameter(tool,n)<=tool.arc_sweep:value=max(value,float(np.asarray(tool.center)@n+tool.radius))
    return value


def separation(poly,tool):
    directions=[];edges=np.roll(poly,-1,axis=0)-poly
    for e in edges:
        n=np.array([e[1],-e[0]]);directions.extend((n,-n))
    directions.extend(np.array(f[1]) for f in tool.faces[:3])
    for vertex in poly:
        directions.append(vertex-np.asarray(tool.center))
        directions.extend(vertex-np.asarray(p) for p in tool.vertices[1:])
    best=(-np.inf,None)
    for v in directions:
        norm=np.linalg.norm(v)
        if norm==0:continue
        n=v/norm;gap=float(np.min(poly@n)-support(tool,n))
        if gap>best[0]:best=gap,n
    return best


def contact_candidates(q,cells,tool,gap_tolerance=1e-12):
    """Real endpoint/face-overlap/arc-closest constraints on external segments.

    Flat faces retain their complete overlap interval endpoints and own gaps.
    Arc points use their own radial normals; no common tangent-plane barrier.
    The nonlinear finite-step gap must still be checked after moving vertices.
    """
    q=np.asarray(q,float);mesh=validate_mesh(q,cells)
    separations=[separation(q[ids],tool)[0] for ids in cells]
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
