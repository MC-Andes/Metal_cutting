"""Verification-only pair-search substitution; all exact mesh checks retained."""
import numpy as np
from verification.cutting2d.q4_contact_candidate.geometry import (
    GeometryFailure, cross, area, jacobians, clip_polygon,
)

REFERENCE_SHA256 = "408b3c1ae23d249826d6a36752ab08c3fe1f52dccc62072a90414ba32438fb04"


def unresolved_vertices(q, tolerance):
    """Conservative coordinate sweep, followed by the unchanged squared norm.

    The radius only selects candidates; it does not replace the original
    tolerance. Outward nextafter also covers the rounding of the square/root.
    Subtraction is evaluated in the same coordinates as the original norm.
    """
    square = tolerance**2
    if np.isinf(square):
        # The dense reference also compares its +inf diagonal with square.
        return bool(len(q))
    radius = np.nextafter(np.sqrt(np.nextafter(square, np.inf)), np.inf)
    order = np.argsort(q[:,0], kind='stable')
    left = 0
    for k, i in enumerate(order):
        while left < k and q[i,0] - q[order[left],0] > radius:
            left += 1
        near = order[left:k]
        if not len(near):
            continue
        near = near[np.abs(q[i,1] - q[near,1]) <= radius]
        if len(near) and np.any(np.sum((q[i] - q[near])**2, axis=1) <= square):
            return True
    return False


def overlap_pairs(lo, hi):
    """Closed-AABB pairs in exactly the original lower-triangle row order."""
    order = np.argsort(lo[:,0], kind='stable')
    active = np.empty(0, dtype=np.int64)
    pairs = []
    for i in order:
        active = active[hi[active,0] >= lo[i,0]]
        hit = active[(hi[i,1] >= lo[active,1]) & (hi[active,1] >= lo[i,1])]
        pairs.extend((max(int(i),int(j)), min(int(i),int(j))) for j in hit)
        active = np.r_[active, i]
    pairs.sort()
    return pairs


def validate_mesh_sparse(q,cells):
    q=np.asarray(q,float);cells=np.asarray(cells)
    if q.ndim!=2 or q.shape[1]!=2 or not np.isfinite(q).all():raise GeometryFailure('Finite N by 2 vertices required')
    if cells.ndim!=2 or cells.shape[1]!=4 or cells.dtype.kind not in 'iu':raise GeometryFailure('Integer E by 4 connectivity required')
    if not len(cells) or cells.min()<0 or cells.max()>=len(q):raise GeometryFailure('Invalid or empty connectivity')
    if len(np.unique(cells))!=len(q):raise GeometryFailure('Unused vertices excluded')
    extent=max(float(np.max(np.ptp(q,axis=0))),np.finfo(float).tiny)
    position_tol=128*np.finfo(float).eps*max(extent,float(np.max(abs(q))))
    if unresolved_vertices(q, position_tol):
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
    for i,j in overlap_pairs(lo, hi):
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
