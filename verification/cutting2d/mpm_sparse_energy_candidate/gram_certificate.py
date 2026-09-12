"""Independent complete Gram-inverse certificate; no QR/LU rank inference.

For the EXACT supplied binary64 A, G*=A.T A is positive semidefinite.
ANY computed R with ||G* R-I||_2 <= e < 1 proves full rank and
 sigma_min(A) >= sqrt((1-e)/||R||_2).
Every column of R and the residual is checked in blocks. The residual bound
includes both sparse product rounding and the ENTIRE error in forming Ghat.
An LU merely proposes R. Dependent columns of the full Z are also all checked.
No integration with the frozen range/Gram caches is performed by this module.
"""
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from cutting2d.energy import ProjectorFailure
from .rank_certificate import EPS,ETA,up,down,gamma,norm_upper,product_and_error,spectral_scale_bounds


def _sum_upper(*arrays):
    value=np.zeros_like(np.asarray(arrays[0],float))
    for term in arrays:value=np.nextafter(value+term,np.inf)
    return value


def formed_gram_with_error(A):
    """Componentwise complete |Ghat - exact(A.T A)| bound, sparsely.

    The explicit overlap pattern includes entries whose products underflow.
    All sums have at most the maximum column support size of A terms. The
    positive sparse product supplies absolute magnitudes, inflated for its
    own rounding. Structural counts use only exact small integer arithmetic.
    """
    A=sparse.csc_matrix(A,dtype=float,copy=True);A.eliminate_zeros()
    k=int(np.max(np.diff(A.indptr),initial=0));g=gamma(k)
    G=(A.T@A).tocsc();absolute=(abs(A).T@abs(A)).tocsc()
    pattern=A.copy();pattern.data=np.ones_like(pattern.data);pattern=(pattern.T@pattern).tocsc()
    # k<2**53 and the dot products sum exact ones, so the overlap pattern
    # cannot omit a term that merely underflowed in A.T A.
    if k>=2**53:raise ValueError('Structural overlap count outside exact integer range')
    errors=absolute.copy();coefficient=up(g/(1-g))
    errors.data=np.nextafter(coefficient*errors.data,np.inf)
    pattern.data[:]=up(2*max(k,1)*ETA)
    errors=(errors+pattern).tocsc();errors.data=np.nextafter(errors.data,np.inf)
    if not np.isfinite(G.data).all() or not np.isfinite(errors.data).all():raise ProjectorFailure('Gram formation overflow/nonfinite')
    return G,errors,k


def certify_gram(Z,retained_columns=None,*,block_size=32,separation=32.):
    if type(block_size) is not int or block_size<1 or separation<32:raise ValueError('Positive block and at least 32-fold separation required')
    started=time.perf_counter();Z=sparse.csc_matrix(Z,dtype=float,copy=True);Z.sum_duplicates();Z.sort_indices();Z.eliminate_zeros()
    if not np.isfinite(Z.data).all():raise ValueError('Finite supplied Z required')
    rows,columns=Z.shape
    retained=np.arange(columns,dtype=np.int64) if retained_columns is None else np.asarray(retained_columns)
    if (retained.ndim!=1 or not np.issubdtype(retained.dtype,np.integer) or len(retained)==0 or
        len(np.unique(retained))!=len(retained) or np.any(retained<0) or np.any(retained>=columns)):
        raise ValueError('Unique explicit retained-column proposal required')
    r=len(retained);A=Z[:,retained];dependent=np.setdiff1d(np.arange(columns),retained)
    t=time.perf_counter();G,Gerror,terms=formed_gram_with_error(A);formation_s=time.perf_counter()-t
    t=time.perf_counter()
    try:lu=splu(G,permc_spec='COLAMD')
    except RuntimeError as error:raise ProjectorFailure('Gram inverse proposal failed; no rank conclusion') from error
    factor_s=time.perf_counter()-t;R2=E2=formation_effect2=rounding_effect2=0.;peak=0;t=time.perf_counter()
    for first in range(0,r,block_size):
        stop=min(first+block_size,r);eye=np.zeros((r,stop-first));eye[first:stop]=np.eye(stop-first)
        R=lu.solve(eye)
        if not np.isfinite(R).all():raise ProjectorFailure('Nonfinite Gram inverse proposal')
        product,error=product_and_error(G,R)
        formation,formation_error=product_and_error(Gerror,abs(R))
        subtraction=np.nextafter(EPS*(abs(product)+eye)+ETA,np.inf)
        residual=_sum_upper(abs(product-eye),error,subtraction,formation,formation_error)
        rn=norm_upper(R);en=norm_upper(residual);fn=norm_upper(_sum_upper(formation,formation_error))
        pn=norm_upper(_sum_upper(error,subtraction))
        R2=up(R2+up(rn*rn));E2=up(E2+up(en*en));formation_effect2=up(formation_effect2+up(fn*fn));rounding_effect2=up(rounding_effect2+up(pn*pn))
        peak=max(peak,sum(a.nbytes for a in (eye,R,product,error,formation,formation_error,subtraction,residual)))
    inverse_check_s=time.perf_counter()-t;e=up(np.sqrt(E2));rnorm=up(np.sqrt(R2))
    minimum=down(np.sqrt(max(0.,down(down(1-e)/rnorm)))) if e<1 and rnorm>0 else 0.
    remainder2=0.;t=time.perf_counter()
    for first in range(0,len(dependent),block_size):
        ids=dependent[first:first+block_size];target=Z[:,ids].toarray();T=lu.solve(np.asarray(A.T@target))
        product,error=product_and_error(A,T)
        bound=_sum_upper(abs(target-product),error,np.nextafter(EPS*(abs(target)+abs(product))+ETA,np.inf))
        n=norm_upper(bound);remainder2=up(remainder2+up(n*n))
        peak=max(peak,sum(a.nbytes for a in (target,T,product,error,bound)))
    dependent_s=time.perf_counter()-t;remainder=up(np.sqrt(remainder2)) if remainder2 else 0.
    low,high=spectral_scale_bounds(Z);cutlow=down(max(Z.shape)*EPS*low);cuthigh=up(max(Z.shape)*EPS*high)
    retained_pass=minimum>separation*cuthigh;discarded_pass=not len(dependent) or remainder<cutlow/separation
    return dict(certified=bool(retained_pass and discarded_pass),rank=r,columns=columns,rows=rows,full_column_rank=r==columns,
        method='complete_Gram_inverse_residual_WITH_Gram_formation_error_v1',retained_columns=retained.tolist(),
        inverse_residual_F_upper=e,inverse_F_upper=rnorm,
        formation_error_effect_F_upper=up(np.sqrt(formation_effect2)),product_rounding_effect_F_upper=up(np.sqrt(rounding_effect2)),
        sigma_r_lower=minimum,sigma_r_plus_one_upper=remainder,sigma_max_lower=low,sigma_max_upper=high,
        cutoff_lower=cutlow,cutoff_upper=cuthigh,separation_factor=separation,
        retained_separated=bool(retained_pass),discarded_separated=bool(discarded_pass),
        visited_inverse_columns=r,visited_dependent_columns=len(dependent),block_size=block_size,
        maximum_Gram_dot_product_terms=terms,Gram_nnz=G.nnz,Gram_error_nnz=Gerror.nnz,
        LU_nnz=lu.L.nnz+lu.U.nnz,peak_explicit_block_bytes=peak,
        timings=dict(formation_s=formation_s,LU_s=factor_s,inverse_check_s=inverse_check_s,dependent_check_s=dependent_s),
        elapsed_s=time.perf_counter()-started,
        scope='Exact real Gram of the frozen binary64 supplied map; all columns checked with outward IEEE bounds; no rank inference from LU, QR, probes or eigensolvers')
