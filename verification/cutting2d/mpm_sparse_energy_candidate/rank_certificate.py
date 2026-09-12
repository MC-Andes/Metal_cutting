"""Complete a posteriori rank bounds on the supplied binary64 sparse map.

For A=Z[:,retained], ANY computed L and T give
 sigma_min(A) >= (1-||L A-I||_2)/||L||_2,
 sigma_(r+1)(Z) <= ||Z_dependent-A T||_2.
We use Frobenius upper bounds, evaluating every column in bounded blocks.
Native QR supplies only the candidate L and T; success/rank of QR is not a
premise of the bounds. Product roundoff uses the standard gamma_k model,
with epsilon (twice unit roundoff), outward scalar rounding, and an absolute
subnormal term. Inputs must remain finite; no overflow/flush-to-zero is
permitted. This certifies a frozen floating matrix, not its continuum rank.
"""
import time
import numpy as np
from scipy import sparse
from cutting2d.energy import ProjectorFailure

EPS=np.finfo(float).eps
ETA=np.finfo(float).smallest_subnormal
def up(x):return float(np.nextafter(float(x),np.inf))
def down(x):return float(np.nextafter(float(x),-np.inf))
def gamma(k):
    value=int(k)*EPS
    if value>=.01:raise ValueError('Roundoff bound dimension outside candidate scope')
    return up(value/(1-value))


def norm_upper(value):
    value=np.asarray(value,float);maximum=float(np.max(abs(value),initial=0.))
    if not np.isfinite(maximum):raise ProjectorFailure('Nonfinite certificate operand')
    if maximum==0.:return 0.
    answer=float(np.linalg.norm(value/maximum))*maximum
    return up(answer*(1+4*gamma(value.size)))


def product_and_error(A,B):
    """Componentwise absolute bound for the COMPLETE sparse-dense product."""
    A=sparse.csr_matrix(A);B=np.asarray(B,float)
    terms=int(np.max(np.diff(A.indptr),initial=0));g=gamma(terms)
    product=A@B;absolute=abs(A)@abs(B)
    error=np.nextafter((g/(1-g))*absolute+up(2*max(terms,1)*ETA),np.inf)
    if not np.isfinite(product).all() or not np.isfinite(error).all():raise ProjectorFailure('Certificate product overflow/nonfinite')
    return product,error


def spectral_scale_bounds(Z):
    """Bound max column norm and sqrt(||Z||1 ||Z||inf) after scaling.

    A single maximum-entry scale suffices: the largest scaled column norm
    and the largest scaled row/column sums are all >= 1. Thus division and
    square underflow contribute at most k*ETA to aggregates of size >= 1;
    the conservative 4*gamma(k) budgets also cover these absolute errors.
    Normalizing before squares avoids both the subnormal-square overestimate
    and unnecessary overflow of a representable spectral scale.
    """
    Z=sparse.csc_matrix(Z,dtype=float,copy=True);Z.sum_duplicates();Z.sort_indices()
    if not np.isfinite(Z.data).all():raise ProjectorFailure('Nonfinite spectral-scale operand')
    maximum=float(np.max(abs(Z.data),initial=0.))
    if maximum==0.:return 0.,0.
    with np.errstate(under='ignore'):Z.data=Z.data/maximum
    row=Z.tocsr()
    kc=int(np.max(np.diff(Z.indptr),initial=0));kr=int(np.max(np.diff(row.indptr),initial=0))
    col_margin=down(1-up(4*gamma(kc)));row_margin=down(1-up(4*gamma(kr)))
    # Gradual underflow is explicitly included in the error budget; zero
    # rounded tiny ratios/squares do not erase the unit maximum entry.
    with np.errstate(under='ignore'):
        norms=np.sqrt(np.asarray(Z.multiply(Z).sum(0)).ravel())
    lower_scaled=max(0.,down(float(np.max(norms,initial=0.))*col_margin))
    one=up(float(np.max(np.asarray(abs(Z).sum(0)),initial=0.))/col_margin)
    inf=up(float(np.max(np.asarray(abs(Z).sum(1)),initial=0.))/row_margin)
    upper_scaled=up(up(np.sqrt(one))*up(np.sqrt(inf)))
    # An unrepresentable upper bound must fail explicitly, never silently
    # create an infinite scale used by a downstream rank decision.
    if maximum>np.finfo(float).max/upper_scaled:
        raise ProjectorFailure('Finite spectral upper bound is not representable')
    lower=max(0.,down(lower_scaled*maximum));upper=up(upper_scaled*maximum)
    if not np.isfinite(upper):raise ProjectorFailure('Finite spectral upper bound is not representable')
    return lower,upper


def certify(factor,*,block_size=32,separation=32.):
    if block_size<1 or separation<32:raise ValueError('Positive block and no weaker than 32-fold rank separation required')
    started=time.perf_counter();Z=factor.Z;rows,columns=Z.shape;r=factor.rank
    retained=factor.retained_columns;dependent=np.setdiff1d(np.arange(columns),retained)
    A=Z[:,retained].tocsr();L2=E2=0.;peak=0
    for first in range(0,r,block_size):
        stop=min(first+block_size,r);size=stop-first;b=np.zeros((columns,size))
        b[retained[first:stop],np.arange(size)]=1.
        # Raw native solve is a triangular-factor utility here, not a physical
        # incompatible-load solve. Its material response equals L_block.T.
        _,Lt=factor.native.solve_gram(b)
        product,error=product_and_error(A.T,Lt)
        identity=np.zeros_like(product);identity[first:stop,:]=np.eye(size)
        residual=product-identity
        # Conservatively cover the final diagonal subtraction as well.
        bound=np.nextafter(abs(residual)+error+EPS*(abs(product)+identity)+ETA,np.inf)
        en=norm_upper(bound);ln=norm_upper(Lt)
        E2=up(E2+up(en*en));L2=up(L2+up(ln*ln));peak=max(peak,Lt.nbytes+b.nbytes+product.nbytes+error.nbytes+bound.nbytes)
    e=up(np.sqrt(E2)) if E2 else 0.;lnorm=up(np.sqrt(L2)) if L2 else 0.
    minimum=down(down(1-e)/lnorm) if e<1 and lnorm>0 else 0.
    remainder2=0.
    for first in range(0,len(dependent),block_size):
        ids=dependent[first:first+block_size];target=Z[:,ids].toarray();_,coeff=factor.native.project(target);T=coeff[retained]
        product,error=product_and_error(A,T)
        residual=target-product
        bound=np.nextafter(abs(residual)+error+EPS*(abs(target)+abs(product))+ETA,np.inf)
        n=norm_upper(bound);remainder2=up(remainder2+up(n*n));peak=max(peak,target.nbytes+coeff.nbytes+product.nbytes+error.nbytes+bound.nbytes)
    remainder=up(np.sqrt(remainder2)) if remainder2 else 0.
    smaxlow,smaxhigh=spectral_scale_bounds(Z);cutlow=down(max(Z.shape)*EPS*smaxlow);cuthigh=up(max(Z.shape)*EPS*smaxhigh)
    retained_pass=bool(r>0 and minimum>separation*cuthigh)
    discarded_pass=bool(not len(dependent) or remainder<cutlow/separation)
    record=dict(certified=retained_pass and discarded_pass,rank=r,columns=columns,rows=rows,full_column_rank=r==columns,
        method='complete_left_inverse_and_dependent_reconstruction_with_gamma_bounds_v1',
        left_inverse_residual_F_upper=e,left_inverse_F_upper=lnorm,sigma_r_lower=minimum,
        sigma_r_plus_one_upper=remainder,sigma_max_lower=smaxlow,sigma_max_upper=smaxhigh,
        cutoff_lower=cutlow,cutoff_upper=cuthigh,separation_factor=separation,
        retained_separated=retained_pass,discarded_separated=discarded_pass,
        visited_left_inverse_columns=r,visited_dependent_columns=len(dependent),block_size=block_size,
        peak_explicit_block_bytes=peak,elapsed_s=time.perf_counter()-started,
        scope='Frozen binary64 supplied Z; standard IEEE round-to-nearest sparse sums, finite arithmetic and gradual underflow assumed; no random/eigensolver rank inference')
    return record


class FullColumnReference:
    """Weyl reuse only for a fully certified column rank and exact identities."""
    def __init__(self,Z,certificate,*,row_ids,column_ids):
        if not certificate['certified'] or not certificate['full_column_rank']:
            raise ProjectorFailure('Weyl reuse requires certified FULL column rank')
        self.Z=sparse.csc_matrix(Z,copy=True);self.certificate=dict(certificate)
        self.row_ids=tuple(row_ids);self.column_ids=tuple(column_ids)
        if len(self.row_ids)!=Z.shape[0] or len(self.column_ids)!=Z.shape[1]:raise ValueError('Exact operator identity lengths required')

    def check(self,Z,*,row_ids,column_ids):
        Z=sparse.csc_matrix(Z)
        if tuple(row_ids)!=self.row_ids or tuple(column_ids)!=self.column_ids or Z.shape!=self.Z.shape:
            raise ProjectorFailure('Weyl reference operator identities changed')
        delta=(Z-self.Z).tocoo();roundoff=(abs(Z)+abs(self.Z)).tocoo()
        # Sparse subtraction is rounded once; include all structural entries.
        change=up(norm_upper(delta.data)+EPS*norm_upper(roundoff.data))
        lower=down(self.certificate['sigma_r_lower']-change);_,smax=spectral_scale_bounds(Z)
        cutoff=up(max(Z.shape)*EPS*smax)
        return dict(certified=lower>self.certificate['separation_factor']*cutoff,
            method='full_column_rank_Weyl_reference_v1',sigma_min_lower=lower,change_F_upper=change,
            cutoff_upper=cutoff,scope='Same explicit row and column identities; no reuse for rank-deficient references')
