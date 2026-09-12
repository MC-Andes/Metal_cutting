"""Explicit SSPRK2/FCT subcycling on the caller's frozen current Q4 geometry.

This wrapper changes neither conductor, source, temperature, nor material mass
except by repeated calls to conductor.step. The returned heat is the sum of
those calls, never a repaired value inferred from the final temperature.
"""
from __future__ import annotations
import math
import numpy as np
from .thermal import ThermalStep

EPS=np.finfo(float).eps
UNIT_ROUNDOFF=EPS/2


def _readonly(value):
    result=np.array(value,copy=True);result.setflags(write=False);return result


def _gamma(count):
    product=int(count)*UNIT_ROUNDOFF
    if product>=.5:raise ValueError('Subcycling roundoff bound is not representable at this count')
    return product/(1-product)


def partition(dt,stable_dt_s):
    """Exact ceiling for the represented float inputs, then equal float steps."""
    dt=float(dt);stable=float(stable_dt_s)
    if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite macro dt required')
    if math.isnan(stable) or stable<=0:raise ValueError('Positive stable_dt_s or infinity required')
    if math.isinf(stable):return 1,dt,dict(partition_residual_s=0.,partition_roundoff_bound_s=0.)
    a,b=dt.as_integer_ratio();c,d=stable.as_integer_ratio()
    numerator=a*d;denominator=b*c
    n=(numerator+denominator-1)//denominator
    # Above 2**53 the integer count is no longer generally representable by the
    # float divisor used below. No silent count rounding or timestep clipping.
    if n>2**53:raise ValueError('Required thermal substep count is not exactly representable')
    subdt=dt/n
    if not np.isfinite(subdt) or subdt<=0 or subdt>stable:
        raise ValueError('Equal represented thermal substeps do not satisfy CFL')
    total=n*subdt;error=total-dt;bound=_gamma(2)*(abs(total)+abs(dt))
    if not np.isfinite(total) or abs(error)>bound:
        raise ValueError('Equal thermal substeps do not represent the requested interval')
    return n,subdt,dict(partition_residual_s=error,partition_roundoff_bound_s=bound)


def _variance(T,C,components):
    _,first=np.unique(components,return_index=True)
    u=T-T[first][components]
    means=np.bincount(components,weights=C*u)/np.bincount(components,weights=C)
    v=u-means[components]
    return float(np.dot(C,v*v))


def advance(conductor,temperature,dt):
    """Return ThermalStep for equal explicit substeps covering macro ``dt``.

    Caller must update and freeze geometry before entry. No physical heating
    source is applied here; plastic/friction heating remains with the caller.
    ``stages`` aggregates the two Euler-stage diagnostics over all substeps,
    so diagnostic storage is O(1) in the number of substeps.
    """
    T0=np.asarray(temperature,float)
    C=np.asarray(conductor.capacity,float)
    if T0.shape!=C.shape or T0.ndim!=1 or not len(T0) or not np.isfinite(T0).all() or np.any(T0<=0):
        raise ValueError('Finite positive material GP temperatures required')
    if not np.isfinite(C).all() or np.any(C<=0):raise ValueError('Finite positive physical capacity required')
    stable=float(conductor.stable_dt_s);n,subdt,time_identity=partition(dt,stable)
    # Account for reductions of positive L1 scales and their repeated scalar
    # accumulation. This factor does not relax the physical timestep or state.
    scale_gamma=_gamma(len(T0)+n+8);scale_denominator=1-scale_gamma
    components=np.asarray(conductor.components,int)
    if components.shape!=T0.shape or np.any(components<0):raise ValueError('Material component map invalid')
    T=T0.copy();heat=np.zeros_like(T);heat_L1=0.;state_L1=0.;max_net=0.;summed_net=0.
    max_stored_mismatch=0.;minT=float(T.min());maxT=float(T.max())
    corners0=conductor.corner_temperatures(T0)
    mincorner=float(corners0.min());maxcorner=float(corners0.max());all_corners_positive=bool(np.all(corners0>0))
    before=_variance(T0,C,components);stage_aggregates=[];stage_count=0;max_withheld=0.
    initial_lower=np.full(conductor.component_count,np.inf);initial_upper=np.full(conductor.component_count,-np.inf)
    np.minimum.at(initial_lower,components,T0);np.maximum.at(initial_upper,components,T0)
    for _ in range(n):
        result=conductor.step(T,subdt);next_T=np.asarray(result.temperature);Q=np.asarray(result.heat_J);diag=result.diagnostics
        if Q.shape!=T0.shape or next_T.shape!=T0.shape or not np.isfinite(Q).all() or not np.isfinite(next_T).all():
            raise ValueError('Conductor returned invalid thermal substep')
        heat+=Q
        heat_L1+=float(np.sum(abs(Q)))
        state_L1+=float(np.sum(C*(abs(T)+abs(next_T))))
        max_stored_mismatch=max(max_stored_mismatch,float(np.max(abs(C*(next_T-T)-Q))))
        step_net=float(Q.sum());max_net=max(max_net,abs(step_net));summed_net+=step_net
        minT=min(minT,float(next_T.min()));maxT=max(maxT,float(next_T.max()))
        mincorner=min(mincorner,float(diag['minimum_reconstructed_corner_K']))
        maxcorner=max(maxcorner,float(diag['maximum_reconstructed_corner_K']))
        all_corners_positive &= bool(diag['reconstructed_corners_positive'])
        max_withheld=max(max_withheld,float(diag['maximum_withheld_flux_fraction']))
        for index,stage in enumerate(diag.get('stages',())):
            stage_count+=1
            if index==len(stage_aggregates):stage_aggregates.append(dict(stage))
            else:
                aggregate=stage_aggregates[index]
                for key,value in stage.items():
                    if key=='minimum_alpha':aggregate[key]=min(aggregate[key],value)
                    elif key=='heat_sum_J':aggregate[key]+=value
                    else:aggregate[key]=max(aggregate[key],value)
        T=next_T
    if not np.isfinite(heat).all() or not np.isfinite(heat_L1) or not np.isfinite(state_L1):
        raise ValueError('Accumulated thermal heat/roundoff scale is not finite')
    stored=C*(T-T0);residual=stored-heat
    stored_L1=float(np.sum(abs(stored)));total_heat_L1=float(np.sum(abs(heat)))
    endpoint_L1=float(np.sum(C*(abs(T0)+abs(T))))
    # fl(T+fl(Q/C)) introduces absolute-temperature addition roundoff. A bound
    # based only on |Q| would reject valid sub-ulp increments or hide that loss.
    state_bound=_gamma(4)*(state_L1+heat_L1)/scale_denominator
    heat_sum_bound=_gamma(n-1)*heat_L1/scale_denominator
    endpoint_bound=_gamma(2)*endpoint_L1/scale_denominator
    residual_bound=_gamma(3)*(stored_L1+total_heat_L1)/scale_denominator
    underflow_bound=(n*len(T0)+4*len(T0))*np.nextafter(0.,1.)*(1+float(np.max(C)))
    bound=state_bound+heat_sum_bound+endpoint_bound+residual_bound+underflow_bound
    residual_L1=float(np.sum(abs(residual)))
    if not np.isfinite(bound) or not np.isfinite(residual_L1) or residual_L1>bound:
        raise ValueError('Summed physical heat and stored C deltaT exceed the L1 floating-point bound')
    after=_variance(T,C,components)
    if not np.isfinite(after) or after>before+1e-11*max(before,np.finfo(float).tiny):
        raise ValueError('Subcycled thermal variance increased beyond roundoff')
    tolerance=128*EPS*max(float(np.max(abs(T0))),1.)
    if np.any(T<=0) or np.any(T<initial_lower[components]-tolerance) or np.any(T>initial_upper[components]+tolerance):
        raise ValueError('Subcycled material temperatures violated component bounds')
    final_corners=conductor.corner_temperatures(T)
    diagnostics=dict(operator='direct_q4_dg_fct_ssprk2_subcycled_v1',substep_operator=conductor.diagnostics['operator'],
        k0_exact=conductor.k==0,dt_s=float(dt),nsubsteps=n,substep_dt_s=subdt,
        stable_dt_s=stable if np.isfinite(stable) else None,stable_dt_unbounded=bool(np.isinf(stable)),
        maximum_substep_over_stable_dt=subdt/stable,**time_identity,
        heat_sum_J=float(heat.sum()),sum_substep_net_heat_J=summed_net,maximum_substep_net_heat_J=max_net,
        accumulated_substep_heat_L1_J=heat_L1,returned_heat_L1_J=total_heat_L1,
        stored_heat_change_J=float(stored.sum()),maximum_stored_heat_residual_J=float(np.max(abs(residual))),
        heat_state_residual_L1_J=residual_L1,heat_state_roundoff_bound_L1_J=bound,
        heat_state_roundoff_bound_terms_J=dict(state_updates=state_bound,heat_summation=heat_sum_bound,
            endpoint_conversion=endpoint_bound,residual_arithmetic=residual_bound,underflow=underflow_bound),
        heat_state_roundoff_L1_scales_J=dict(accumulated_capacity_times_absolute_temperatures=state_L1,
            accumulated_substep_absolute_heat=heat_L1,endpoint_capacity_times_absolute_temperatures=endpoint_L1),
        maximum_single_substep_heat_state_residual_J=max_stored_mismatch,
        component_heat_sum_J=np.bincount(components,weights=heat,minlength=conductor.component_count).tolist(),
        variance_before_J_K=before,variance_after_J_K=after,variance_change_J_K=after-before,
        minimum_temperature_K=float(T.min()),maximum_temperature_K=float(T.max()),
        minimum_temperature_over_substeps_K=minT,maximum_temperature_over_substeps_K=maxT,
        minimum_reconstructed_corner_before_K=float(corners0.min()),
        minimum_reconstructed_corner_K=float(final_corners.min()),maximum_reconstructed_corner_K=float(final_corners.max()),
        reconstructed_corners_positive=bool(np.all(final_corners>0)),
        minimum_reconstructed_corner_over_substeps_K=mincorner,maximum_reconstructed_corner_over_substeps_K=maxcorner,
        reconstructed_corners_positive_all_substeps=bool(all_corners_positive),
        maximum_withheld_flux_fraction=max_withheld,stages=stage_aggregates,
        stages_are_aggregated=True,substep_stage_count=stage_count,
        no_temperature_clipping=True,no_heat_repair=True,no_implicit_solve=True)
    return ThermalStep(_readonly(T),_readonly(heat),diagnostics)
