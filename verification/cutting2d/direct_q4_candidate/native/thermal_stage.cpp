// Isolated arithmetic implementation of DirectQ4Thermal._stage.
// Binary64 only. Build forbids fast math, reassociation and FMA contraction.
// Endpoint i/j accumulations remain separate and visit the original edge order.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace py = pybind11;
using Float = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Index = py::array_t<std::int64_t, py::array::c_style>;

// These signed-zero semantics are verified against NumPy on the recorded host.
static inline double min_np(double a, double b) {
    if (std::isnan(a)) return a;
    if (std::isnan(b)) return b;
    if (a == 0. && b == 0.) return (std::signbit(a) || std::signbit(b)) ? -0. : 0.;
    return a < b ? a : b;
}
static inline double max_np(double a, double b) {
    if (std::isnan(a)) return a;
    if (std::isnan(b)) return b;
    if (a == 0. && b == 0.) return (std::signbit(a) && std::signbit(b)) ? -0. : 0.;
    return a > b ? a : b;
}
class FrozenStage {
    std::vector<std::int64_t> ii, jj, comp;
    std::vector<double> low_edge, correction_edge, C;
    std::size_t n, m, nc;
public:
    FrozenStage(Index i, Index j, Float low, Float correction, Float capacity,
                Index components, std::int64_t count) {
        if (i.ndim()!=1 || j.ndim()!=1 || low.ndim()!=1 || correction.ndim()!=1 ||
            capacity.ndim()!=1 || components.ndim()!=1 || count<=0)
            throw std::invalid_argument("One-dimensional stage arrays and positive component count required");
        n=capacity.size();m=i.size();nc=count;
        if (!n || j.size()!=m || low.size()!=m || correction.size()!=m || components.size()!=n)
            throw std::invalid_argument("Stage array shapes disagree");
        ii.assign(i.data(),i.data()+m);jj.assign(j.data(),j.data()+m);
        comp.assign(components.data(),components.data()+n);
        low_edge.assign(low.data(),low.data()+m);
        correction_edge.assign(correction.data(),correction.data()+m);
        C.assign(capacity.data(),capacity.data()+n);
        for (std::size_t a=0;a<n;++a)
            if (!(C[a]>0.) || !std::isfinite(C[a]) || comp[a]<0 || std::size_t(comp[a])>=nc)
                throw std::invalid_argument("Invalid capacity/component map");
        for (std::size_t e=0;e<m;++e)
            if (ii[e]<0 || jj[e]<0 || std::size_t(ii[e])>=n || std::size_t(jj[e])>=n ||
                !std::isfinite(low_edge[e]) || !std::isfinite(correction_edge[e]))
                throw std::invalid_argument("Invalid edge or coefficient");
    }
    py::dict compute(Float input, double dt) const {
        if(input.ndim()!=1 || std::size_t(input.size())!=n)
            throw std::invalid_argument("Stage temperature shape disagrees");
        py::array_t<double> low_heat(n), raw_heat(n), low(n), lower(n), upper(n),
            raw_flux(m), alpha(m), limited(m), correction_heat(n), heat(n), output(n);
        const double *u=input.data();
        auto *lh=low_heat.mutable_data(), *rh=raw_heat.mutable_data(), *lo=low.mutable_data();
        auto *lb=lower.mutable_data(), *ub=upper.mutable_data(), *rf=raw_flux.mutable_data();
        auto *al=alpha.mutable_data(), *lim=limited.mutable_data(), *ch=correction_heat.mutable_data();
        auto *h=heat.mutable_data(), *out=output.mutable_data();
        {
            py::gil_scoped_release release;
            // Each vector corresponds to ONE original np.bincount, initialized +0.
            std::vector<double> li(n,0.),lj(n,0.),ri(n,0.),rj(n,0.),
                pi(n,0.),pj(n,0.),ni(n,0.),nj(n,0.),ci(n,0.),cj(n,0.);
            std::vector<double> lcomp(nc,std::numeric_limits<double>::infinity()),
                ucomp(nc,-std::numeric_limits<double>::infinity()),rp(n),rm(n);
            for(std::size_t a=0;a<n;++a) {
                lcomp[comp[a]]=min_np(lcomp[comp[a]],u[a]);
                ucomp[comp[a]]=max_np(ucomp[comp[a]],u[a]);
            }
            for(std::size_t e=0;e<m;++e) {
                const auto i=ii[e],j=jj[e];
                const double difference=u[j]-u[i];
                const double low_flux=(-dt*low_edge[e])*difference;
                const double raw=(dt*correction_edge[e])*difference;
                rf[e]=raw;
                li[i]+=low_flux;lj[j]+=low_flux;
                ri[i]+=raw;rj[j]+=raw;
                pi[i]+=max_np(raw,0.);pj[j]+=max_np(-raw,0.);
                ni[i]+=min_np(raw,0.);nj[j]+=min_np(-raw,0.);
            }
            for(std::size_t a=0;a<n;++a) {
                lh[a]=li[a]-lj[a];rh[a]=ri[a]-rj[a];
                lo[a]=u[a]+lh[a]/C[a];
                lb[a]=lcomp[comp[a]];ub[a]=ucomp[comp[a]];
                const double positive=pi[a]+pj[a],negative=ni[a]+nj[a];
                const double qp=C[a]*max_np(ub[a]-lo[a],0.);
                const double qm=C[a]*min_np(lb[a]-lo[a],0.);
                rp[a]=min_np(positive>0. ? qp/positive : 1.,1.);
                rm[a]=min_np(negative<0. ? qm/negative : 1.,1.);
            }
            for(std::size_t e=0;e<m;++e) {
                const auto i=ii[e],j=jj[e];
                al[e]=rf[e]>0. ? min_np(rp[i],rm[j]) : min_np(rm[i],rp[j]);
                lim[e]=al[e]*rf[e];
                ci[i]+=lim[e];cj[j]+=lim[e];
            }
            for(std::size_t a=0;a<n;++a) {
                ch[a]=ci[a]-cj[a];h[a]=lh[a]+ch[a];out[a]=u[a]+h[a]/C[a];
            }
        }
        py::dict result;
        result["low_heat"]=low_heat;result["raw_heat"]=raw_heat;
        result["low"]=low;result["lower"]=lower;result["upper"]=upper;
        result["raw_flux"]=raw_flux;result["alpha"]=alpha;result["limited"]=limited;
        result["correction_heat"]=correction_heat;result["heat"]=heat;result["new"]=output;
        return result;
    }
};
PYBIND11_MODULE(_native_fct_stage,m) {
    m.doc()="Isolated original-order binary64 SSPRK2/FCT stage; no solver integration";
    py::class_<FrozenStage>(m,"FrozenStage")
        .def(py::init<Index,Index,Float,Float,Float,Index,std::int64_t>())
        .def("compute",&FrozenStage::compute);
}
