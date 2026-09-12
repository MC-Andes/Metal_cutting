#include <array>
#include <memory>
#include <pybind11/numpy.h>
#include <thread>
// SPDX-License-Identifier: MIT
// Finite isotropic Hencky elasticity / multiplicative Johnson-Cook plasticity.
// The material update is local and implicit. It supplies no global FE tangent.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using Mat = Eigen::Matrix3d;
using Vec = Eigen::Vector3d;
static constexpr const char* VERSION = "finite-jc-hencky-3d-stable-v2";
static constexpr const char* SCHEMA = "finite-jc-state-v1";

struct Failure : std::runtime_error {
    std::string code;
    Failure(std::string c, std::string message)
        : std::runtime_error(std::move(message)), code(std::move(c)) {}
};
static void require(bool condition, const std::string& code, const std::string& message) {
    if (!condition) throw Failure(code, message);
}
static double number(const py::dict& d, const char* key) {
    require(d.contains(key), "missing_field", std::string("Missing field: ")+key);
    double x = py::cast<double>(d[key]);
    require(std::isfinite(x), "nonfinite_input", std::string("Non-finite field: ")+key);
    return x;
}
static double optional_number(const py::dict& d, const char* key, double fallback) {
    return d.contains(key) ? number(d, key) : fallback;
}
static Mat matrix(py::handle value) {
    auto rows = py::cast<py::sequence>(value);
    Mat result;
    if (py::len(rows) == 9) {
        for (int i=0; i<9; ++i) result(i/3, i%3)=py::cast<double>(rows[i]);
    } else {
        require(py::len(rows)==3, "invalid_matrix", "Expected a nested 3 x 3 matrix or row-major list of 9 values");
        for (int i=0; i<3; ++i) {
            auto row=py::cast<py::sequence>(rows[i]);
            require(py::len(row)==3, "invalid_matrix", "Expected three columns in each matrix row");
            for (int j=0; j<3; ++j) result(i,j)=py::cast<double>(row[j]);
        }
    }
    require(result.allFinite(), "nonfinite_input", "Matrix contains a non-finite entry");
    return result;
}
static py::list matrix_list(const Mat& value) {
    py::list result;
    for (int i=0; i<3; ++i) {
        py::list row;
        for (int j=0; j<3; ++j) row.append(value(i,j));
        result.append(row);
    }
    return result;
}
static py::list voigt(const Mat& value) {
    py::list result;
    for (auto pair : std::vector<std::pair<int,int>>{{0,0},{1,1},{2,2},{0,1},{1,2},{0,2}})
        result.append(value(pair.first,pair.second));
    return result;
}
static Mat sym(const Mat& value) { return 0.5*(value+value.transpose()); }
static Mat dev(const Mat& value) { return value-value.trace()/3.0*Mat::Identity(); }
static double mises(const Mat& value) { return std::sqrt(1.5*dev(value).squaredNorm()); }

struct Material {
    double E,nu,G,K,rho,cp,A,B,C,n,m,rate0,T0,Tm,beta;
    explicit Material(const py::dict& d) {
        E=number(d,"YOUNG_MODULUS"); nu=number(d,"POISSON_RATIO");
        rho=number(d,"DENSITY"); cp=number(d,"SPECIFIC_HEAT");
        A=number(d,"JC_PARAMETER_A"); B=number(d,"JC_PARAMETER_B");
        C=number(d,"JC_PARAMETER_C"); n=number(d,"JC_PARAMETER_n");
        m=number(d,"JC_PARAMETER_m"); rate0=number(d,"REFERENCE_STRAIN_RATE");
        T0=number(d,"REFERENCE_TEMPERATURE"); Tm=number(d,"MELD_TEMPERATURE");
        beta=number(d,"TAYLOR_QUINNEY_COEFFICIENT");
        require(E>0 && nu>-1 && nu<0.5, "invalid_material", "Need E > 0 and -1 < nu < 0.5");
        require(rho>0 && cp>0, "invalid_material", "Reference density and specific heat must be positive");
        require(A>=0 && B>=0 && C>=0 && n>0 && m>0, "invalid_material", "Need A,B,C >= 0 and n,m > 0");
        require(rate0>0 && T0>=0 && Tm>T0 && beta>=0 && beta<=1, "invalid_material", "Invalid rate, temperature, or Taylor-Quinney parameters");
        G=E/(2*(1+nu)); K=E/(3*(1-2*nu));
        require(std::isfinite(G) && std::isfinite(K) && std::isfinite(rho*cp), "invalid_material", "Material modulus or volumetric heat capacity overflow");
    }
    double strength(double alpha, double rate, double temperature) const {
        require(alpha>=0 && rate>=0 && temperature>=0, "invalid_state", "Negative accumulated plastic strain, rate, or absolute temperature");
        const double theta=std::clamp((temperature-T0)/(Tm-T0),0.0,1.0);
        // Thermal softening exists independently of the heat conversion beta.
        const double result=(A+B*std::pow(alpha,n))*(1+C*std::log(std::max(1.0,rate/rate0)))*(1-std::pow(theta,m));
        require(std::isfinite(result) && result>=0, "nonfinite_strength", "Johnson-Cook strength became non-finite or negative");
        return result;
    }
};

struct Options {
    double abs_tol=1e-7, rel_tol=1e-11, dp_tol=2e-14;
    double min_j=1e-12, iso_tol=1e-9;
    double max_dp=std::numeric_limits<double>::infinity();
    int iterations=128, samples=128;
    explicit Options(const py::dict& d=py::dict()) {
        static const std::vector<std::string> valid={"absolute_tolerance_pa","relative_tolerance","plastic_increment_tolerance","minimum_jacobian","isochoric_tolerance","max_plastic_increment","max_iterations","bracket_samples"};
        for (auto item : d) {
            const auto key=py::cast<std::string>(item.first);
            require(std::find(valid.begin(),valid.end(),key)!=valid.end(),"invalid_options","Unknown option: "+key);
        }
        abs_tol=optional_number(d,"absolute_tolerance_pa",abs_tol);
        rel_tol=optional_number(d,"relative_tolerance",rel_tol);
        dp_tol=optional_number(d,"plastic_increment_tolerance",dp_tol);
        min_j=optional_number(d,"minimum_jacobian",min_j);
        iso_tol=optional_number(d,"isochoric_tolerance",iso_tol);
        max_dp=optional_number(d,"max_plastic_increment",max_dp);
        iterations=d.contains("max_iterations") ? py::cast<int>(d["max_iterations"]) : iterations;
        samples=d.contains("bracket_samples") ? py::cast<int>(d["bracket_samples"]) : samples;
        require(abs_tol>0 && rel_tol>0 && dp_tol>0 && min_j>0 && iso_tol>0 && max_dp>0,
            "invalid_options","All tolerances and maximum plastic increment must be positive");
        require(iterations>0 && iterations<=4096 && samples>=16 && samples<=65536,
            "invalid_options","Need 1 <= max_iterations <= 4096 and 16 <= bracket_samples <= 65536");
    }
    double tolerance(double scale) const { return abs_tol+rel_tol*std::abs(scale); }
};

struct Elastic {
    Mat F,Fp,Fe,e,tau,sigma;
    double J,qt,q,psi;
};
static Elastic elastic_legacy_arithmetic(const Mat& F, const Mat& Fp, const Material& p, const Options& o) {
    Elastic out;
    out.F=F; out.Fp=Fp; out.J=F.determinant();
    require(F.allFinite() && Fp.allFinite() && std::isfinite(out.J) && out.J>o.min_j,
        "invalid_jacobian","Total deformation must have a finite determinant above minimum_jacobian");
    const double Jp=Fp.determinant();
    require(std::isfinite(Jp) && Jp>0 && std::abs(Jp-1)<=o.iso_tol,
        "nonisochoric_plastic_state","Plastic deformation determinant is not one within isochoric_tolerance");
    Eigen::FullPivLU<Mat> factor(Fp.transpose());
    require(factor.isInvertible(),"singular_plastic_state","Plastic deformation is numerically singular");
    out.Fe=factor.solve(F.transpose()).transpose();
    const Mat b=sym(out.Fe*out.Fe.transpose());
    Eigen::SelfAdjointEigenSolver<Mat> eig(b);
    require(eig.info()==Eigen::Success && eig.eigenvalues().allFinite() && eig.eigenvalues().minCoeff()>0,
        "invalid_elastic_metric","Elastic left Cauchy-Green tensor is not numerically positive definite");
    out.e=eig.eigenvectors()*(0.5*eig.eigenvalues().array().log()).matrix().asDiagonal()*eig.eigenvectors().transpose();
    out.e=sym(out.e);
    out.tau=p.K*out.e.trace()*Mat::Identity()+2*p.G*dev(out.e);
    out.sigma=out.tau/out.J;
    out.qt=mises(out.tau); out.q=mises(out.sigma);
    out.psi=0.5*p.K*std::pow(out.e.trace(),2)+p.G*dev(out.e).squaredNorm();
    require(out.Fe.allFinite() && out.tau.allFinite() && out.sigma.allFinite() && std::isfinite(out.psi) && std::isfinite(out.qt) && std::isfinite(out.q),
        "nonfinite_elastic_response","Elastic response overflowed");
    return out;
}

// Isolated arithmetic candidate. Route limits are not acceptance tolerances.
// ||D||_F <= 1/2 implies singular values of I+D lie in [1/2,3/2].
// Thus b-I has eigenvalues >= -3/4, separated from log1p's singularity.
struct StableRoute { double d_norm=0,rcond=0;std::string route="stable_log1p"; };
static Elastic elastic(const Mat& F,const Mat& Fp,const Material& p,const Options& o,StableRoute* route=nullptr) {
    StableRoute local; if(!route)route=&local;
    Elastic out;out.F=F;out.Fp=Fp;out.J=F.determinant();
    require(F.allFinite() && Fp.allFinite() && std::isfinite(out.J) && out.J>o.min_j,
        "invalid_jacobian","Total deformation must have a finite determinant above minimum_jacobian");
    const double Jp=Fp.determinant();
    require(std::isfinite(Jp) && Jp>0 && std::abs(Jp-1)<=o.iso_tol,
        "nonisochoric_plastic_state","Plastic deformation determinant is not one within isochoric_tolerance");
    Eigen::FullPivLU<Mat> factor(Fp.transpose());
    require(factor.isInvertible(),"singular_plastic_state","Plastic deformation is numerically singular");
    const Mat D=factor.solve((F-Fp).transpose()).transpose();
    route->d_norm=D.norm();route->rcond=factor.rcond();
    if(!D.allFinite() || route->d_norm>0.5 || route->rcond<std::sqrt(std::numeric_limits<double>::epsilon())) {
        route->route="original_fallback_domain";return elastic_legacy_arithmetic(F,Fp,p,o);
    }
    const Mat delta=sym(D+D.transpose()+D*D.transpose());
    Eigen::SelfAdjointEigenSolver<Mat> eig(delta);
    if(eig.info()!=Eigen::Success || !eig.eigenvalues().allFinite() || eig.eigenvalues().minCoeff()<=-1) {
        route->route="original_fallback_eigensolver";return elastic_legacy_arithmetic(F,Fp,p,o);
    }
    const Vec logs=eig.eigenvalues().unaryExpr([](double x){return 0.5*std::log1p(x);});
    out.Fe=Mat::Identity()+D;
    out.e=sym(eig.eigenvectors()*logs.asDiagonal()*eig.eigenvectors().transpose());
    out.tau=p.K*out.e.trace()*Mat::Identity()+2*p.G*dev(out.e);
    out.sigma=out.tau/out.J;out.qt=mises(out.tau);out.q=mises(out.sigma);
    out.psi=0.5*p.K*std::pow(out.e.trace(),2)+p.G*dev(out.e).squaredNorm();
    require(out.Fe.allFinite() && out.tau.allFinite() && out.sigma.allFinite() && std::isfinite(out.psi) && std::isfinite(out.qt) && std::isfinite(out.q),
        "nonfinite_elastic_response","Elastic response overflowed");
    return out;
}
static Mat stable_flow(const Eigen::SelfAdjointEigenSolver<Mat>& flow,double dp,bool* stable=nullptr) {
    const Vec exponent=dp*flow.eigenvalues();
    const bool near=exponent.allFinite() && exponent.cwiseAbs().maxCoeff()<=0.5;
    if(stable)*stable=near;
    if(!near)return flow.eigenvectors()*exponent.array().exp().matrix().asDiagonal()*flow.eigenvectors().transpose();
    return Mat::Identity()+flow.eigenvectors()*exponent.unaryExpr([](double x){return std::expm1(x);}).asDiagonal()*flow.eigenvectors().transpose();
}

struct State {
    Elastic el;
    double alpha=0,rate=0,T=0,initial_T=0;
    double wp=0,qp=0,qext=0,stored=0,thermal=0,mechanical=0,initial_psi=0;
};
static py::dict state_dict(const State& s) {
    py::dict out;
    out["schema"]=SCHEMA; out["kernel_version"]=VERSION;
    out["F"]=matrix_list(s.el.F); out["Fp"]=matrix_list(s.el.Fp);
    out["Fe"]=matrix_list(s.el.Fe); out["elastic_log_strain"]=matrix_list(s.el.e);
    out["tau"]=matrix_list(s.el.tau); out["sigma"]=matrix_list(s.el.sigma);
    out["tau_voigt"]=voigt(s.el.tau); out["sigma_voigt"]=voigt(s.el.sigma);
    out["J"]=s.el.J; out["q_cauchy"]=s.el.q; out["q_kirchhoff"]=s.el.qt;
    out["mean_stress_cauchy"]=s.el.sigma.trace()/3.0;
    out["pressure_cauchy"]=-s.el.sigma.trace()/3.0;
    out["alpha"]=s.alpha; out["alpha_dot"]=s.rate;
    out["temperature"]=s.T; out["initial_temperature"]=s.initial_T;
    out["elastic_energy"]=s.el.psi; out["initial_elastic_energy"]=s.initial_psi;
    out["plastic_work"]=s.wp; out["plastic_heat"]=s.qp;
    out["external_heat"]=s.qext; out["stored_energy"]=s.stored;
    out["thermal_energy"]=s.thermal; out["mechanical_work"]=s.mechanical;
    return out;
}
static void close_scalar(double supplied, double expected, double relative, double absolute, const char* field) {
    require(std::abs(supplied-expected)<=absolute+relative*std::max(std::abs(supplied),std::abs(expected)),
        "inconsistent_state",std::string("State field is inconsistent with its deformation/history: ")+field);
}
static State read_state(const py::dict& d, const Material& p, const Options& o) {
    require(d.contains("schema"),"incompatible_state","State schema required");
    require(py::cast<std::string>(d["schema"])==SCHEMA,"incompatible_state","Unsupported state schema");
    require(d.contains("kernel_version"),"incompatible_state","Constitutive arithmetic version required");
    require(py::cast<std::string>(d["kernel_version"])==VERSION,"incompatible_state","State belongs to another kernel version");
    require(d.contains("F") && d.contains("Fp"),"missing_field","State requires F and Fp");
    State s; s.el=elastic(matrix(d["F"]),matrix(d["Fp"]),p,o);
    s.alpha=number(d,"alpha"); s.rate=number(d,"alpha_dot");
    s.T=number(d,"temperature"); s.initial_T=number(d,"initial_temperature");
    s.wp=number(d,"plastic_work"); s.qp=number(d,"plastic_heat");
    s.qext=number(d,"external_heat"); s.stored=number(d,"stored_energy");
    s.thermal=number(d,"thermal_energy"); s.mechanical=number(d,"mechanical_work");
    s.initial_psi=optional_number(d,"initial_elastic_energy",s.el.psi+s.wp-s.mechanical);
    require(s.alpha>=0 && s.rate>=0 && s.T>=0 && s.initial_T>=0 && s.wp>=0 && s.qp>=0 && s.stored>=0 && s.initial_psi>=-1e-8,
        "invalid_state","Negative plastic history, stored energy, or absolute temperature");
    if (d.contains("elastic_energy")) close_scalar(number(d,"elastic_energy"),s.el.psi,1e-9,1e-6,"elastic_energy");
    if (d.contains("J")) close_scalar(number(d,"J"),s.el.J,1e-11,1e-13,"J");
    if (d.contains("sigma")) require((matrix(d["sigma"])-s.el.sigma).norm()<=1e-5+1e-9*s.el.sigma.norm(),
        "inconsistent_state","Stored Cauchy stress differs from stress derived from F and Fp");
    close_scalar(s.wp,s.stored+s.qp,1e-10,1e-6,"plastic_work partition");
    close_scalar(s.thermal,s.qp+s.qext,1e-10,1e-6,"thermal_energy partition");
    close_scalar(p.rho*p.cp*(s.T-s.initial_T),s.thermal,1e-9,1e-5+1e-12*p.rho*p.cp*std::max(s.T,s.initial_T),"temperature / thermal_energy");
    close_scalar(s.mechanical,s.el.psi-s.initial_psi+s.wp,1e-9,1e-6,"mechanical_work bookkeeping");
    return s;
}

static py::dict initialize_state(const py::dict& material, py::object F, py::object temperature) {
    const Material p(material); const Options o;
    State s; s.el=elastic(F.is_none()?Mat::Identity():matrix(F),Mat::Identity(),p,o);
    s.T=temperature.is_none()?p.T0:py::cast<double>(temperature); s.initial_T=s.T;
    require(std::isfinite(s.T) && s.T>=0,"invalid_temperature","Initial temperature must be finite and nonnegative");
    s.initial_psi=s.el.psi;
    const double Y=p.strength(0,0,s.T);
    require(s.el.qt<=s.el.J*Y+o.tolerance(std::max(s.el.qt,s.el.J*Y)),"plastic_initial_state",
        "initialize_state accepts an elastic prestrain only; establish plastic history by evaluate from the identity state");
    return state_dict(s);
}

static py::dict diagnostics_base() {
    py::dict d;
    d["kernel_version"]=VERSION;
    d["constitutive_model"]="3D multiplicative isochoric J2 with isotropic Hencky elasticity and Cauchy Johnson-Cook yield";
    d["energy_quadrature"]="backward endpoint: delta_Wp0=q_kirchhoff_new*delta_alpha; first order";
    d["mechanical_work_definition"]="sum(delta_elastic_energy + endpoint_delta_plastic_work); constitutive bookkeeping, not independent external work";
    d["root_scan_scope"]="sampled admissible interval and rate/thermal breakpoints; finite scan is not a proof of uniqueness between samples";
    d["substeps"]=1;
    return d;
}

static py::dict evaluate(py::object F, const py::dict& old_dict, const py::dict& material,
                         double dt, double external_heat, const py::dict& options) {
    py::dict out, diag=diagnostics_base();
    // A fresh deepcopy on failure is essential: the caller owns the input state.
    auto copy_input=[&]() { return py::module_::import("copy").attr("deepcopy")(old_dict); };
    try {
        const Material p(material); const Options o(options); const State old=read_state(old_dict,p,o);
        require(std::isfinite(dt) && dt>0,"invalid_time_step","Time increment must be finite and positive");
        require(std::isfinite(external_heat),"invalid_heat_input","External heat density must be finite");
        State next=old;
        const Elastic trial=elastic(matrix(F),old.el.Fp,p,o);
        const double Tbase=old.T+external_heat/(p.rho*p.cp);
        require(std::isfinite(Tbase) && Tbase>=0,"invalid_temperature","External heat produces a negative or non-finite absolute temperature");
        const double qt=trial.qt, J=trial.J, upper=qt/(3*p.G);
        const double threshold=J*p.strength(old.alpha,0,Tbase);
        const double stress_tol=o.tolerance(std::max(qt,threshold));
        double dp=0, residual=qt-threshold;
        int iterations=0, root_count=0;
        double bracket_width=0, max_residual_increase=0;
        std::string branch="elastic";
        diag["trial_q_kirchhoff_pa"]=qt; diag["trial_static_strength_cauchy_pa"]=threshold/J;
        diag["stress_tolerance_pa"]=stress_tol;
        diag["max_plastic_increment"]=std::isfinite(o.max_dp)?py::cast(o.max_dp):py::none();
        diag["external_heat_increment_j_m3_reference"]=external_heat;
        if (residual>stress_tol) {
            branch="plastic";
            auto residual_at=[&](double x) {
                // x lies on [0, q_trial/(3G)]; no stress or plastic clipping.
                // The analytic interval endpoint has exactly zero deviator.
                // Evaluate that identity directly instead of retaining the
                // rounding error of a multiply after the defining division.
                const double qend=(x==upper)?0.0:qt-3*p.G*x;
                const double T=Tbase+p.beta*qend*x/(p.rho*p.cp);
                require(std::isfinite(qend) && qend>=-stress_tol && std::isfinite(T) && T>=0,
                    "invalid_local_trial","Local trial has invalid temperature or equivalent Kirchhoff stress");
                return qend-J*p.strength(old.alpha+x,x/dt,T);
            };
            // Scan the entire admissible interval, including constitutive kinks.
            // The explicit monotonicity rejection avoids choosing an arbitrary
            // softening root; it is deliberately more conservative than Newton.
            std::vector<double> xs;
            for (int i=0;i<=o.samples;++i) xs.push_back(upper*(double(i)/o.samples));
            const double rate_kink=dt*p.rate0;
            if (rate_kink>0 && rate_kink<upper) xs.push_back(rate_kink);
            if (p.beta>0) {
                for (double Tk : {p.T0,p.Tm}) {
                    const double c=(Tk-Tbase)*p.rho*p.cp/p.beta;
                    const double discriminant=qt*qt-12*p.G*c;
                    if (c>=0 && discriminant>=0) {
                        const double root=std::sqrt(discriminant);
                        // Stable lower quadratic root avoids cancellation.
                        const double lo=(qt+root)>0 ? 2*c/(qt+root) : 0;
                        const double hi=(qt+root)/(6*p.G);
                        if(lo>0 && lo<upper) xs.push_back(lo);
                        if(hi>0 && hi<upper) xs.push_back(hi);
                    }
                }
            }
            std::sort(xs.begin(),xs.end()); xs.erase(std::unique(xs.begin(),xs.end()),xs.end());
            std::vector<double> rs; rs.reserve(xs.size());
            for (double x : xs) rs.push_back(residual_at(x));
            bool nonmonotonic=false;
            double a=0,b=upper,fa=rs.front(),fb=rs.back();
            for (std::size_t i=1;i<xs.size();++i) {
                const double increase=rs[i]-rs[i-1];
                max_residual_increase=std::max(max_residual_increase,increase);
                if(increase>10*stress_tol) nonmonotonic=true;
                if (rs[i-1]>0 && rs[i]<=0) { ++root_count; a=xs[i-1]; b=xs[i]; fa=rs[i-1]; fb=rs[i]; }
                else if(rs[i-1]<0 && rs[i]>0) ++root_count;
            }
            diag["sampled_root_count"]=root_count;
            diag["max_sampled_residual_increase_pa"]=max_residual_increase;
            diag["bracket_samples_evaluated"]=xs.size();
            require(root_count<=1,"multiple_roots","Several consistency roots detected; reduce the step with an explicitly specified deformation/heat path");
            require(!nonmonotonic,"nonmonotonic_residual","Non-monotonic local consistency residual detected; no arbitrary softening branch is selected");
            require(rs.front()>0 && rs.back()<=stress_tol,"unbracketed_return","Could not bracket plastic consistency within the admissible interval");
            if(root_count==0 && std::abs(rs.back())<=stress_tol) {
                a=xs[xs.size()-2]; b=upper; fa=rs[rs.size()-2]; fb=rs.back(); root_count=1;
            }
            require(root_count==1,"unbracketed_return","No unique bracketed plastic consistency root");
            if(std::abs(fb)<=stress_tol) { dp=b; residual=fb; }
            else {
                bool converged=false;
                for(iterations=1;iterations<=o.iterations;++iterations) {
                    dp=0.5*(a+b); residual=residual_at(dp);
                    if(std::abs(residual)<=stress_tol) { converged=true; break; }
                    if(residual>0) { a=dp; fa=residual; } else { b=dp; fb=residual; }
                    // Width alone cannot certify consistency. Report failure
                    // if floating point resolution prevents a stress solution.
                    if(b-a<=o.dp_tol*std::max(1.0,upper)) {
                        const double candidate=(std::abs(fa)<std::abs(fb))?a:b;
                        const double candidate_r=(std::abs(fa)<std::abs(fb))?fa:fb;
                        if(std::abs(candidate_r)<=stress_tol) { dp=candidate;residual=candidate_r;converged=true; }
                        if(converged) break;
                        if(std::nextafter(a,b)>=b) break;
                    }
                }
                require(converged,"local_nonconvergence","Bracketed return failed the stress residual tolerance");
            }
            bracket_width=b-a;
            diag["plastic_increment"]=dp;
            require(dp<=o.max_dp,"plastic_increment_limit","Plastic increment exceeds max_plastic_increment; no update has been committed");
            Eigen::FullPivLU<Mat> fe_factor(trial.Fe.transpose());
            require(fe_factor.isInvertible(),"singular_elastic_state","Elastic deformation is numerically singular");
            const Mat mandel=trial.Fe.transpose()*trial.tau*fe_factor.inverse();
            const double asymmetry=(mandel-mandel.transpose()).norm();
            require(asymmetry<=1e-5+1e-9*mandel.norm(),"nonsymmetric_mandel","Isotropic elastic Mandel stress is not numerically symmetric");
            const Mat sm=sym(mandel); const double qm=mises(sm);
            require(qm>0,"zero_plastic_direction","Plastic flow has zero trial Mandel equivalent stress");
            const Mat N=1.5*dev(sm)/qm;
            Eigen::SelfAdjointEigenSolver<Mat> flow(N);
            require(flow.info()==Eigen::Success,"flow_eigensolver_failure","Cannot compute the isochoric plastic exponential");
            const Mat expflow=stable_flow(flow,dp);
            const Mat Fp=expflow*old.el.Fp;
            next.el=elastic(trial.F,Fp,p,o);
            diag["mandel_asymmetry_pa"]=asymmetry;
            diag["flow_trace"]=N.trace();
            diag["radial_stress_error_pa"]=next.el.qt-(qt-3*p.G*dp);
            require(std::abs(next.el.qt-(qt-3*p.G*dp))<=20*stress_tol,
                "radial_mapping_inconsistency","Reconstructed stress differs from the solved radial stress");
        } else {
            next.el=trial;
        }
        const double wp=next.el.qt*dp, qp=p.beta*wp;
        next.alpha=old.alpha+dp; next.rate=dp/dt;
        next.T=Tbase+qp/(p.rho*p.cp);
        next.wp=old.wp+wp; next.qp=old.qp+qp; next.qext=old.qext+external_heat;
        next.stored=old.stored+(1-p.beta)*wp; next.thermal=old.thermal+qp+external_heat;
        next.mechanical=old.mechanical+next.el.psi-old.el.psi+wp;
        require(std::isfinite(next.T) && std::isfinite(next.wp) && std::isfinite(next.mechanical),"nonfinite_history","State history overflowed");
        const double Y=p.strength(next.alpha,next.rate,next.T);
        const double final_residual=next.el.qt-next.el.J*Y;
        if(dp>0) require(std::abs(final_residual)<=30*stress_tol,"final_consistency_failure","Reconstructed final state fails plastic consistency");
        else require(final_residual<=stress_tol,"final_consistency_failure","Elastic state lies outside the current yield surface");
        // Verify all cumulative identities before exposing the new state.
        py::dict committed_candidate=state_dict(next);
        read_state(committed_candidate,p,o);
        diag["branch"]=branch; diag["plastic_increment"]=dp;
        diag["sampled_root_count"]=root_count; diag["iterations"]=iterations;
        diag["bracket_width"]=bracket_width; diag["root_residual_kirchhoff_pa"]=residual;
        diag["yield_residual_kirchhoff_pa"]=final_residual;
        diag["yield_strength_cauchy_pa"]=Y;
        diag["plastic_work_increment_j_m3_reference"]=wp;
        diag["plastic_heat_increment_j_m3_reference"]=qp;
        diag["elastic_energy_increment_j_m3_reference"]=next.el.psi-old.el.psi;
        diag["plastic_jacobian"]=next.el.Fp.determinant();
        diag["code"]="ok";
        out["success"]=true; out["state"]=committed_candidate; out["diagnostics"]=diag;
    } catch (const Failure& e) {
        diag["code"]=e.code; diag["message"]=e.what();
        out["success"]=false; out["state"]=copy_input(); out["diagnostics"]=diag;
    } catch (const std::exception& e) {
        diag["code"]="invalid_input_or_internal_exception"; diag["message"]=e.what();
        out["success"]=false; out["state"]=copy_input(); out["diagnostics"]=diag;
    }
    return out;
}

static py::dict deepcopy(const py::dict& value) {
    return py::cast<py::dict>(py::module_::import("copy").attr("deepcopy")(value));
}
class Session {
    py::dict material_,options_,committed_,pending_;
    bool has_pending_=false;
public:
    Session(py::dict material, py::object state, py::dict options)
      : material_(deepcopy(material)),options_(deepcopy(options)) {
        const Material p(material_); const Options o(options_);
        committed_=state.is_none()?initialize_state(material_,py::none(),py::none()):deepcopy(py::cast<py::dict>(state));
        read_state(committed_,p,o);
    }
    py::dict trial(py::object F,double dt,double external_heat) {
        // Every trial starts from committed history, including after a failed
        // trial. A failed trial invalidates any previously pending candidate.
        has_pending_=false; pending_=py::dict();
        py::dict result=evaluate(F,committed_,material_,dt,external_heat,options_);
        if(py::cast<bool>(result["success"])) {
            pending_=deepcopy(py::cast<py::dict>(result["state"])); has_pending_=true;
        }
        return result;
    }
    py::dict commit() {
        require(has_pending_,"no_successful_trial","No successful pending trial to commit");
        committed_=deepcopy(pending_); pending_=py::dict();has_pending_=false;
        return deepcopy(committed_);
    }
    py::dict rollback() { pending_=py::dict();has_pending_=false;return deepcopy(committed_); }
    py::dict state() const { return deepcopy(committed_); }
    bool has_pending_trial() const { return has_pending_; }
    Session clone() const { return Session(material_,committed_,options_); }
    py::dict serialize() const {
        py::dict out;out["schema"]="finite-jc-session-v1";out["kernel_version"]=VERSION;
        out["material"]=deepcopy(material_);out["options"]=deepcopy(options_);out["state"]=deepcopy(committed_);
        return out;
    }
};
static Session restore_session(py::dict serialized) {
    require(serialized.contains("schema") && py::cast<std::string>(serialized["schema"])=="finite-jc-session-v1",
        "incompatible_session","Unsupported session serialization schema");
    require(serialized.contains("kernel_version") && py::cast<std::string>(serialized["kernel_version"])==VERSION,
        "incompatible_session","Serialized session belongs to another kernel version");
    return Session(py::cast<py::dict>(serialized["material"]),serialized["state"],py::cast<py::dict>(serialized["options"]));
}

struct NativeDiagnostics {
    double bracket_samples_evaluated=std::numeric_limits<double>::quiet_NaN();
    double bracket_width=std::numeric_limits<double>::quiet_NaN();
    std::string branch;
    std::string code;
    double elastic_energy_increment_j_m3_reference=std::numeric_limits<double>::quiet_NaN();
    double external_heat_increment_j_m3_reference=std::numeric_limits<double>::quiet_NaN();
    double flow_trace=std::numeric_limits<double>::quiet_NaN();
    double iterations=std::numeric_limits<double>::quiet_NaN();
    double mandel_asymmetry_pa=std::numeric_limits<double>::quiet_NaN();
    double max_plastic_increment=std::numeric_limits<double>::quiet_NaN();
    double max_sampled_residual_increase_pa=std::numeric_limits<double>::quiet_NaN();
    double plastic_heat_increment_j_m3_reference=std::numeric_limits<double>::quiet_NaN();
    double plastic_increment=std::numeric_limits<double>::quiet_NaN();
    double plastic_jacobian=std::numeric_limits<double>::quiet_NaN();
    double plastic_work_increment_j_m3_reference=std::numeric_limits<double>::quiet_NaN();
    double radial_stress_error_pa=std::numeric_limits<double>::quiet_NaN();
    double root_residual_kirchhoff_pa=std::numeric_limits<double>::quiet_NaN();
    double sampled_root_count=std::numeric_limits<double>::quiet_NaN();
    double stress_tolerance_pa=std::numeric_limits<double>::quiet_NaN();
    double trial_q_kirchhoff_pa=std::numeric_limits<double>::quiet_NaN();
    double trial_static_strength_cauchy_pa=std::numeric_limits<double>::quiet_NaN();
    double yield_residual_kirchhoff_pa=std::numeric_limits<double>::quiet_NaN();
    double yield_strength_cauchy_pa=std::numeric_limits<double>::quiet_NaN();
};

static State validate_native(const State& stored,const Material& p,const Options& o) {
    State s=stored;
    s.el=elastic(stored.el.F,stored.el.Fp,p,o);
    // Match number() checks in the serialized read_state path explicitly;
    // an infinity must not pass a relative check through inf <= inf.
    const std::pair<const char*,double> fields[]={
        {"alpha",s.alpha},{"alpha_dot",s.rate},{"temperature",s.T},{"initial_temperature",s.initial_T},
        {"plastic_work",s.wp},{"plastic_heat",s.qp},{"external_heat",s.qext},{"stored_energy",s.stored},
        {"thermal_energy",s.thermal},{"mechanical_work",s.mechanical},{"initial_elastic_energy",s.initial_psi},
        {"elastic_energy",stored.el.psi},{"J",stored.el.J}};
    for(const auto& field:fields)if(!std::isfinite(field.second))throw Failure("nonfinite_input",std::string("Non-finite field: ")+field.first);
    require(s.alpha>=0 && s.rate>=0 && s.T>=0 && s.initial_T>=0 && s.wp>=0 && s.qp>=0 && s.stored>=0 && s.initial_psi>=-1e-8,
        "invalid_state","Negative plastic history, stored energy, or absolute temperature");
    close_scalar(stored.el.psi,s.el.psi,1e-9,1e-6,"elastic_energy");
    close_scalar(stored.el.J,s.el.J,1e-11,1e-13,"J");
    require((stored.el.sigma-s.el.sigma).norm()<=1e-5+1e-9*s.el.sigma.norm(),
        "inconsistent_state","Stored Cauchy stress differs from stress derived from F and Fp");
    close_scalar(s.wp,s.stored+s.qp,1e-10,1e-6,"plastic_work partition");
    close_scalar(s.thermal,s.qp+s.qext,1e-10,1e-6,"thermal_energy partition");
    close_scalar(p.rho*p.cp*(s.T-s.initial_T),s.thermal,1e-9,1e-5+1e-12*p.rho*p.cp*std::max(s.T,s.initial_T),"temperature / thermal_energy");
    close_scalar(s.mechanical,s.el.psi-s.initial_psi+s.wp,1e-9,1e-6,"mechanical_work bookkeeping");
    return s;
}
static State native_evaluate(const Mat& F,const State& prior,const Material& p,const Options& o,double dt,double external_heat,NativeDiagnostics& diag) {
const State old=validate_native(prior,p,o);
        State next=old;
        const Elastic trial=elastic(F,old.el.Fp,p,o);
        const double Tbase=old.T+external_heat/(p.rho*p.cp);
        require(std::isfinite(Tbase) && Tbase>=0,"invalid_temperature","External heat produces a negative or non-finite absolute temperature");
        const double qt=trial.qt, J=trial.J, upper=qt/(3*p.G);
        const double threshold=J*p.strength(old.alpha,0,Tbase);
        const double stress_tol=o.tolerance(std::max(qt,threshold));
        double dp=0, residual=qt-threshold;
        int iterations=0, root_count=0;
        double bracket_width=0, max_residual_increase=0;
        std::string branch="elastic";
        diag.trial_q_kirchhoff_pa=qt; diag.trial_static_strength_cauchy_pa=threshold/J;
        diag.stress_tolerance_pa=stress_tol;
        diag.max_plastic_increment=o.max_dp;
        diag.external_heat_increment_j_m3_reference=external_heat;
        if (residual>stress_tol) {
            branch="plastic";
            auto residual_at=[&](double x) {
                // x lies on [0, q_trial/(3G)]; no stress or plastic clipping.
                // The analytic interval endpoint has exactly zero deviator.
                // Evaluate that identity directly instead of retaining the
                // rounding error of a multiply after the defining division.
                const double qend=(x==upper)?0.0:qt-3*p.G*x;
                const double T=Tbase+p.beta*qend*x/(p.rho*p.cp);
                require(std::isfinite(qend) && qend>=-stress_tol && std::isfinite(T) && T>=0,
                    "invalid_local_trial","Local trial has invalid temperature or equivalent Kirchhoff stress");
                return qend-J*p.strength(old.alpha+x,x/dt,T);
            };
            // Scan the entire admissible interval, including constitutive kinks.
            // The explicit monotonicity rejection avoids choosing an arbitrary
            // softening root; it is deliberately more conservative than Newton.
            std::vector<double> xs;
            for (int i=0;i<=o.samples;++i) xs.push_back(upper*(double(i)/o.samples));
            const double rate_kink=dt*p.rate0;
            if (rate_kink>0 && rate_kink<upper) xs.push_back(rate_kink);
            if (p.beta>0) {
                for (double Tk : {p.T0,p.Tm}) {
                    const double c=(Tk-Tbase)*p.rho*p.cp/p.beta;
                    const double discriminant=qt*qt-12*p.G*c;
                    if (c>=0 && discriminant>=0) {
                        const double root=std::sqrt(discriminant);
                        // Stable lower quadratic root avoids cancellation.
                        const double lo=(qt+root)>0 ? 2*c/(qt+root) : 0;
                        const double hi=(qt+root)/(6*p.G);
                        if(lo>0 && lo<upper) xs.push_back(lo);
                        if(hi>0 && hi<upper) xs.push_back(hi);
                    }
                }
            }
            std::sort(xs.begin(),xs.end()); xs.erase(std::unique(xs.begin(),xs.end()),xs.end());
            std::vector<double> rs; rs.reserve(xs.size());
            for (double x : xs) rs.push_back(residual_at(x));
            bool nonmonotonic=false;
            double a=0,b=upper,fa=rs.front(),fb=rs.back();
            for (std::size_t i=1;i<xs.size();++i) {
                const double increase=rs[i]-rs[i-1];
                max_residual_increase=std::max(max_residual_increase,increase);
                if(increase>10*stress_tol) nonmonotonic=true;
                if (rs[i-1]>0 && rs[i]<=0) { ++root_count; a=xs[i-1]; b=xs[i]; fa=rs[i-1]; fb=rs[i]; }
                else if(rs[i-1]<0 && rs[i]>0) ++root_count;
            }
            diag.sampled_root_count=root_count;
            diag.max_sampled_residual_increase_pa=max_residual_increase;
            diag.bracket_samples_evaluated=xs.size();
            require(root_count<=1,"multiple_roots","Several consistency roots detected; reduce the step with an explicitly specified deformation/heat path");
            require(!nonmonotonic,"nonmonotonic_residual","Non-monotonic local consistency residual detected; no arbitrary softening branch is selected");
            require(rs.front()>0 && rs.back()<=stress_tol,"unbracketed_return","Could not bracket plastic consistency within the admissible interval");
            if(root_count==0 && std::abs(rs.back())<=stress_tol) {
                a=xs[xs.size()-2]; b=upper; fa=rs[rs.size()-2]; fb=rs.back(); root_count=1;
            }
            require(root_count==1,"unbracketed_return","No unique bracketed plastic consistency root");
            if(std::abs(fb)<=stress_tol) { dp=b; residual=fb; }
            else {
                bool converged=false;
                for(iterations=1;iterations<=o.iterations;++iterations) {
                    dp=0.5*(a+b); residual=residual_at(dp);
                    if(std::abs(residual)<=stress_tol) { converged=true; break; }
                    if(residual>0) { a=dp; fa=residual; } else { b=dp; fb=residual; }
                    // Width alone cannot certify consistency. Report failure
                    // if floating point resolution prevents a stress solution.
                    if(b-a<=o.dp_tol*std::max(1.0,upper)) {
                        const double candidate=(std::abs(fa)<std::abs(fb))?a:b;
                        const double candidate_r=(std::abs(fa)<std::abs(fb))?fa:fb;
                        if(std::abs(candidate_r)<=stress_tol) { dp=candidate;residual=candidate_r;converged=true; }
                        if(converged) break;
                        if(std::nextafter(a,b)>=b) break;
                    }
                }
                require(converged,"local_nonconvergence","Bracketed return failed the stress residual tolerance");
            }
            bracket_width=b-a;
            diag.plastic_increment=dp;
            require(dp<=o.max_dp,"plastic_increment_limit","Plastic increment exceeds max_plastic_increment; no update has been committed");
            Eigen::FullPivLU<Mat> fe_factor(trial.Fe.transpose());
            require(fe_factor.isInvertible(),"singular_elastic_state","Elastic deformation is numerically singular");
            const Mat mandel=trial.Fe.transpose()*trial.tau*fe_factor.inverse();
            const double asymmetry=(mandel-mandel.transpose()).norm();
            require(asymmetry<=1e-5+1e-9*mandel.norm(),"nonsymmetric_mandel","Isotropic elastic Mandel stress is not numerically symmetric");
            const Mat sm=sym(mandel); const double qm=mises(sm);
            require(qm>0,"zero_plastic_direction","Plastic flow has zero trial Mandel equivalent stress");
            const Mat N=1.5*dev(sm)/qm;
            Eigen::SelfAdjointEigenSolver<Mat> flow(N);
            require(flow.info()==Eigen::Success,"flow_eigensolver_failure","Cannot compute the isochoric plastic exponential");
            const Mat expflow=stable_flow(flow,dp);
            const Mat Fp=expflow*old.el.Fp;
            next.el=elastic(trial.F,Fp,p,o);
            diag.mandel_asymmetry_pa=asymmetry;
            diag.flow_trace=N.trace();
            diag.radial_stress_error_pa=next.el.qt-(qt-3*p.G*dp);
            require(std::abs(next.el.qt-(qt-3*p.G*dp))<=20*stress_tol,
                "radial_mapping_inconsistency","Reconstructed stress differs from the solved radial stress");
        } else {
            next.el=trial;
        }
        const double wp=next.el.qt*dp, qp=p.beta*wp;
        next.alpha=old.alpha+dp; next.rate=dp/dt;
        next.T=Tbase+qp/(p.rho*p.cp);
        next.wp=old.wp+wp; next.qp=old.qp+qp; next.qext=old.qext+external_heat;
        next.stored=old.stored+(1-p.beta)*wp; next.thermal=old.thermal+qp+external_heat;
        next.mechanical=old.mechanical+next.el.psi-old.el.psi+wp;
        require(std::isfinite(next.T) && std::isfinite(next.wp) && std::isfinite(next.mechanical),"nonfinite_history","State history overflowed");
        const double Y=p.strength(next.alpha,next.rate,next.T);
        const double final_residual=next.el.qt-next.el.J*Y;
        if(dp>0) require(std::abs(final_residual)<=30*stress_tol,"final_consistency_failure","Reconstructed final state fails plastic consistency");
        else require(final_residual<=stress_tol,"final_consistency_failure","Elastic state lies outside the current yield surface");
        validate_native(next,p,o);
        diag.branch=branch; diag.plastic_increment=dp;
        diag.sampled_root_count=root_count; diag.iterations=iterations;
        diag.bracket_width=bracket_width; diag.root_residual_kirchhoff_pa=residual;
        diag.yield_residual_kirchhoff_pa=final_residual;
        diag.yield_strength_cauchy_pa=Y;
        diag.plastic_work_increment_j_m3_reference=wp;
        diag.plastic_heat_increment_j_m3_reference=qp;
        diag.elastic_energy_increment_j_m3_reference=next.el.psi-old.el.psi;
        diag.plastic_jacobian=next.el.Fp.determinant();
        diag.code="ok";
        return next;
}
static py::dict diagnostics_dict(const NativeDiagnostics& d) { auto out=diagnostics_base();
if(!std::isnan(d.bracket_samples_evaluated)) out["bracket_samples_evaluated"]=d.bracket_samples_evaluated;
if(!std::isnan(d.bracket_width)) out["bracket_width"]=d.bracket_width;
out["branch"]=d.branch;
out["code"]=d.code;
if(!std::isnan(d.elastic_energy_increment_j_m3_reference)) out["elastic_energy_increment_j_m3_reference"]=d.elastic_energy_increment_j_m3_reference;
if(!std::isnan(d.external_heat_increment_j_m3_reference)) out["external_heat_increment_j_m3_reference"]=d.external_heat_increment_j_m3_reference;
if(!std::isnan(d.flow_trace)) out["flow_trace"]=d.flow_trace;
if(!std::isnan(d.iterations)) out["iterations"]=d.iterations;
if(!std::isnan(d.mandel_asymmetry_pa)) out["mandel_asymmetry_pa"]=d.mandel_asymmetry_pa;
out["max_plastic_increment"]=std::isfinite(d.max_plastic_increment)?py::cast(d.max_plastic_increment):py::none();
if(!std::isnan(d.max_sampled_residual_increase_pa)) out["max_sampled_residual_increase_pa"]=d.max_sampled_residual_increase_pa;
if(!std::isnan(d.plastic_heat_increment_j_m3_reference)) out["plastic_heat_increment_j_m3_reference"]=d.plastic_heat_increment_j_m3_reference;
if(!std::isnan(d.plastic_increment)) out["plastic_increment"]=d.plastic_increment;
if(!std::isnan(d.plastic_jacobian)) out["plastic_jacobian"]=d.plastic_jacobian;
if(!std::isnan(d.plastic_work_increment_j_m3_reference)) out["plastic_work_increment_j_m3_reference"]=d.plastic_work_increment_j_m3_reference;
if(!std::isnan(d.radial_stress_error_pa)) out["radial_stress_error_pa"]=d.radial_stress_error_pa;
if(!std::isnan(d.root_residual_kirchhoff_pa)) out["root_residual_kirchhoff_pa"]=d.root_residual_kirchhoff_pa;
if(!std::isnan(d.sampled_root_count)) out["sampled_root_count"]=d.sampled_root_count;
if(!std::isnan(d.stress_tolerance_pa)) out["stress_tolerance_pa"]=d.stress_tolerance_pa;
if(!std::isnan(d.trial_q_kirchhoff_pa)) out["trial_q_kirchhoff_pa"]=d.trial_q_kirchhoff_pa;
if(!std::isnan(d.trial_static_strength_cauchy_pa)) out["trial_static_strength_cauchy_pa"]=d.trial_static_strength_cauchy_pa;
if(!std::isnan(d.yield_residual_kirchhoff_pa)) out["yield_residual_kirchhoff_pa"]=d.yield_residual_kirchhoff_pa;
if(!std::isnan(d.yield_strength_cauchy_pa)) out["yield_strength_cauchy_pa"]=d.yield_strength_cauchy_pa;
return out; }

// Verification candidate: native histories and read-only NumPy views.
struct BatchData {
    std::vector<State> states;
    std::vector<NativeDiagnostics> diagnostics;
    std::vector<std::array<double,16>> tangent;
    std::vector<std::array<unsigned char,8>> tangent_branch_changes;
};

class NativeBatch {
    Material p_; Options o_;
    std::shared_ptr<BatchData> committed_,pending_;
    bool busy_=false;
    void available()const {require(!busy_,"batch_busy","Concurrent use of one batch is not supported");}
public:
    NativeBatch(py::dict material,py::list states,py::dict options):p_(material),o_(options),committed_(std::make_shared<BatchData>()) {
        require(py::len(states)>0,"invalid_shape","Positive state count required");
        for(auto item:states) {
            State s=read_state(py::cast<py::dict>(item),p_,o_);
            require(s.T>0 && s.T<p_.Tm,"solid_temperature_limit","Committed temperature leaves 0 < T < Tm");
            committed_->states.push_back(s);
        }
    }
    py::dict trial_impl(py::array_t<double,py::array::c_style> deformation,double dt,
                   py::array_t<double,py::array::c_style> heat,int threads,bool tangent,double fd_step) {
        available();pending_.reset();
        const auto f=deformation.request(),h=heat.request();const size_t n=committed_->states.size();
        require(f.ndim==3 && f.shape[0]==(long)n && f.shape[1]==3 && f.shape[2]==3,"invalid_shape","F must be contiguous N by 3 by 3");
        require(h.ndim==1 && h.shape[0]==(long)n,"invalid_shape","heat density must be contiguous N");
        require(std::isfinite(dt)&&dt>0,"invalid_time_step","Positive finite time increment required");
        require(threads>0 && threads<=32,"invalid_threads","Need 1 to 32 threads");
        require(!tangent||(std::isfinite(fd_step)&&fd_step>0),"invalid_fd_step","Positive finite relative finite-difference step required");
        const double* fp=static_cast<const double*>(f.ptr);const double* hp=static_cast<const double*>(h.ptr);
        auto candidate=std::make_shared<BatchData>();candidate->states.resize(n);candidate->diagnostics.resize(n);
        if(tangent){candidate->tangent.resize(n);candidate->tangent_branch_changes.resize(n);}
        std::vector<std::string> codes(n),messages(n);
        struct Busy {bool& value;Busy(bool& v):value(v){value=true;}~Busy(){value=false;}} in_flight(busy_);
        {
            py::gil_scoped_release release;
            auto work=[&](size_t first,size_t last) {
                for(size_t i=first;i<last;++i) {
                    try {
                        Mat F=Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>>(fp+9*i);
                        require(F.allFinite(),"nonfinite_input","F must be finite");
                        require(std::isfinite(hp[i]),"invalid_heat_input","External heat density must be finite");
                        const double base=committed_->states[i].T+hp[i]/(p_.rho*p_.cp);
                        require(base>0 && base<p_.Tm,"solid_temperature_limit","External heat leaves 0 < T < Tm");
                        candidate->states[i]=native_evaluate(F,committed_->states[i],p_,o_,dt,hp[i],candidate->diagnostics[i]);
                        require(candidate->states[i].T<p_.Tm,"solid_temperature_limit","Coupled plastic heating reaches melting");
                        if(tangent) {
                            require(F(2,2)==1 && F(2,0)==0 && F(2,1)==0 && F(0,2)==0 && F(1,2)==0,
                                    "tangent_requires_plane_strain","Tangent only covers Fzz=1 with zero transverse shear");
                            for(int k=0;k<2;++k)for(int l=0;l<2;++l) {
                                const int component=2*k+l;const double step=fd_step*std::max(1.0,std::abs(F(k,l)));
                                Mat P[2];
                                for(int side=0;side<2;++side) {
                                    Mat perturbed=F;perturbed(k,l)+=(side==0?step:-step);
                                    require(perturbed(k,l)!=F(k,l),"unresolved_fd_step","Finite-difference perturbation rounded to zero");
                                    NativeDiagnostics diagnostic;
                                    try {
                                        const State response=native_evaluate(perturbed,committed_->states[i],p_,o_,dt,hp[i],diagnostic);
                                        require(response.T<p_.Tm,"solid_temperature_limit","Tangent perturbation reaches melting");
                                        Eigen::FullPivLU<Mat> solve(perturbed);
                                        require(solve.isInvertible(),"singular_tangent_deformation","Tangent deformation is numerically singular");
                                        P[side]=solve.solve(response.el.tau.transpose()).transpose();
                                        candidate->tangent_branch_changes[i][2*component+side]=(diagnostic.branch!=candidate->diagnostics[i].branch);
                                    } catch(const Failure& e) {
                                        throw Failure(e.code,"Tangent perturbation component "+std::to_string(component)+" side "+std::to_string(side)+": "+e.what());
                                    }
                                }
                                const Mat derivative=(P[0]-P[1])/(2*step);
                                require(derivative.allFinite(),"nonfinite_tangent","Constitutive finite-difference tangent overflowed");
                                for(int j=0;j<2;++j)for(int a=0;a<2;++a)candidate->tangent[i][4*(2*j+a)+component]=derivative(j,a);
                            }
                        }
                    } catch(const Failure& e) {codes[i]=e.code;messages[i]=e.what();}
                      catch(const std::exception& e) {codes[i]="invalid_input_or_internal_exception";messages[i]=e.what();}
                }
            };
            if(threads==1)work(0,n);
            else {
                std::vector<std::thread> workers;
                try {
                    for(int j=0;j<threads;++j)workers.emplace_back(work,n*j/threads,n*(j+1)/threads);
                } catch(...) {for(auto& worker:workers)worker.join();throw;}
                for(auto& worker:workers)worker.join();
            }
        }
        py::dict result;
        for(size_t i=0;i<n;++i)if(!codes[i].empty()) {
            result["success"]=false;result["index"]=i;result["code"]=codes[i];result["message"]=messages[i];return result;
        }
        pending_=candidate;result["success"]=true;return result;
    }
    py::dict trial(py::array_t<double,py::array::c_style> F,double dt,py::array_t<double,py::array::c_style> heat,int threads) {
        return trial_impl(F,dt,heat,threads,false,0.);
    }
    py::dict trial_with_tangent(py::array_t<double,py::array::c_style> F,double dt,py::array_t<double,py::array::c_style> heat,int threads,double step) {
        return trial_impl(F,dt,heat,threads,true,step);
    }
    void commit() {available();require(bool(pending_),"no_successful_trial","No successful pending trial to commit");committed_=pending_;pending_.reset();}
    void reject() {available();pending_.reset();}
    py::list states(bool pending)const {
        available();require(!pending||bool(pending_),"no_successful_trial","No successful pending trial");
        py::list out;for(const auto& s:(pending?pending_:committed_)->states)out.append(state_dict(s));return out;
    }
    py::list diagnostics(bool pending)const {
        available();require(!pending||bool(pending_),"no_successful_trial","No successful pending trial");
        py::list out;for(const auto& d:(pending?pending_:committed_)->diagnostics)out.append(diagnostics_dict(d));return out;
    }
    py::dict fields(bool pending)const {
        available();require(!pending||bool(pending_),"no_successful_trial","No successful pending trial");
        auto data=pending?pending_:committed_;const long n=data->states.size();const auto& s=data->states[0];
        auto* owner=new std::shared_ptr<BatchData>(data);
        py::capsule capsule(owner,[](void* p){delete static_cast<std::shared_ptr<BatchData>*>(p);});
        py::dict out;
        auto tensor=[&](const char* key,const Mat& value){
            py::array a(py::dtype::of<double>(),{n,3L,3L},{long(sizeof(State)),long(sizeof(double)),long(3*sizeof(double))},value.data(),capsule);
            a.attr("setflags")(false);out[key]=a;
        };
        auto scalar=[&](const char* key,const double& value){
            py::array a(py::dtype::of<double>(),{n},{long(sizeof(State))},&value,capsule);
            a.attr("setflags")(false);out[key]=a;
        };
        tensor("F",s.el.F);tensor("Fp",s.el.Fp);tensor("Fe",s.el.Fe);tensor("elastic_log_strain",s.el.e);
        tensor("sigma",s.el.sigma);tensor("tau",s.el.tau);
        scalar("J",s.el.J);scalar("q_cauchy",s.el.q);scalar("q_kirchhoff",s.el.qt);
        scalar("alpha",s.alpha);scalar("alpha_dot",s.rate);scalar("T",s.T);scalar("initial_temperature",s.initial_T);
        scalar("elastic_energy",s.el.psi);scalar("initial_elastic_energy",s.initial_psi);
        scalar("plastic_work",s.wp);scalar("plastic_heat",s.qp);scalar("external_heat",s.qext);
        scalar("stored_energy",s.stored);scalar("thermal_energy",s.thermal);scalar("mechanical_work",s.mechanical);
        if(!data->tangent.empty()) {
            py::array a(py::dtype::of<double>(),{n,2L,2L,2L,2L},{128L,64L,32L,16L,8L},data->tangent.data(),capsule);
            a.attr("setflags")(false);out["tangent"]=a;
            py::array changes(py::dtype::of<unsigned char>(),{n,4L,2L},{8L,2L,1L},data->tangent_branch_changes.data(),capsule);
            changes.attr("setflags")(false);out["tangent_branch_changes"]=changes;
        }
        return out;
    }
    size_t native_state_bytes()const {
        available();auto bytes=[](const std::shared_ptr<BatchData>& x){return x?x->states.capacity()*sizeof(State)+x->diagnostics.capacity()*sizeof(NativeDiagnostics)+x->tangent.capacity()*128+x->tangent_branch_changes.capacity()*8:0;};
        return bytes(committed_)+bytes(pending_);
    }
};

PYBIND11_MODULE(direct_q4_finite_jc,m) {
    m.attr("__version__")=VERSION;
    m.def("initialize_state",&initialize_state,py::arg("material"),py::arg("F")=py::none(),py::arg("temperature")=py::none());
    py::register_exception<Failure>(m,"ConstitutiveError");
    py::class_<NativeBatch>(m,"NativeBatch")
      .def(py::init<py::dict,py::list,py::dict>(),py::arg("material"),py::arg("states"),py::arg("options")=py::dict())
      .def("trial",&NativeBatch::trial,py::arg("F"),py::arg("dt"),py::arg("heat_density"),py::arg("threads")=1)
      .def("trial_with_tangent",&NativeBatch::trial_with_tangent,py::arg("F"),py::arg("dt"),py::arg("heat_density"),py::arg("threads")=1,py::arg("fd_step")=2e-6)
      .def("commit",&NativeBatch::commit).def("reject",&NativeBatch::reject)
      .def("states",&NativeBatch::states,py::arg("pending")=false)
      .def("fields",&NativeBatch::fields,py::arg("pending")=false)
      .def("diagnostics",&NativeBatch::diagnostics,py::arg("pending")=false)
      .def("native_state_bytes",&NativeBatch::native_state_bytes);
}
