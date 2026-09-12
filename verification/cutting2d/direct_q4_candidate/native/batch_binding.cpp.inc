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
