#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/SparseQR>
#include <Eigen/OrderingMethods>
#include <stdexcept>
#include <utility>

namespace py=pybind11;
using Sparse=Eigen::SparseMatrix<double,Eigen::ColMajor,int>;
using Dense=Eigen::MatrixXd;
template<class Ordering>
class SparseQRRange {
    using Factor=Eigen::SparseQR<Sparse,Ordering>;
    Sparse Z_,R_,R11_;
    Factor qr_;
    double threshold_;
    int rank_;
    Dense from_basis(const Dense& t) const {
        Dense padded=Dense::Zero(Z_.rows(),t.cols());padded.topRows(rank_)=t;
        return qr_.matrixQ()*padded;
    }
    Dense coefficients(const Dense& t) const {
        Dense permuted=Dense::Zero(Z_.cols(),t.cols());
        if(rank_)permuted.topRows(rank_)=R11_.triangularView<Eigen::Upper>().solve(t);
        return qr_.colsPermutation()*permuted;
    }
public:
    SparseQRRange(const Sparse& Z,double threshold):Z_(Z),threshold_(threshold),rank_(0) {
        if(Z.rows()<Z.cols()||Z.cols()==0||!std::isfinite(threshold)||threshold<=0)
            throw std::invalid_argument("Tall nonempty finite map and positive explicit pivot threshold required");
        for(int k=0;k<Z_.nonZeros();++k)if(!std::isfinite(Z_.valuePtr()[k]))throw std::invalid_argument("Nonfinite sparse entry");
        Z_.makeCompressed();qr_.setPivotThreshold(threshold_);qr_.compute(Z_);
        if(qr_.info()!=Eigen::Success)throw std::runtime_error("SparseQR failed: "+qr_.lastErrorMessage());
        rank_=qr_.rank();
        // Eigen documents unsorted internal R entries. Row-major conversion
        // followed by column-major copying explicitly sorts before extraction.
        Eigen::SparseMatrix<double,Eigen::RowMajor,int> sorted=qr_.matrixR();
        R_=sorted;R11_=R_.topLeftCorner(rank_,rank_);R11_.makeCompressed();
    }
    std::pair<Dense,Dense> project(const Dense& y) const {
        if(y.rows()!=Z_.rows()||!y.allFinite())throw std::invalid_argument("Projection RHS mismatch");
        const Dense transformed=qr_.matrixQ().adjoint()*y;
        const Dense t=transformed.topRows(rank_);
        return {from_basis(t),coefficients(t)};
    }
    std::pair<Dense,Dense> solve_gram(const Dense& b) const {
        if(b.rows()!=Z_.cols()||!b.allFinite())throw std::invalid_argument("Gram load mismatch");
        const Dense permuted=qr_.colsPermutation().transpose()*b;
        Dense t(rank_,b.cols());
        if(rank_)t=R11_.transpose().template triangularView<Eigen::Lower>().solve(permuted.topRows(rank_));
        return {coefficients(t),from_basis(t)};
    }
    Dense basis_action(const Dense& t) const {
        if(t.rows()!=rank_||!t.allFinite())throw std::invalid_argument("Basis-coordinate mismatch");
        return from_basis(t);
    }
    Eigen::VectorXd diagonal() const {
        Eigen::VectorXd answer(rank_);
        for(int i=0;i<rank_;++i)answer(i)=R11_.coeff(i,i);
        return answer;
    }
    Eigen::VectorXi permutation() const { return qr_.colsPermutation().indices(); }
    int rows() const{return Z_.rows();}
    int columns() const{return Z_.cols();}
    int rank() const{return rank_;}
    int r_nonzeros() const{return R_.nonZeros();}
    double threshold() const{return threshold_;}
};

template<class Ordering>
void bind_range(py::module_& m,const char* name) {
    using Range=SparseQRRange<Ordering>;
    py::class_<Range>(m,name)
      .def(py::init<const Sparse&,double>(),py::arg("Z"),py::arg("pivot_threshold"),py::call_guard<py::gil_scoped_release>())
      .def("project",&Range::project,py::call_guard<py::gil_scoped_release>())
      .def("solve_gram",&Range::solve_gram,py::call_guard<py::gil_scoped_release>())
      .def("basis_action",&Range::basis_action,py::call_guard<py::gil_scoped_release>())
      .def("diagonal",&Range::diagonal).def("permutation",&Range::permutation)
      .def_property_readonly("rows",&Range::rows).def_property_readonly("columns",&Range::columns)
      .def_property_readonly("rank",&Range::rank).def_property_readonly("r_nonzeros",&Range::r_nonzeros)
      .def_property_readonly("pivot_threshold",&Range::threshold);
}
PYBIND11_MODULE(mpm_sparseqr_native,m) {
    m.attr("__version__")="mpm-eigen-sparseqr-range-v2";
    bind_range<Eigen::COLAMDOrdering<int>>(m,"SparseQRRange");
    bind_range<Eigen::NaturalOrdering<int>>(m,"SparseQRNaturalRange");
}
