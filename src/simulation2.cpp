#include <cassert>
#include <numeric>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "../headers/solvers.hpp"

namespace ublas = boost::numeric::ublas;

template <typename wp>
ublas::vector<wp> f(ublas::vector<wp> const& xk)
{
    ublas::vector<wp> eval(50);
    ublas::vector<int> tmp(50);
    std::iota(tmp.begin(), tmp.end(), 0);
    std::transform(xk.begin(), xk.end(), tmp.begin(), eval.begin(),
                    [](double elem, int index) {
                        return -10.0 * std::pow(elem - index * 0.1, 3);
                    });
    return eval;
}

template <typename wp>
ublas::diagonal_matrix<wp, ublas::row_major, ublas::vector<wp>> jac_f(ublas::vector<wp> const& xk)
{
    ublas::vector<wp> diag(50);
    ublas::vector<int> tmp(50);
    std::iota(tmp.begin(), tmp.end(), 0);
    std::transform(xk.begin(), xk.end(), tmp.begin(), diag.begin(),
                    [](double elem, int index) {
                        return -30.0 * std::pow(elem - index * 0.1, 2);
                    });
    ublas::diagonal_matrix<wp, ublas::row_major, ublas::vector<wp>> eval(50, diag);
    return eval;
}

template <typename wp>
struct f_functor
{
    ublas::vector<wp> operator()(ublas::vector<wp> const& xk) const
    {
        ublas::vector<wp> eval(50);
        ublas::vector<int> tmp(50);
        std::iota(tmp.begin(), tmp.end(), 0);
        std::transform(xk.begin(), xk.end(), tmp.begin(), eval.begin(),
                        [](double elem, int index) {
                            return -10.0 * std::pow(elem - index * 0.1, 3);
                        });
        return eval;
    }
};

template <typename wp>
struct jac_f_functor
{
    ublas::diagonal_matrix<wp, ublas::row_major, ublas::vector<wp>> operator()(ublas::vector<wp> const& xk) const
    {
        ublas::vector<wp> diag(50);
        ublas::vector<int> tmp(50);
        std::iota(tmp.begin(), tmp.end(), 0);
        std::transform(xk.begin(), xk.end(), tmp.begin(), diag.begin(),
                        [](double elem, int index) {
                            return -30.0 * std::pow(elem - index * 0.1, 2);
                        });
        ublas::diagonal_matrix<wp, ublas::row_major, ublas::vector<wp>> eval(50, diag);
        return eval;
    }

};

template <typename wp>
auto f_lambda = [] (ublas::vector<wp> const& xk) -> ublas::vector<wp> 
{
    ublas::vector<wp> eval(50);
    ublas::vector<int> tmp(50);
    std::iota(tmp.begin(), tmp.end(), 0);
    std::transform(xk.begin(), xk.end(), tmp.begin(), eval.begin(),
                    [](double elem, int index) {
                        return -10.0 * std::pow(elem - index * 0.1, 3);
                    });
    return eval;
};

template <typename wp>
auto jac_f_lambda = [] (ublas::vector<wp> const& xk) -> ublas::diagonal_matrix<wp, ublas::row_major, ublas::vector<wp>>
{
    ublas::vector<wp> diag(50);
    ublas::vector<int> tmp(50);
    std::iota(tmp.begin(), tmp.end(), 0);
    std::transform(xk.begin(), xk.end(), tmp.begin(), diag.begin(),
                    [](double elem, int index) {
                        return -30.0 * std::pow(elem - index * 0.1, 2);
                    });
    ublas::diagonal_matrix<wp, ublas::row_major, ublas::vector<wp>> eval(50, diag);
    return eval;
};


int main(int argc, char *argv[])
{
    assert(argc == 3);
    int N = atoi(argv[1]);
    double T = atof(argv[2]);
    assert(N > 0);
    assert(T > 0);

    ublas::matrix<double> x(N+1,50);
    ublas::vector<double> tk(N+1);

    // Initialization
    auto QuickInit = [&x] () -> void {
        ublas::matrix_row<ublas::matrix<double>>x0(x,0);
        std::iota(x0.begin(), x0.end(), 1);
        x0 = x0/100;
    };

    QuickInit();
    ivp::EulerForward(T, N, tk, x, f<double>);
    ivp::DisplayResults(tk, x, "./output/fwe_simulation2.out");

    QuickInit();
    ivp::Heun(T, N, tk, x, f_lambda<double>);
    ivp::DisplayResults(tk, x, "./output/heun_simulation2.out");

    f_functor<double> theFunction;
    jac_f_functor<double> theJacobian;

    QuickInit();
    ivp::EulerBackward(T, N, tk, x, theFunction, theJacobian);
    ivp::DisplayResults(tk, x, "./output/bwe_simulation2.out");

    return 0;
}