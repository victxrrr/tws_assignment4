#include <cassert>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include "../headers/solvers.hpp"

namespace ublas = boost::numeric::ublas;

namespace params 
{
    double beta, mu, gamma, alpha, delta;
}

template <typename wp>
ublas::vector<wp> f(ublas::vector<wp> const& xk)
{
    ublas::vector<wp> eval(5);
    eval(0) = -params::beta*xk(0)*xk(1)/(xk(0) + xk(1) + xk(3)) + params::mu*xk(3);
    eval(1) = (params::beta*xk(0)/(xk(0) + xk(1) + xk(3)) - params::gamma - params::delta - params::alpha)*xk(1);
    eval(2) = params::delta*xk(1) - (params::gamma + params::alpha)*xk(2);
    eval(3) = params::gamma*(xk(1) + xk(2)) - params::mu*xk(3);
    eval(4) = params::alpha*(xk(1) + xk(2));
    return eval;
}

template <typename wp>
ublas::matrix<wp> jac_f(ublas::vector<wp> const& xk)
{
    ublas::matrix<wp> eval(5, 5);
    wp den = (xk(0) + xk(1) + xk(3))*(xk(0) + xk(1) + xk(3));
    eval(0,0) = -params::beta * (xk(1)*xk(1) + xk(1)*xk(3))/den;
    eval(1,0) = -eval(0,0);
    eval(2,0) = 0.0;
    eval(3,0) = 0.0;
    eval(4,0) = 0.0;
    eval(0,1) = -params::beta * (xk(0)*xk(0) + xk(0)*xk(3))/den;
    eval(1,1) = -eval(0,1) - params::gamma - params::delta - params::alpha;
    eval(2,1) = params::delta;
    eval(3,1) = params::gamma;
    eval(4,1) = params::alpha;
    eval(0,2) = 0.0;
    eval(1,2) = 0.0;
    eval(2,2) = -(params::gamma + params::alpha);
    eval(3,2) = params::gamma;
    eval(4,2) = params::alpha;
    eval(0,3) = params::beta*xk(0)*xk(1)/den + params::mu;
    eval(1,3) = -eval(0,3) + params::mu;
    eval(2,3) = 0.0;
    eval(3,3) = -params::mu;
    eval(4,3) = 0.0;
    eval(0,4) = 0.0;
    eval(1,4) = 0.0;
    eval(2,4) = 0.0;
    eval(3,4) = 0.0;
    eval(4,4) = 0.0;
    return eval;
}

int main(int argc, char *argv[])
{
    assert(argc == 3);
    int N = atoi(argv[1]);
    double T = atof(argv[2]);
    assert(N > 0);
    assert(T > 0);

    ublas::matrix<double> x(N+1,5);
    ublas::vector<double> tk(N+1);

    ivp::Initialize(params::beta, params::mu, params::gamma, params::alpha, params::delta, x, "input/parameters.in");
    params::delta = 0.0;
    ivp::EulerForward(T, N, tk, x, f<double>);
    ivp::DisplayResults(tk, x, "./output/fwe_no_measures.out");

    ivp::Initialize(params::beta, params::mu, params::gamma, params::alpha, params::delta, x, "input/parameters.in");
    params::delta = 0.9;
    ivp::Heun(T, N, tk, x, f<double>);
    ivp::DisplayResults(tk, x, "./output/heun_lockdown.out");

    ivp::Initialize(params::beta, params::mu, params::gamma, params::alpha, params::delta, x, "input/parameters.in");
    params::delta = 0.2;
    ivp::EulerBackward(T, N, tk, x, f<double>, jac_f<double>);
    ivp::DisplayResults(tk, x, "./output/bwe_quarantine.out", true);

    return 0;
}