#ifndef IVP_SOLVERS_HPP
#define IVP_SOLVERS_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <fstream>
#include <iomanip>

namespace ublas = boost::numeric::ublas;

namespace ivp {

    template <typename wp>
    void Initialize(wp& beta, wp& mu, wp& gamma, wp& alpha, wp& delta, ublas::matrix<wp>& x, const char *filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Cannot open file." << std::endl;
            exit(1);
        }

        file >> beta >> mu >> gamma >> alpha >> delta >> x(0,0) >> x(0,1);
        x(0,2) = 0.;
        x(0,3) = 0.;
        x(0,4) = 0.;

        file.close();
    }

    template <typename wp, typename F>
    void EulerForward(wp const T, int const N, ublas::vector<wp>& tk, ublas::matrix<wp>& x, F const& f)
    {
       tk(0) = 0.0;
       for (int k = 0; k < N; k++)
       {
        tk(k+1) = (T/N)*(k+1);
        ublas::matrix_row<ublas::matrix<wp>>xk(x,k);
        ublas::matrix_row<ublas::matrix<wp>>xkk(x,k+1);
        xkk.assign(xk + (T/N)*f(xk));
       }
    }

    template <typename wp, typename F>
    void Heun(wp const T, int const N, ublas::vector<wp>& tk, ublas::matrix<wp>& x, F const& f)
    {
        tk(0) = 0.0;
        for (int k = 0; k < N; k++)
        {
            tk(k+1) = (T/N)*(k+1);
            ublas::matrix_row<ublas::matrix<wp>>xk(x,k);
            ublas::matrix_row<ublas::matrix<wp>>xkk(x,k+1);
            xkk.assign(xk + (T/N)*(0.5*f(xk) + 0.5*f(xk + (T/N)*f(xk))));
        }
    }

    template <typename wp, typename F, typename G>
    void EulerBackward(wp const T, int const N, ublas::vector<wp>& tk, ublas::matrix<wp>& x, F const& f, G const& jac_f, wp const tol=1e-10, int const nmax=100) 
    {
        ublas::vector<wp> bx;
        ublas::matrix<wp> A;

        tk(0) = 0.0;
        for (int k = 0; k < N; k++)
        {
            tk(k+1) = (T/N)*(k+1);
            ublas::matrix_row<ublas::matrix<wp>>xk(x,k);
            ublas::matrix_row<ublas::matrix<wp>>xkk(x,k+1);

            xkk.assign(xk);
            int s;
            for (s = 0; s < nmax; s++)
            {
                bx = xk + (T/N)*f(xkk) - xkk;
                if (norm_2(bx) < tol*norm_2(xkk)) break;

                A = (T/N)*jac_f(xkk);
                ublas::identity_matrix<wp> I(A.size1());
                A = A - I;

                ublas::permutation_matrix<size_t> pm(A.size1());
                ublas::lu_factorize(A,pm);
                ublas::lu_substitute(A,pm,bx);

                xkk.assign(xkk - bx);
            }
            if (s == nmax)
            {
                std::cerr << "Warning: Maximum number of iterations reached." << std::endl;
            }
        }
    }

    template <typename wp>
    void DisplayResults(ublas::vector<wp>& tk, ublas::matrix<wp>& x, const char *filename, const bool verbose=false)
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Cannot open file." << std::endl;
            exit(1);
        }

        for (std::size_t i = 0; i < tk.size(); i++)
        {
            if (verbose) std::cout<<std::setw(12)<<tk(i)<<" ";
            file<<tk(i)<<" ";
            for (int j = 0; j < 5; j++)
            {
                if (verbose) std::cout<<std::setw(12)<<x(i,j)<<" ";
                file<<x(i,j)<<" ";
            }
            if (verbose) std::cout<<std::endl;
            file<<"\n";
        }
        file.close();
    }
}

#endif
