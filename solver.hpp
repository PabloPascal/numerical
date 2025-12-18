#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "linalg.hpp"

namespace LIN{

class Solver
{

public:
    template <typename T>
    static Vector<T> GausseSolverSystem(Matrix<T>& A, Vector<T>& b)
    {
        if(A.get_cols() != A.get_rows() || 
            A.get_rows() != b.getSize())
        {
            throw "cannot work with non square matrix";
        }

        size_t size = A.get_cols();
        T coeff = 0;

        Vector<T> x(b.getSize(), 0);

        for(size_t j = 0; j < size; j++)
        {
            for(size_t i = size-1; i > j; i--)
            {
                if(A[i-1][j] == 0) continue;

                coeff = A[i][j] / A[i-1][j];

                for(size_t k = j; k < size; k++)
                {
                    A(i, k) = A(i, k) - coeff * A(i-1, k);
                }

                b[i] = b[i] - coeff * b[i - 1];
            }
        }

        
        for(int i = size - 1; i >= 0; i--)
        {
            T sum = 0;
            for(int j = size - 1; j > i; j--)
            {
                sum += A(i, j)*x[j];
            }

            x[i] = (b[i] - sum) / A(i, i);

        }

        return x;
    }


    template <typename T> 
    static T determinant(const Matrix<T>& A)
    {

        if(A.get_cols() != A.get_rows())
        {
            throw std::invalid_argument("not square matrix");
        }

        Matrix<T> B(A);
        Vector<T> x(A.get_cols());
        Solver::GausseSolverSystem(B, x);

        T result = 1;
        for(size_t i = 0; i < B.get_cols(); i++)
        {
            result *= B(i, i);
        }
        return result;
    }


};

}

#endif