#ifndef MATRIX_TYPE
#define MATRIX_TYPE
#include "linalg.hpp"
#include <cmath>

//______________________MATRIX TYPE_____________________



namespace LIN
{

double PI = 3.14159265359;

using matrix_d = Matrix<double>;
using matrix_f = Matrix<float>;
using vec_d = Vector<double>;
using vec_f = Vector<float>;


template <typename T>
class Identity : public Matrix<T>
{
public:
    Identity(size_t N) : Matrix<T>(N, N)
    {

        for(size_t i = 0; i < N; i++)
        {
            for(size_t j = 0; j < N; j++)
            {
                if(i != j)
                    this->_data[i * this->_cols + j] = 0;
                if(i == j)
                    this->_data[i * this->_cols + j] = 1;
            }
        }

    }

};


class Rotation2 : public Matrix<double>
{
public:
    Rotation2(double angle) : Matrix<double>(2,2)
    {
        double radian = angle * PI / 180;

        this->_data[0 + 0] = cos(radian);
        this->_data[0 + 1] =-sin(radian);
        this->_data[1 + 1] = sin(radian);
        this->_data[1 + 2] = cos(radian);
    }
};





} // namespace LIN




#endif