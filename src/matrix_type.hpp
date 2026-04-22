#ifndef MATRIX_TYPE
#define MATRIX_TYPE
#include "linalg.hpp"
#include <cmath>

//______________________MATRIX TYPE_____________________



namespace linalg
{

template <std::floating_point T>
Tensor<T> I(size_t n){

    Tensor<T> I(n, n, 0);

    for(size_t i = 0; i < n; ++i){
        for(size_t j = 0; j < n; ++j){
            I(i, i) = 1;
        }
    }

    return I;
}    



} // namespace LIN




#endif