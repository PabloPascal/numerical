#ifndef MATRIX_TYPE
#define MATRIX_TYPE
#include "linalg.hpp"
#include <cmath>
#include <type_traits>

//______________________MATRIX TYPE_____________________

#define PI 3.14159265359




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

template <std::floating_point T>
Tensor<T> Rotate2d(T angle){

    angle = angle * PI/(T)180;
    Tensor<T> Rotate(2,2);

    Rotate(0,0) = std::cos(angle); 
    Rotate(0,1) = -std::sin(angle);
    Rotate(1,0) = std::sin(angle);
    Rotate(1,1) = std::cos(angle);    
        
    return Rotate;

}

} // namespace LIN




#endif