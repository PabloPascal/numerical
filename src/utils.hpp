#include "linalg.hpp"
#include "matrix_type.hpp"
#include <stdio.h>

template <std::floating_point T>
void print_matrix(const linalg::Tensor<T>& tensor){

    for(size_t i = 0; i < tensor.rows(); ++i){
        for(size_t j = 0; j < tensor.cols(); ++j){
            printf("%f ", tensor(i, j));
        }
        printf("\n");
    }

}


