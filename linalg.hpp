#ifndef LINALG_HPP
#define LINALG_HPP

#include <cstddef>
#include <vector>
#include <random>
#include <iterator>
#include <concepts>
#include <type_traits>
#include <utility>
#include <omp.h>
#include <algorithm> 



namespace linalg{

template <std::floating_point T> 
class Tensor
{
    size_t _rows;
    size_t _cols;

    std::vector<T> _data;
public:

    Tensor() : _rows(0), _cols(0) {}

    //initialization with init value 
    Tensor(size_t rows, size_t cols, T init_value = 0) : _rows(rows), _cols(cols)
    {
        _data.resize(_rows * _cols);
        std::fill(_data.begin(), _data.end(), init_value);
    } 

    //random initialization
    Tensor(size_t rows, size_t cols, std::pair<T, T> interval, bool normalize = false) : _rows(rows), _cols(cols)
    {
        _data.resize(_rows * _cols);
        random_init(interval, normalize);
    } 
    
    Tensor(size_t rows, size_t cols, const std::vector<T>& data) : _rows(rows), _cols(cols)
    {
        if(data.size() != cols * rows) 
            throw std::length_error("different size with data in constructor");
        
        _data = data;

    }
    Tensor(size_t rows, size_t cols, std::vector<T>&& data) : _rows(rows), _cols(cols)
    {
        if(data.size() != cols * rows) 
            throw std::length_error("different size with data in constructor");
        
        _data = std::move(data);

    }




    void random_init(std::pair<T, T> interval, bool normalize = false)
    {
        T left = interval.first;
        T right = interval.second;

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(static_cast<T>(left), static_cast<T>(right));
       
        const T range = static_cast<T>(right - left);

        for(size_t i = 0; i < _rows*_cols; i++){
            T val = dis(gen);
            if (normalize) val /= range;
            _data[i] = val;
        }
    }

    size_t cols() const {return _cols;}
    size_t rows() const {return _rows;}


    std::vector<T> get_data() const {return _data;}
    const std::vector<T>& get_ref_data() const {return _data;}
    const T* data() const { return _data.data(); }
    T* data() { return _data.data(); }


    T operator()(size_t i, size_t j) const 
    {
        return _data[_cols * i + j];
    } 

    T& operator()(size_t i, size_t j) 
    {
        return _data[_cols * i + j];
    } 

};


template <std::floating_point T> 
Tensor<T> operator*(const Tensor<T>& A, const Tensor<T>& B)
{

    if(A.cols() != B.rows())
        throw std::length_error("different size");         

    
    const size_t M = A.rows();   
    const size_t N = B.cols();   
    const size_t K = A.cols();   

    Tensor<T> C(M, N, T(0));     
    
    const T* A_data = A.data();
    const T* B_data = B.data();
    T*       C_data = C.data();

    const size_t BLOCK = 64;

#pragma omp parallel for collapse(2) schedule(dynamic)

    for (size_t i0 = 0; i0 < M; i0 += BLOCK) {
        size_t i_end = std::min(i0 + BLOCK, M);

        for (size_t j0 = 0; j0 < N; j0 += BLOCK) {
            size_t j_end = std::min(j0 + BLOCK, N);

            for (size_t k0 = 0; k0 < K; k0 += BLOCK) {
                size_t k_end = std::min(k0 + BLOCK, K);

                for (size_t i = i0; i < i_end; ++i) {
                    T* C_row = C_data + i * N;          
                    const T* A_row = A_data + i * K;    

                    for (size_t k = k0; k < k_end; ++k) {
                        T a_ik = A_row[k];
                        if (a_ik == T(0)) continue;    
                        const T* B_row = B_data + k * N; 

                        #pragma omp simd
                        for (size_t j = j0; j < j_end; ++j) {
                            C_row[j] += a_ik * B_row[j];
                        }
                    }
                }
            }
        }
    }


    return C;
}    



template <std::floating_point T> 
Tensor<T> operator+(const Tensor<T>& A, const Tensor<T>& B)
{
    if(A.cols() != B.cols() || A.rows() != B.rows())
        throw std::length_error("different size");

    size_t num_cols = A.cols();
    size_t num_rows = A.rows();

    Tensor<T> C(num_rows, num_cols);

#pragma omp parallel for collapse(2) schedule(dynamic)
    for(size_t i = 0; i < num_cols; i++){
        for(size_t j = 0; j < num_rows; j++){
            C(i,j) = A(i, j) + B(i, j);
        }
    }

    return C;

}





}//LIN SPACE



#endif 