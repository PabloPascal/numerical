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
protected:
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


    void transpose() {
    if (_rows != _cols) {
        Tensor<T> temp(_cols, _rows);
        const size_t rows = _rows, cols = _cols;
        const T* src = _data.data();
        T* dst = temp._data.data();
        const size_t BLOCK = 32;
        for (size_t i = 0; i < rows; i += BLOCK) {
            size_t i_end = std::min(i + BLOCK, rows);
            for (size_t j = 0; j < cols; j += BLOCK) {
                size_t j_end = std::min(j + BLOCK, cols);
                for (size_t ii = i; ii < i_end; ++ii) {
                    const T* src_row = src + ii * cols;
                    for (size_t jj = j; jj < j_end; ++jj) {
                        dst[jj * rows + ii] = src_row[jj];
                    }
                }
            }
        }
        _rows = temp._rows;
        _cols = temp._cols;
        _data = std::move(temp._data);
        return;
    }

    const size_t n = _rows;
    T* data = _data.data();
    const size_t BLOCK = 32;

    for (size_t i = 0; i < n; i += BLOCK) {
        size_t i_end = std::min(i + BLOCK, n);
        for (size_t j = i; j < n; j += BLOCK) {
            size_t j_end = std::min(j + BLOCK, n);
            if (i == j) {
                for (size_t ii = i; ii < i_end; ++ii) {
                    T* row = data + ii * n;
                    for (size_t jj = std::max(j, ii + 1); jj < j_end; ++jj) {
                        std::swap(row[jj], data[jj * n + ii]);
                    }
                }
            } else {
                for (size_t ii = i; ii < i_end; ++ii) {
                    T* row_ii = data + ii * n;
                    for (size_t jj = j; jj < j_end; ++jj) {
                        std::swap(row_ii[jj], data[jj * n + ii]);
                    }
                }
            }
        }
    }
    }

};


template <std::floating_point T>
class vec : public Tensor<T> 
{
public:
    vec(size_t size) : Tensor<T>(size, 1){}

    vec(size_t size, std::vector<T> data) : Tensor<T>(size, 1, data){}

    vec(size_t size, T init_value) : Tensor<T>(size, 1, init_value){}

    vec(size_t size, std::pair<T, T> interval, bool normalize = false) : Tensor<T>(size, 1, interval, normalize){}

    vec(const Tensor<T>& tensor) : Tensor<T>(tensor){
        if(!(tensor.cols() == 1 || tensor.rows() == 1)){
            throw std::invalid_argument("not a vector tensor");
        }
    }

    T length(){
        T sq_sum = 0;
        for(size_t i = 0; i < this->_rows; ++i){
            sq_sum += this->_data[i] * this->_data[i];
        }
        return std::sqrt(sq_sum);
    }

    T& operator[](size_t i){
        return this->_data[i];
    } 

    T operator[](size_t i) const {
        return this->_data[i];
    }

    size_t size() const{
        return this->_rows * this->_cols;
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

    
    Tensor<T> C(A.rows(), A.cols());


    const T* Adata = A.data();  
    const T* Bdata = B.data(); 
    T* Cdata = C.data();

    size_t full_size = A.cols() * A.rows();

 #pragma omp parallel for simd schedule(static)    
 for(size_t i = 0; i < full_size; ++i){
        Cdata[i] = Adata[i] + Bdata[i];
    }

    return C;

}



template <std::floating_point T> 
T dot_product(const vec<T>& a, const vec<T>& b){

    if(!(a.cols() == 1 || a.rows() == 1)) throw std::invalid_argument("not a vector!");
    if(!(b.cols() == 1 || b.rows() == 1)) throw std::invalid_argument("not a vector!");
    
    size_t a_vec_size = std::max(a.cols(), a.rows());
    size_t b_vec_size = std::max(b.cols(), b.rows());

    if(a_vec_size != b_vec_size) throw std::invalid_argument("not same size!");

    const T* a_data = a.data();
    const T* b_data = b.data();

    T result = (T)0;

    for(size_t i=0; i < a_vec_size; ++i){
        result += a_data[i]*b_data[i];
    }
    return result;
}



template <std::floating_point T>
Tensor<T> transpose(const Tensor<T>& A) {
    const size_t rows = A.rows();
    const size_t cols = A.cols();
    Tensor<T> C(cols, rows);

    const T* A_data = A.data();
    T* C_data = C.data();

    const size_t BLOCK = 32;  // размер блока, можно настроить

    for (size_t i = 0; i < rows; i += BLOCK) {
        size_t i_end = std::min(i + BLOCK, rows);
        for (size_t j = 0; j < cols; j += BLOCK) {
            size_t j_end = std::min(j + BLOCK, cols);
            for (size_t ii = i; ii < i_end; ++ii) {
                const T* src_row = A_data + ii * cols;
                for (size_t jj = j; jj < j_end; ++jj) {
                    // C(jj, ii) = A(ii, jj)
                    C_data[jj * rows + ii] = src_row[jj];
                }
            }
        }
    }
    return C;
}



template <std::floating_point T>
Tensor<T> hadamar_product(const Tensor<T>& a, const Tensor<T>& b){

    if(a.cols() != b.cols() || a.rows() != b.rows()) 
        throw std::invalid_argument("not saze size");

    const size_t rows = a.rows();
    const size_t cols = a.cols();
     
    Tensor<T> c(rows, cols, 0);

    T* c_data = c.data();
    const T* a_data = a.data();
    const T* b_data = b.data();

    for(size_t i = 0; i < cols * rows; ++i){
        c_data[i] = a_data[i] * b_data[i];
    }   

    return c;

}


template <std::floating_point T>
Tensor<T> operator*(T alpha, const Tensor<T>& tensor){

    const T* t_data = tensor.data();
    
    const size_t rows = tensor.rows(); 
    const size_t cols = tensor.cols();

    Tensor<T> c(rows, cols);

    T* c_data = c.data();
    for(size_t i = 0; i < rows * cols; ++i){
        c_data[i] = t_data[i] * alpha;
    }

    return c;

}

template <std::floating_point T, typename Func>
Tensor<T> apply(const Tensor<T>& a, Func&& func){

    const T* a_data = a.data();
    const size_t rows = a.rows();
    const size_t cols = a.cols();

    Tensor<T> c(rows, cols);

    T* c_data = c.data();

    for(size_t i = 0; i < rows * cols; ++i){
        c_data[i] = func(a_data[i]);
    }


    return c;
}



template <std::floating_point T>
Tensor<T> cross(const vec<T>& a, const vec<T>& b){

    if(a.rows() * a.cols() != b.rows() * b.cols() && a.rows() * a.cols() != 3) 
        throw std::invalid_argument("not valid size, should be 3d vectors");

    const T* a_data = a.data();
    const T* b_data = b.data();

    T c1 = a_data[1] * b_data[2] - b_data[1] * a_data[2];
    T c2 = -(a_data[0]*b_data[2] - b_data[0]*a_data[2]);
    T c3 = a_data[0]*b_data[1] - b_data[0]*a_data[1];

    Tensor<T> C(3, 1, {c1, c2, c3});

    return C;
}


template <std::floating_point T>
Tensor<T> outer_product(const vec<T>& vec1, const vec<T>& vec2){

    Tensor<T> C(vec1.size(), vec2.size());
    
    for(size_t i = 0; i < vec1.size(); i++){
        for(size_t j = 0; j < vec2.size(); j++){
            C(i, j) = vec1[i]*vec2[j];
        }
    }

    return C;
}


}//LIN SPACE






#endif 