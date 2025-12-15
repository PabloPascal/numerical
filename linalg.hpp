#ifndef __LINALG__
#define __LINALG__

#include <cstddef>
#include <vector>
#include <random>

namespace LIN{

template <typename T>
class Matrix
{
private:

    void resize()
    {
        _data.resize(_rows);
        for(size_t i = 0; i < _rows; i++)
        {
            _data[i].resize(_cols);
        }
    }
    void copy(const Matrix& matrix)
    {
        for(size_t i = 0; i < _rows; i++)
        {
            for(size_t j = 0; j < _cols; j++)
            {
                _data[i][j] = matrix._data[i][j];
            }
        }
    }

public:

    Matrix(size_t rows, size_t cols, T init_value = 0) : _rows(rows), _cols(cols)
    {
        resize();
        fill(init_value);
    } 

    Matrix(const Matrix& matrix){
        _rows = matrix._rows;
        _cols = matrix._cols;

        resize();
        copy(matrix);

    }

    void fill(T value)
    {
        for(size_t i = 0; i < _rows; i++)
        {
            for(size_t j = 0; j < _cols; j++)
            {
                _data[i][j] = value;
            }
        }
    }

    T operator()(size_t i, size_t j) const {
        return _data[i][j];
    }
    Matrix& operator=(const Matrix& M)
    {
        if(_cols != M._cols || _rows != M._rows)
        {
            throw std::length_error("not same size");
        }

        copy(M);
        return *this;
    }
    void set(size_t i, size_t j, T value){
        _data[i][j] = value;
    }

    Matrix& transpose()
    {
        if(_cols == _rows){
            for(size_t i = 0; i < _rows; i++)
            {
                for(size_t j = 0; j < _cols; j++)
                {
                    _data[i][j] = _data[j][i];
                }
            }
        }else{

            size_t new_rows = _cols;
            size_t new_cols = _rows;

            std::vector<std::vector<T>> new_data(new_rows);
            for(size_t i = 0; i < new_rows; i++)
            {
                new_data[i].resize(new_cols);
            }

            for(size_t i = 0; i < _rows; i++)
            {
                for(size_t j = 0; j < _cols; j++)
                {
                    new_data[j][i] = _data[i][j];
                }
            }

            _data = new_data;
            _cols = new_cols;
            _rows = new_rows;
        
        }

        return *this;
    }

    void random_init()
    {
        std::random_device rd;
        static std::mt19937 gen(rd()); 
        std::uniform_int_distribution<> dis(1, 100);

        for(size_t i = 0; i < _rows; i++)
        {
            for(size_t j = 0; j < _cols; j++)
            {
                T random_num = dis(gen);
                _data[i][j] = random_num;
            }
        }
    }

    size_t get_cols() const {return _cols;} 
    size_t get_rows() const {return _rows;}

protected:

    size_t _rows;
    size_t _cols;

    std::vector<std::vector<T>> _data;


};

template <typename T>
Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B)
{
    if(A.get_cols() != B.get_rows())
    {
        throw std::length_error("cannot multiply matrix because of dimensions");
    }


    Matrix<T> C(A.get_rows(), B.get_cols());

    for(size_t i = 0; i < A.get_rows(); i++)
    {
        for(size_t j = 0; j < B.get_cols(); j++)
        {
            T s = 0;
            for(size_t k = 0; k < A.get_cols(); k++)
            {
                s += A(i, k) * B(k, j);
            }
            C.set(i, j, s); 
        }
    }

    return C;

}

template <typename T>
Matrix<T> operator+(const Matrix<T>& A, const Matrix<T>& B)
{
    if(A.get_cols() != B.get_cols() || A.get_rows() != B.get_rows())
    {
        throw std::invalid_argument("bad dimensions");
    }

    size_t n = A.get_cols();
    size_t m = B.get_rows();

    Matrix<T> C(n, m);

    for(size_t i = 0; i < n; i++)
    {
        for(size_t j = 0; j < m; j++)
        {
            T s = A(i, j) + B(i, j);
            C.set(i, j, s);
        }
    }
    return C;
}

template <typename T>
Matrix<T> operator*(const T scalar, const Matrix<T>& A)
{
    size_t n = A.get_cols();
    size_t m = A.get_rows();

    Matrix<T> C(n, m);

    for(size_t i = 0; i < n; i++)
    {
        for(size_t j = 0; j < m; j++)
        {
            T s = scalar * A(i, j);
            C.set(i, j, s);
        }
    }
    return C;
}

template <typename T>
Matrix<T> operator*(const Matrix<T>& A, const T scalar)
{
    size_t n = A.get_cols();
    size_t m = A.get_rows();

    Matrix<T> C(n, m);

    for(size_t i = 0; i < n; i++)
    {
        for(size_t j = 0; j < m; j++)
        {
            T s = scalar * A(i, j);
            C.set(i, j, s);
        }
    }
    return C;
}

template <typename T>
Matrix<T> hadamarProduct(const Matrix<T>& A, const Matrix<T>& B)
{
    if(A.get_cols() != B.get_cols() || A.get_rows() != B.get_rows())
    {
        throw std::invalid_argument("bad dimensions");
    }

    size_t n = A.get_cols();
    size_t m = B.get_rows();

    Matrix<T> C(n, m);

    for(size_t i = 0; i < n; i++)
    {
        for(size_t j = 0; j < m; j++)
        {
            T s = A(i, j) * B(i, j);
            C.set(i, j, s);
        }
    }
    return C;
}


//MATRIX TYPE
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
                    this->_data[i][j] = 0;
                if(i == j)
                    this->_data[i][j] = 1;
            }
        }

    }

};





}//LIN SPACE

#endif 