#ifndef LINALG_HPP
#define LINALG_HPP

#include <cstddef>
#include <vector>
#include <random>

#ifdef _WIN32
#include <omp.h>
#endif 



namespace LIN{

template <typename T>
class Vector;

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

    Matrix(const std::vector<std::vector<T>>& matrix_data)
    {

        if(matrix_data.size() <= 0)
        {
            throw std::invalid_argument("matrix data is null");
        }

        _rows = matrix_data.size();
        _cols = matrix_data[0].size();
        
        _data = matrix_data; 
    }


    Matrix(std::vector<std::vector<T>>&& matrix_data)
    {

        if(matrix_data.size() <= 0)
        {
            throw std::invalid_argument("matrix data is null");
        }

        _rows = matrix_data.size();
        _cols = matrix_data[0].size();
        
        _data = std::move(matrix_data); 
    }



    Matrix(size_t rows, size_t cols, const std::vector<T>& values)
    {
        if(rows * cols != values.size())
        {
            throw "MATRIX CONSTRUCTOR (rows, cols, std::initializer_list):\n \
            the initializing values do not match the size";
        }
        _rows = rows;
        _cols = cols;
        resize();
        
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                _data[i][j] = values[i * _cols + j];
            }
        }
    }

    Matrix(size_t rows, size_t cols, std::vector<T>&& values)
    {
        if(rows * cols != values.size())
        {
            throw "MATRIX CONSTRUCTOR (rows, cols, std::initializer_list):\n \
            the initializing values do not match the size";
        }
        _rows = rows;
        _cols = cols;
        resize();
        
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                _data[i][j] = std::move(values[i * _cols + j]);
            }
        }
    }


    Matrix(size_t rows, size_t cols, const std::initializer_list<T>& values)
    {
        if(rows * cols != values.size())
        {
            throw "MATRIX CONSTRUCTOR (rows, cols, std::initializer_list):\n \
            the initializing values do not match the size";
        }
        _rows = rows;
        _cols = cols;
        resize();
        
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                _data[i][j] = *(values.begin() + i*cols + j);
            }
        }
    }

    Matrix(size_t rows, size_t cols, std::initializer_list<T>&& values)
    {
        if(rows * cols != values.size())
        {
            throw "MATRIX CONSTRUCTOR (rows, cols, std::initializer_list):\n \
            the initializing values do not match the size";
        }
        _rows = rows;
        _cols = cols;
        resize();
        
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                _data[i][j] = std::move(*(values.begin() + i*cols + j));
            }
        }
    }



    Matrix(size_t rows, size_t cols, T init_value = 0) : _rows(rows), _cols(cols)
    {
        resize();
        fill(init_value);
    } 

    Matrix(const Matrix& matrix) : _rows(matrix._rows), _cols(matrix._cols)
    {
        _data = matrix._data;
    }

    Matrix(Matrix&& matrix)
    {
        _data = std::move(matrix._data);
        _cols = matrix._cols;
        _rows = matrix._rows;
        
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

    T& operator()(size_t i, size_t j) {
        return _data[i][j];
    }

    T operator()(size_t i, size_t j) const {
        return _data[i][j];
    }
   
    Vector<T> operator[](size_t i){
        if(i >= _rows)
        {
            throw std::out_of_range("index out of range");
        }
        return Vector<T>(_data[i], false);
    }

    Vector<T> get_column(size_t index)
    {
        if(index > _cols) throw std::out_of_range("index bigger than num of columns");

        std::vector<T> column(_rows);

        for(size_t i = 0; i < _rows; i++)
        {
            column[i] = _data[i][index];
        }

        return Vector(column);
    }

    Vector<T> get_row(size_t index)
    {
        return this->operator[](index);
    }

    Matrix& operator=(const Matrix& M)
    {
        if(_cols != M._cols || _rows != M._rows)
        {
            throw std::length_error("not same size");
        }

        _data = M._data;
        return *this;
    }
    Matrix& operator=(Matrix&& M)
    {
        if(_cols != M._cols || _rows != M._rows)
        {
            throw std::length_error("not same size");
        }

        _data = std::move(M._data);
        return *this;
    }

    void set(size_t i, size_t j, T value){
        _data[i][j] = value;
    }

    void set_column(size_t column_number, const Vector<T>& column)
    {

        if(column_number > _cols || column.getSize() != _rows)
        {
            throw std::out_of_range("index out of range");
        }

        for(size_t i = 0; i < _rows; i++)
        {
            _data[i][column_number] = column[i];
        }


    }
    void set_row(size_t row_number, const Vector<T>& row)
    {

        if(row_number > _cols || row.getSize() != _cols)
        {
            throw std::out_of_range("index out of range");
        }

        for(size_t i = 0; i < _cols; i++)
        {
            _data[row_number][i] = row[i];
        }

    }

    void set_column(size_t column_number, Vector<T>&& column)
    {

        if(column_number > _cols || column.getSize() != _rows)
        {
            throw std::out_of_range("index out of range");
        }

        for(size_t i = 0; i < _rows; i++)
        {
            _data[i][column_number] = column[i];
        }


    }
    void set_row(size_t row_number, Vector<T>&& row)
    {

        if(row_number > _cols || row.getSize() != _cols)
        {
            throw std::out_of_range("index out of range");
        }

        for(size_t i = 0; i < _cols; i++)
        {
            _data[row_number][i] = row[i];
        }

    }



    Matrix& transpose()
    {
        if(_cols == _rows){
            for(size_t i = 0; i < _rows; i++)
            {
                for(size_t j = i; j < _cols; j++)
                {
                    std::swap(_data[i][j], _data[j][i]);
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

    void random_init(int leftLimit = 0, int rightLimit = 100, bool normalize = false)
    {
        if(leftLimit > rightLimit) std::swap(leftLimit, rightLimit);
        std::random_device rd;
        static std::mt19937 gen(rd()); 
        std::uniform_int_distribution<> dis(leftLimit, rightLimit);
        int length = rightLimit - leftLimit;

        for(size_t i = 0; i < _rows; i++)
        {
            for(size_t j = 0; j < _cols; j++)
            {
                T random_num = dis(gen);
                if(normalize)
                    _data[i][j] = random_num / (double)length;
                else 
                    _data[i][j] = random_num;
            }
        }
    }

    size_t get_cols() const {return _cols;} 
    size_t get_rows() const {return _rows;}

    void swap_rows(size_t i, size_t j)
    {
        if(i >= _rows || j >= _rows)
        {
            throw std::out_of_range("i or j bigger than num rows");
        }

        std::swap(_data[i], _data[j]);
    }

    void swap_cols(size_t i, size_t j)
    {
        if(i >= _cols || j >= _cols)
        {
            throw std::out_of_range("i or j bigger than num cols");
        }

        for(size_t k = 0; k < _rows; k++)
        {
            std::swap(_data[k][i], _data[k][j]);
        }
    }


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

#pragma omp parallel for
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
Matrix<T> operator-(const Matrix<T>& A, const Matrix<T>& B)
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
            T s = A(i, j) - B(i, j);
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




template <typename T>
class Vector{
public:

    using iterator = typename std::vector<T>::iterator;
    using const_iterator =typename std::vector<T>::const_iterator;

    iterator begin()
    {
        return _data.begin();
    }
    iterator end()
    {
        return _data.end();
    }

    iterator operator+(iterator it)
    {
        _current = _data.begin() + it;
        return _current;
    }

    bool operator!=(iterator it)
    {
        if(_current != it)
            return true;
        
        return false;
    }


    Vector(std::vector<T>&& container, bool column_vector = true)
    {
        _column_vector = column_vector;
        _data = std::move(container);
    }


    Vector(const std::vector<T>& container, bool column_vector = true)
    {
        _column_vector = column_vector;
        _data = container;
    }

    Vector(const std::initializer_list<T>& values, bool column_vector = true)
    {
        _data = values;
        _column_vector = column_vector;
    }

    Vector(std::initializer_list<T>&& values, bool column_vector = true)
    {
        _data = std::move(values);
        _column_vector = column_vector;
    }

    Vector(size_t N, T init_value = 0, bool column_vector = true)
    {
        _data.resize(N, init_value);
        _column_vector = column_vector;
    }
    Vector(const Vector& v)
    {
        _data = v._data;
        _column_vector = v._column_vector;
    }
    Vector(Vector&& v)
    {
        _data = std::move(v._data);
        _column_vector = v._column_vector;
    }


    T& operator[](size_t index) 
    {
        return _data[index];
    } 

    T operator[](size_t index) const 
    {
        return _data[index];
    } 

    Vector& operator=(const Vector& vec)
    {
        if(_column_vector != vec._column_vector)
        {
            throw std::invalid_argument("different type");
        }
        if(_data.size() != vec._data.size())
        {
            throw std::invalid_argument("different size");
        }

        _data = vec._data;
        return *this;
    }
    Vector& operator=(Vector&& vec)
    {
        if(_column_vector != vec._column_vector)
        {
            throw std::invalid_argument("different type");
        }
        if(_data.size() != vec._data.size())
        {
            throw std::invalid_argument("different size");
        }
        _data = std::move(vec._data);
        return *this;
    }

    void set(size_t index, T value) {_data[index] = value;}
    size_t getSize() const {return _data.size();}

    Vector& transpose()
    {
        _column_vector = false;
        return *this;
    }

    bool isColumn() const {return _column_vector;}

    void random_init()
    {
        std::random_device rd;
        static std::mt19937 gen(rd()); 
        std::uniform_int_distribution<> dis(1, 100);

        for(size_t i = 0; i < _data.size(); i++)
        {
            _data[i] = dis(gen);
        }

    }

    Vector<T>& normalize()
    {

        for(size_t i = 0; i < _data.size(); i++)
        {
            _data[i] = _data[i] / len(*this);
        }

        return this;
    }


private:
    bool            _column_vector;
    std::vector<T>  _data;
    iterator        _current;
};


template <typename T>
T dot_product(const Vector<T>& v1, const Vector<T>& v2){

    if(v1.getSize() != v2.getSize()){
        throw std::invalid_argument("not the same size!");
    }
    
    T res = 0;
    for(size_t i = 0; i < v1.getSize(); i++)
    {
        res += v1[i] * v2[i];
    }

    return res;
}


template <typename T>
Vector<T> operator*(const Matrix<T>& M, const Vector<T>& v)
{
    if(!v.isColumn())
    {
        throw std::invalid_argument("vector cannot multuply... pls transpose");
    }
    if(v.getSize() != M.get_cols())
    {
        throw std::invalid_argument("vector cannot multuply. Different size");
    }

    Vector<T> res(M.get_rows(), 0, v.isColumn());

    for(size_t i = 0; i < M.get_rows(); i++)
    {
        T val = 0;
        for(size_t j = 0; j < v.getSize(); j++)
        {
            val += M(i, j) * v[j];
        }
        res.set(i, val);
    }

    return res;

}



template <typename T>
Vector<T> operator*(const Vector<T>& v, const Matrix<T>& M)
{
    if(v.isColumn())
    {
        throw std::invalid_argument("vector cannot multuply... pls transpose");
    }
    if(v.getSize() != M.get_rows())
    {
        throw std::invalid_argument("vector cannot multuply. Different size");
    }

    Vector<T> res(M.get_rows(), 0, v.isColumn());

    for(size_t i = 0; i < M.get_cols(); i++)
    {
        T val = 0;
        for(size_t j = 0; j < v.getSize(); j++)
        {
            val += M(i, j) * v[j];
        }
        res.set(i, val);
    }

    return res;

}

template <typename T>
Vector<T> operator*(T scalar, const Vector<T>& vec)
{
    Vector<T> res(vec.getSize(), 1, vec.isColumn());
    for(size_t i = 0; i < vec.getSize(); i++)
    {
        res.set(i, scalar * vec[i]);
    }
    
    return res;
}

template <typename T>
Vector<T> operator+(const Vector<T>& v1, const Vector<T>& v2)
{

    if(v1.getSize() != v2.getSize())
    {
        throw std::invalid_argument("difference dimension");
    }

    Vector<T> res(v1.getSize());
    for(size_t i = 0; i < v1.getSize(); i++)
    {
        T s = v1[i] + v2[i];
        res.set(i,s);
    }
    return res;
}


template <typename T>
Vector<T> operator-(const Vector<T>& v1, const Vector<T>& v2)
{

    if(v1.getSize() != v2.getSize())
    {
        throw std::invalid_argument("difference dimension");
    }

    Vector<T> res(v1.getSize());
    for(size_t i = 0; i < v1.getSize(); i++)
    {
        T s = v1[i] - v2[i];
        res.set(i,s);
    }
    return res;
}



template <typename T>
double len(const Vector<T>& v1)
{
    return std::sqrt(dot_product(v1, v1));
}


}//LIN SPACE

#endif 