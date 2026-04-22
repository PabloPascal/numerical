
#include "src/linalg.hpp"
#include <chrono>
#include <iostream>
#include <array>
#include <functional>


void print_matrix(const linalg::Tensor<float>& a){

    for(size_t i = 0; i < a.rows(); i++){
        for(size_t j = 0; j < a.cols(); j++){
            std::cout << a(i, j) << " ";
        }
        std::cout << "\n";
    }


}


template <std::floating_point T>
std::vector<T> benchmark(size_t count, 
    const linalg::Tensor<T>& a, const linalg::Tensor<T>& b, 
    std::function<linalg::Tensor<T>(linalg::Tensor<T>, linalg::Tensor<T>)> op) 
{
    std::vector<float> save;

    for(size_t i = 0; i < count; i++)
    {
        auto t_start = std::chrono::steady_clock::now();
        op(a, b);
        auto t_end = std::chrono::steady_clock::now();

        save.emplace_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()
        );
    }
    return save;
}


std::array<float, 4> statistic(const std::vector<float>& distr)
{
    float mean = 0;
    float std = 0;
    for(float x : distr){
        mean += x;
    }
    mean /= distr.size();

    for(float x : distr){
        std += (x-mean)*(x-mean);
    }
    std /= distr.size();
    std = std::sqrt(std);


    return {*std::min_element(distr.begin(), distr.end()),
            *std::max_element(distr.begin(), distr.end()),
            mean,
            std
            };

}



void time_test()
{
    size_t count = 300;

    linalg::Tensor<float> bigA(1000, 1000, {0, 100}, true);
    linalg::Tensor<float> bigB(1000, 1000, {0, 100}, true);

    // Multiply time test
    std::cout << "Multiply time test\n";  

    auto mat_mul = [](const linalg::Tensor<float>& a, const linalg::Tensor<float>& b){
        return a*b;
    };

    std::vector<float> distr = benchmark<float>(count, bigA, bigB, mat_mul);
    std::array<float, 4> res = statistic(distr);

    printf("count = %d \n", count);
    printf("min = %f, max = %f, mean = %f, std = %f \n", res[0], res[1], res[2], res[3]);


    std::cout << "Sum time test\n";
    distr = benchmark<float>(count, bigA, bigB, linalg::operator+<float>);
    res = statistic(distr);

    printf("count = %d \n", count);
    printf("min = %f, max = %f, mean = %f, std = %f \n", res[0], res[1], res[2], res[3]);



}



void test_sum()
{
    linalg::Tensor<float> a(2,2, {1,2,3,4});
    linalg::Tensor<float> b(2,2, {2,3,4,2});

    print_matrix(a);
    std::cout << "\n";
    print_matrix(b);
    std::cout << "\n";
    print_matrix(a+b);

}


void test_mult()
{
    linalg::vec<float> a(3, {1,2,3});
    linalg::vec<float> b(3, {2,2,2});

    float dot = linalg::dot_product(a,b);

    std::cout << dot << std::endl;
}



void test_transpose()
{
    linalg::Tensor<float> a(3,2, std::make_pair(0, 5), false);


    print_matrix(a);
    std::cout << "\n";

    linalg::Tensor<float> b = linalg::transpose(a);

    print_matrix(b);
    std::cout << "\n";

}




void test_hadamar()
{
    linalg::vec<float> a(5, {1,2,3,4,5});
    linalg::vec<float> b(5, 2);
    
    auto c = linalg::hadamard_product(a, b);
    
    print_matrix(c);

}


void test_product_with_scalar()
{
    linalg::Tensor<float> a(3,2, {1,2,3,4,5,6});
    float s = 3;
    
    auto c = s * a;
    
    print_matrix(c);

}


void test_apply_func()
{
    linalg::Tensor<float> a(3,2, {1,2,3,4,5,6});
    
    auto func = [](float s){return std::sin(s);};
    
    auto c = linalg::apply<float>(a, func);
    
    print_matrix(c);

}



void test_cross_product(){
    linalg::vec<float> v1(3, {1,2,2});
    linalg::vec<float> v2(3, {2,1,3});

    auto v3 = linalg::cross(v1, v2);

    print_matrix(v3);

}
