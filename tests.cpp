
#include "linalg.hpp"
#include <chrono>
#include <iostream>
#include <array>
#include <functional>


void print_matrix(const linalg::Tensor<float>& a){

    for(size_t i = 0; i < a.cols(); i++){
        for(size_t j = 0; j < a.rows(); j++){
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


    std::vector<float> distr = benchmark<float>(count, bigA, bigB, linalg::operator*<float>);
    std::array<float, 4> res = statistic(distr);

    printf("count = %d \n", count);
    printf("min = %f, max = %f, mean = %f, std = %f \n", res[0], res[1], res[2], res[3]);

}



void test_sum()
{
size_t count = 300;

    linalg::Tensor<float> bigA(1000, 1000, {0, 100}, true);
    linalg::Tensor<float> bigB(1000, 1000, {0, 100}, true);


    std::vector<float> distr = benchmark<float>(count, bigA, bigB, linalg::operator+<float>);
    std::array<float, 4> res = statistic(distr);

    printf("count = %d \n", count);
    printf("min = %f, max = %f, mean = %f, std = %f \n", res[0], res[1], res[2], res[3]);


}


void test_mult()
{
size_t count = 300;

    linalg::Tensor<float> bigA(1000, 1000, {0, 100}, true);
    linalg::Tensor<float> bigB(1000, 1000, {0, 100}, true);


    std::vector<float> distr = benchmark<float>(count, bigA, bigB, linalg::operator+<float>);
    std::array<float, 4> res = statistic(distr);

    printf("count = %d \n", count);
    printf("min = %f, max = %f, mean = %f, std = %f \n", res[0], res[1], res[2], res[3]);

}
