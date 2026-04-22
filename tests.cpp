#include <gtest/gtest.h>
#include "linalg.hpp"  // ваш заголовочный файл
#include <cmath>

// Допуск для сравнения чисел с плавающей точкой
const double EPS = 1e-9;

// ------------------------------------------------------------
// Тесты для класса Tensor
// ------------------------------------------------------------

TEST(TensorTest, ConstructorDefault) {
    linalg::Tensor<double> t;
    EXPECT_EQ(t.rows(), 0);
    EXPECT_EQ(t.cols(), 0);
    EXPECT_TRUE(t.get_data().empty());
}

TEST(TensorTest, ConstructorWithInitValue) {
    linalg::Tensor<double> t(2, 3, 5.0);
    EXPECT_EQ(t.rows(), 2);
    EXPECT_EQ(t.cols(), 3);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_DOUBLE_EQ(t(i, j), 5.0);
}

TEST(TensorTest, ConstructorWithVector) {
    std::vector<double> data = {1, 2, 3, 4};
    linalg::Tensor<double> t(2, 2, data);
    EXPECT_EQ(t(0,0), 1);
    EXPECT_EQ(t(0,1), 2);
    EXPECT_EQ(t(1,0), 3);
    EXPECT_EQ(t(1,1), 4);
}

TEST(TensorTest, ConstructorWithVectorWrongSize) {
    std::vector<double> data = {1, 2, 3};
    EXPECT_THROW(linalg::Tensor<double> t(2, 2, data), std::length_error);
}

TEST(TensorTest, AccessOperator) {
    linalg::Tensor<double> t(2, 2, 0.0);
    t(0,1) = 7.5;
    EXPECT_DOUBLE_EQ(t(0,1), 7.5);
    const auto& ct = t;
    EXPECT_DOUBLE_EQ(ct(0,1), 7.5);
}

TEST(TensorTest, TransposeSquare) {
    linalg::Tensor<double> t(3, 3, 0.0);
    t(0,1) = 1; t(0,2) = 2;
    t(1,0) = 3; t(1,2) = 4;
    t(2,0) = 5; t(2,1) = 6;
    t.transpose();
    // Ожидаем: (0,1) было 1 -> теперь (1,0)
    EXPECT_DOUBLE_EQ(t(1,0), 1);
    EXPECT_DOUBLE_EQ(t(2,0), 2);
    EXPECT_DOUBLE_EQ(t(0,1), 3);
    EXPECT_DOUBLE_EQ(t(2,1), 4);
    EXPECT_DOUBLE_EQ(t(0,2), 5);
    EXPECT_DOUBLE_EQ(t(1,2), 6);
}

TEST(TensorTest, TransposeNonSquare) {
    linalg::Tensor<double> t(2, 3, 0.0);
    t(0,0)=1; t(0,1)=2; t(0,2)=3;
    t(1,0)=4; t(1,1)=5; t(1,2)=6;
    t.transpose();
    EXPECT_EQ(t.rows(), 3);
    EXPECT_EQ(t.cols(), 2);
    EXPECT_DOUBLE_EQ(t(0,0), 1);
    EXPECT_DOUBLE_EQ(t(1,0), 2);
    EXPECT_DOUBLE_EQ(t(2,0), 3);
    EXPECT_DOUBLE_EQ(t(0,1), 4);
    EXPECT_DOUBLE_EQ(t(1,1), 5);
    EXPECT_DOUBLE_EQ(t(2,1), 6);
}

TEST(TensorTest, OperatorPlus) {
    linalg::Tensor<double> a(2,2,1.0);
    linalg::Tensor<double> b(2,2,2.0);
    auto c = a + b;
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_DOUBLE_EQ(c(i,j), 3.0);
}

TEST(TensorTest, OperatorMinus) {
    linalg::Tensor<double> a(2,2,5.0);
    linalg::Tensor<double> b(2,2,3.0);
    auto c = a - b;
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_DOUBLE_EQ(c(i,j), 2.0);
}

TEST(TensorTest, OperatorMultiplyScalar) {
    linalg::Tensor<double> t(2,2,2.0);
    auto res = 3.0 * t;
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_DOUBLE_EQ(res(i,j), 6.0);
}

TEST(TensorTest, OperatorDivideScalar) {
    linalg::Tensor<double> t(2,2,6.0);
    auto res = t / 2.0;
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_DOUBLE_EQ(res(i,j), 3.0);
}

TEST(TensorTest, MatrixMultiplication) {
    // 2x3 * 3x2 = 2x2
    linalg::Tensor<double> A(2,3,0.0);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    linalg::Tensor<double> B(3,2,0.0);
    B(0,0)=7; B(0,1)=8;
    B(1,0)=9; B(1,1)=10;
    B(2,0)=11; B(2,1)=12;
    auto C = A * B;
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    // Ручной расчёт
    EXPECT_DOUBLE_EQ(C(0,0), 1*7+2*9+3*11);  // 58
    EXPECT_DOUBLE_EQ(C(0,1), 1*8+2*10+3*12); // 64
    EXPECT_DOUBLE_EQ(C(1,0), 4*7+5*9+6*11);  // 139
    EXPECT_DOUBLE_EQ(C(1,1), 4*8+5*10+6*12); // 154
}

TEST(TensorTest, MatrixMultiplicationWrongDim) {
    linalg::Tensor<double> A(2,3);
    linalg::Tensor<double> B(2,2);
    EXPECT_THROW(A * B, std::length_error);
}

TEST(TensorTest, HadamardProduct) {
    linalg::Tensor<double> A(2,2,0.0);
    A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    linalg::Tensor<double> B(2,2,0.0);
    B(0,0)=5; B(0,1)=6; B(1,0)=7; B(1,1)=8;
    auto C = hadamard_product(A, B); // опечатка в имени функции в коде: hadamar_product
    EXPECT_DOUBLE_EQ(C(0,0), 5);
    EXPECT_DOUBLE_EQ(C(0,1), 12);
    EXPECT_DOUBLE_EQ(C(1,0), 21);
    EXPECT_DOUBLE_EQ(C(1,1), 32);
}

TEST(TensorTest, ApplyFunction) {
    linalg::Tensor<double> t(2,2,2.0);
    auto res = apply(t, [](double x) { return x * x; });
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_DOUBLE_EQ(res(i,j), 4.0);
}

TEST(TensorTest, TransposeFunction) {
    linalg::Tensor<double> t(2,3);
    t(0,0)=1; t(0,1)=2; t(0,2)=3;
    t(1,0)=4; t(1,1)=5; t(1,2)=6;
    auto tr = transpose(t);
    EXPECT_EQ(tr.rows(), 3);
    EXPECT_EQ(tr.cols(), 2);
    EXPECT_DOUBLE_EQ(tr(0,0),1); EXPECT_DOUBLE_EQ(tr(1,0),2); EXPECT_DOUBLE_EQ(tr(2,0),3);
    EXPECT_DOUBLE_EQ(tr(0,1),4); EXPECT_DOUBLE_EQ(tr(1,1),5); EXPECT_DOUBLE_EQ(tr(2,1),6);
}

// ------------------------------------------------------------
// Тесты для класса vec
// ------------------------------------------------------------

TEST(VecTest, ConstructorAndSize) {
    linalg::vec<double> v(5);
    EXPECT_EQ(v.size(), 5);
    EXPECT_EQ(v.rows(), 5);
    EXPECT_EQ(v.cols(), 1);
}

TEST(VecTest, AccessOperator) {
    linalg::vec<double> v(3, 0.0);
    v[0] = 1.0;
    v[1] = 2.0;
    v[2] = 3.0;
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[1], 2.0);
    EXPECT_DOUBLE_EQ(v[2], 3.0);
    const auto& cv = v;
    EXPECT_DOUBLE_EQ(cv[0], 1.0);
}

TEST(VecTest, Length) {
    linalg::vec<double> v(3);
    v[0] = 3.0; v[1] = 4.0; v[2] = 0.0;
    EXPECT_DOUBLE_EQ(v.length(), 5.0);
}

TEST(VecTest, Norm) {
    linalg::vec<double> v(3);
    v[0] = 3.0; v[1] = 4.0; v[2] = 0.0;
    v.norm();
    EXPECT_NEAR(v[0], 0.6, EPS);
    EXPECT_NEAR(v[1], 0.8, EPS);
    EXPECT_NEAR(v[2], 0.0, EPS);
    EXPECT_NEAR(v.length(), 1.0, EPS);
}

TEST(VecTest, DotProduct) {
    linalg::vec<double> a(3, {1,2,3});
    linalg::vec<double> b(3, {4,5,6});
    double dot = dot_product(a, b);
    EXPECT_DOUBLE_EQ(dot, 1*4 + 2*5 + 3*6); // 32
}

TEST(VecTest, DotProductWrongSize) {
    linalg::vec<double> a(3);
    linalg::vec<double> b(4);
    EXPECT_THROW(dot_product(a, b), std::invalid_argument);
}

TEST(VecTest, CrossProduct) {
    linalg::vec<double> a(3, {1,0,0});
    linalg::vec<double> b(3, {0,1,0});
    auto c = cross(a, b);
    EXPECT_DOUBLE_EQ(c(0,0), 0);
    EXPECT_DOUBLE_EQ(c(1,0), 0);
    EXPECT_DOUBLE_EQ(c(2,0), 1);
}

TEST(VecTest, CrossProductWrongDim) {
    linalg::vec<double> a(4);
    linalg::vec<double> b(4);
    EXPECT_THROW(cross(a, b), std::invalid_argument);
}

TEST(VecTest, OuterProduct) {
    linalg::vec<double> a(3, {1,2,3});
    linalg::vec<double> b(2, std::vector<double>{4,5});
    auto M = outer_product(a, b);
    EXPECT_EQ(M.rows(), 3);
    EXPECT_EQ(M.cols(), 2);
    EXPECT_DOUBLE_EQ(M(0,0), 4); EXPECT_DOUBLE_EQ(M(0,1), 5);
    EXPECT_DOUBLE_EQ(M(1,0), 8); EXPECT_DOUBLE_EQ(M(1,1),10);
    EXPECT_DOUBLE_EQ(M(2,0),12); EXPECT_DOUBLE_EQ(M(2,1),15);
}

TEST(VecTest, ConstructFromTensor) {
    linalg::Tensor<double> t(5,1, 3.14);
    linalg::vec<double> v(t);
    EXPECT_EQ(v.size(), 5);
    EXPECT_DOUBLE_EQ(v[0], 3.14);
    EXPECT_THROW(linalg::vec<double> v2(linalg::Tensor<double>(3,3)), std::invalid_argument);
}

// ------------------------------------------------------------
// Тесты на исключения и граничные случаи
// ------------------------------------------------------------

TEST(EdgeTest, ZeroSizeTensor) {
    linalg::Tensor<double> t(0,5);
    EXPECT_EQ(t.rows(), 0);
    EXPECT_EQ(t.cols(), 5);
    // Умножение с нулевым размером не должно падать
    linalg::Tensor<double> a(0,3);
    linalg::Tensor<double> b(3,0);
    auto c = a * b;
    EXPECT_EQ(c.rows(), 0);
    EXPECT_EQ(c.cols(), 0);
}

TEST(EdgeTest, VectorOfLengthOne) {
    linalg::vec<double> v(1, 5.0);
    EXPECT_DOUBLE_EQ(v.length(), 5.0);
    v.norm();
    EXPECT_DOUBLE_EQ(v[0], 1.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}