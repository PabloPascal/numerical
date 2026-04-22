#include <iostream>
#include "src/linalg.hpp"
#include <chrono>
#include <math.h>
#include "src/matrix_type.hpp"


using namespace linalg;

int main()
{

    linalg::vec<double> a(4);
    linalg::vec<double> b(4);

    

    cross(a,b);

    //printf("Hello\n");


    return 0;
}
