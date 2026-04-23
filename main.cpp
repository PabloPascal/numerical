#include <iostream>
#include "src/linalg.hpp"
#include <chrono>
#include <math.h>
#include "src/matrix_type.hpp"
#include "src/utils.hpp"

using namespace linalg;

int main()
{

    auto rot = Rotate2d<float>(30);

    printf("%f %f \n%f %f", rot(0,0), rot(0,1), rot(1,0), rot(1,1));

    print_matrix(rot);

    return 0;
}
