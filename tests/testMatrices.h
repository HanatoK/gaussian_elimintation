#ifndef TESTMATRICES_H
#define TESTMATRICES_H

#include "Matrix.h"

namespace TEST {

Matrix matA{{ 2.0,  0.5,  1.0, -2.0,  3.0},
            { 0.5,  1.0,  0.1,  4.0, -9.0},
            { 1.0,  0.1, -3.0, -2.0,  0.0},
            {-2.0,  4.0, -2.0,  0.2, -1.0},
            { 3.0, -9.0,  0.0, -1.0, -0.3}};
Matrix matB{{-2.7, -0.5},
            { 2.5, -1.9},
            { 1.2,  7.1},
            {-2.8, -4.0},
            {-0.1, -9.6}};
Matrix matC{{0.0, 1.2, -3.0},
            {-5.0, 0.5, 1.3},
            {2.0, 0.6, 0}};

}

#endif // TESTMATRICES_H
