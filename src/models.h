#ifndef _96a3fddd_ce2d_434d_b798_2ee72aba3278
#define _96a3fddd_ce2d_434d_b798_2ee72aba3278

#include <Eigen/Core>
#include "Acquisition.h"

double freed(
    double T1, double T2, Eigen::Matrix3d const & D, 
    Acquisition const & dw, Acquisition const & non_dw);

double freed(
    double T1, double T2, Eigen::Matrix3d const & D, 
    Acquisition const & acquisition);

#endif // _96a3fddd_ce2d_434d_b798_2ee72aba3278
