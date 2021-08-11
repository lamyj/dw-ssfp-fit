#include "diffusion_tensor.h"

#include <cmath>
#include <utility>

#include <Eigen/Core>

std::pair<double, double> uniform_to_spherical(double u, double v)
{
    return {2*M_PI*u, std::acos(2*v-1)};
}

Eigen::Matrix3d build_diffusion_tensor(
    double theta, double phi, double psi, 
    double lambda1, double lambda2, double lambda3)
{
    using std::sin, std::cos, std::pow;
    
    Eigen::Vector3d const axis{
        cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)};
    
    // Rodrigue's formula for a rotation of an angle psi around the axis.
    Eigen::Matrix3d V;
    V(0,0) = cos(psi) + pow(axis[0], 2) * (1-cos(psi));
    V(1,1) = cos(psi) + pow(axis[1], 2) * (1-cos(psi));
    V(2,2) = cos(psi) + pow(axis[2], 2) * (1-cos(psi));
    
    V(0,1) = -axis[2]*sin(psi) + axis[0]*axis[1]*(1-cos(psi));
    V(1,0) =  axis[2]*sin(psi) + axis[0]*axis[1]*(1-cos(psi));

    V(0,2) =  axis[1]*sin(psi) + axis[0]*axis[2]*(1-cos(psi));
    V(2,0) = -axis[1]*sin(psi) + axis[0]*axis[2]*(1-cos(psi));

    V(1,2) = -axis[0]*sin(psi) + axis[1]*axis[2]*(1-cos(psi));
    V(2,1) =  axis[0]*sin(psi) + axis[1]*axis[2]*(1-cos(psi));
    
    // Eigen-recomposition of the tensor.
    Eigen::Vector3d const lambda{lambda1, lambda2, lambda3};
    Eigen::Matrix3d const D = V * lambda.asDiagonal() * V.transpose();
    return D;
}
