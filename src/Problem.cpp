#include "Problem.h"

#include <functional>
#include <utility>
#include <pagmo/types.hpp>

#include "diffusion_tensor.h"

Eigen::Matrix3d
Problem
::get_diffusion_tensor(pagmo::vector_double const & dv)
{
    auto const [theta, phi] = uniform_to_spherical(
        dv[Variables::u], dv[Variables::v]);
    return build_diffusion_tensor(
        theta, phi, dv[Variables::psi], 
        dv[Variables::lambda1], dv[Variables::lambda2], dv[Variables::lambda3]);
}

pagmo::vector_double
Problem
::fitness(pagmo::vector_double const & dv) const
{
    auto const D = Problem::get_diffusion_tensor(dv);
    
    auto const & non_dw_acquisition = this->scheme[this->non_dw_index];
    auto const non_dw_signal = this->signals[this->non_dw_index];
    
    double residuals = 0;
    double relative_residuals = 0;
    for(std::size_t i=0, end=this->scheme.size(); i!=end; ++i)
    {
        if(i == this->non_dw_index)
        {
            continue;
        }
        
        auto const & acquisition = this->scheme[i];
        auto measured_signal = this->signals[i]/non_dw_signal;
        
        auto const simulated_signal = this->simulator(
            this->T1, this->T2, D, acquisition, non_dw_acquisition);
        residuals += std::pow(simulated_signal-measured_signal, 2);
    }
    
    return {residuals};
}

std::pair<pagmo::vector_double, pagmo::vector_double>
Problem
::get_bounds() const
{
    pagmo::vector_double minimum(Variables::size);
    pagmo::vector_double maximum(Variables::size);
    
    minimum[Variables::u] = 0; maximum[Variables::u] = 1;
    minimum[Variables::v] = 0; maximum[Variables::v] = 1;
    
    minimum[Variables::psi] = -M_PI; maximum[Variables::psi] = +M_PI;
    
    // Eigenvalues between 1 µm²/s and 10000 µm²/s
    minimum[Variables::lambda1] = 1 * 1e-12; maximum[Variables::lambda1] = 1e+4 * 1e-12;
    minimum[Variables::lambda2] = 1 * 1e-12; maximum[Variables::lambda2] = 1e+4 * 1e-12;
    minimum[Variables::lambda3] = 1 * 1e-12; maximum[Variables::lambda3] = 1e+4 * 1e-12;
    
    return {minimum, maximum};
}
