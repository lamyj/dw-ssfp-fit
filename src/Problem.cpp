#include "Problem.h"

#include <utility>
#include <pagmo/types.hpp>

#include "diffusion_tensor.h"

pagmo::vector_double
Problem
::fitness(pagmo::vector_double const & dv) const
{
    double theta, phi;
    std::tie(theta, phi) = uniform_to_spherical(
        dv[static_cast<int>(Variables::u)], dv[static_cast<int>(Variables::v)]);
    auto const D = build_diffusion_tensor(
        theta, phi, dv[static_cast<int>(Variables::psi)], 
        dv[static_cast<int>(Variables::lambda1)],
        dv[static_cast<int>(Variables::lambda2)],
        dv[static_cast<int>(Variables::lambda3)]);
    
    auto const & reference_acquisition = 
        this->acquisitions[reference_acquisition_index];
    auto const reference_signal = 
        this->signals[reference_acquisition_index];
    
    double residuals = 0;
    for(std::size_t i=0, end=this->acquisitions.size(); i!=end; ++i)
    {
        if(i == this->reference_acquisition_index)
        {
            continue;
        }
        
        auto const & acquisition = this->acquisitions[i];
        auto measured_signal = this->signals[i]/reference_signal;
        
        auto const simulated_signal = this->simulator(
            this->T1, this->T2, D, acquisition, reference_acquisition);
        residuals += std::pow(simulated_signal-measured_signal, 2);
    }
    
    return {residuals};
}

std::pair<pagmo::vector_double, pagmo::vector_double>
Problem
::get_bounds() const
{
    pagmo::vector_double minimum(static_cast<int>(Variables::size));
    pagmo::vector_double maximum(static_cast<int>(Variables::size));
    
    minimum[static_cast<int>(Variables::u)] = 0;
    maximum[static_cast<int>(Variables::u)] = 1;
    
    minimum[static_cast<int>(Variables::v)] = 0;
    maximum[static_cast<int>(Variables::v)] = 1;
    
    minimum[static_cast<int>(Variables::psi)] = -M_PI;
    maximum[static_cast<int>(Variables::psi)] = +M_PI;
    
    minimum[static_cast<int>(Variables::lambda1)] = 1e-3;
    maximum[static_cast<int>(Variables::lambda1)] = 1e+3;
    
    minimum[static_cast<int>(Variables::lambda2)] = 1e-3;
    maximum[static_cast<int>(Variables::lambda2)] = 1e+3;
    
    minimum[static_cast<int>(Variables::lambda3)] = 1e-3;
    maximum[static_cast<int>(Variables::lambda3)] = 1e+3;
    
    return {minimum, maximum};
}
