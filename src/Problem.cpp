#include "Problem.h"

#include <functional>
#include <utility>
#include <vector>
#include <pagmo/types.hpp>

#include "diffusion_tensor.h"

#include <iostream>

pagmo::vector_double
Problem
::get_true_dv(pagmo::vector_double const & scaled_dv)
{
    auto true_dv = scaled_dv;
    for(std::size_t i=0, end=true_dv.size(); i!=end; ++i)
    {
        auto const & true_min = Problem::_true_bounds.first[i];
        auto const & true_max = Problem::_true_bounds.second[i];
        
        // All elements in scaled_dv are in [-1,1]: rescale to true interval.
        true_dv[i] = (scaled_dv[i] - -1)/2 * (true_max-true_min) + true_min;
    }
    return true_dv;
}

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
::fitness(pagmo::vector_double const & scaled_dv) const
{
    auto const true_dv = Problem::get_true_dv(scaled_dv);
    auto const D = Problem::get_diffusion_tensor(true_dv);
    
    auto const & non_dw_acquisition = this->scheme[this->non_dw_index];
    auto const non_dw_signal = this->signals[this->non_dw_index];
    
    auto const simulated_signal_non_dw = this->simulator(
        this->T1, this->T2, this->B1, D, non_dw_acquisition);
    
    double residuals = 0;
    for(std::size_t i=0, end=this->scheme.size(); i!=end; ++i)
    {
        if(i == this->non_dw_index)
        {
            continue;
        }
        
        auto const & acquisition = this->scheme[i];
        auto measured_signal = this->signals[i]/non_dw_signal;
        
        auto const simulated_signal_dw = this->simulator(
            this->T1, this->T2, this->B1, D, acquisition);
        
        auto const simulated_signal = simulated_signal_dw/simulated_signal_non_dw;
        
        // The signal is normalized and < 1: don't use square norm, but absolute
        // value to avoid too low residuals.
        residuals += std::abs(simulated_signal-measured_signal);
    }
    
    // Normalize by number of acquisition.
    return {residuals/(this->scheme.size()-1)};
}

std::pair<pagmo::vector_double, pagmo::vector_double>
Problem
::get_bounds() const
{
    pagmo::vector_double minimum(Variables::size, -1);
    pagmo::vector_double maximum(Variables::size, +1);
    
    return {minimum, maximum};
}


std::pair<pagmo::vector_double, pagmo::vector_double> const
Problem
::_true_bounds{
    {0, 0, -M_PI, 1  *1e-12, 1  *1e-12, 1  *1e-12},
    {1, 1, +M_PI, 1e4*1e-12, 1e4*1e-12, 1e4*1e-12}};
