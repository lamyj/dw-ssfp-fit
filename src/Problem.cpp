#include "Problem.h"

#include <functional>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>
#include <pagmo/types.hpp>
#include <sycomore/Species.h>
#include <sycomore/units.h>

#include "diffusion_tensor.h"

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
    using namespace sycomore::units;
    
    auto const true_dv = Problem::get_true_dv(scaled_dv);
    auto const D = Problem::get_diffusion_tensor(true_dv);
    
    auto const & non_dw_acquisition = this->scheme[this->non_dw_index];
    auto const non_dw_signal = this->signals[this->non_dw_index];
    if(non_dw_signal == 0.)
    {
        throw std::runtime_error("Non DW signal is null");
    }
    
    sycomore::Array<sycomore::Quantity> D_(9);
    for(unsigned int row=0; row<3; ++row)
    {
        for(unsigned int col=0; col<3; ++col)
        {
            D_[3*row + col] = D(row, col) * std::pow(m, 2)/s;
        }
    }
    
    sycomore::Species const species(this->T1*s, this->T2*s, D_);
    
    auto const simulated_signal_non_dw = this->simulator(
        species, non_dw_acquisition, this->B1);
    if(simulated_signal_non_dw == 0.)
    {
        std::cerr
            << "Simulated non DW signal is null: "
            << "alpha=" << non_dw_acquisition.alpha << ", "
            << "G_diffusion=" << non_dw_acquisition.G_diffusion << ", "
            << "tau_diffusion=" << non_dw_acquisition.tau_diffusion << ", "
            << "direction=" << non_dw_acquisition.direction << ", "
            << "TE=" << non_dw_acquisition.TE << ", "
            << "TR=" << non_dw_acquisition.TR << ", "
            << "pixel_bandwidth=" << non_dw_acquisition.pixel_bandwidth << ", "
            << "resolution=" << non_dw_acquisition.resolution << ", "
            << "G_max=" << non_dw_acquisition.G_max << ", "
            << "T1=" << this->T1 << ", "
            << "T2=" << this->T2 << ", "
            << "B1=" << this->B1 << ", "
            << "D=" << D
            << std::endl;
        return {std::numeric_limits<pagmo::vector_double::value_type>::max()};
        // throw std::runtime_error("Simulated non DW signal is null");
    }
    
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
            species, acquisition, this->B1);
        
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
