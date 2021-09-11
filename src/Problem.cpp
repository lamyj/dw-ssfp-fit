#include "Problem.h"

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pagmo/types.hpp>
#include <sycomore/Array.h>
#include <sycomore/Quantity.h>
#include <sycomore/Species.h>
#include <sycomore/units.h>

#include "Acquisition.h"
#include "diffusion_tensor.h"

Problem
::Problem(
    std::vector<Acquisition> const & scheme, unsigned int reference_index,
    std::vector<double> const & signals,
    std::optional<double> T1, std::optional<double> T2, double B1,
    Simulator simulator)
: _scheme(scheme), _reference_index(reference_index), _signals(signals),
    _T1(T1), _T2(T2), _B1(B1), _simulator(simulator)
{
    this->_true_bounds = {
        {0, 0, -M_PI, 1  *1e-12, 1  *1e-12, 1  *1e-12},
        {1, 1, +M_PI, 1e4*1e-12, 1e4*1e-12, 1e4*1e-12}};
    
    // NOTE: the DV use R1, R2 and not T1, T2 for better contrast and hence
    // better convergence
    if(!this->_T1)
    {
        // Min T1 of 2 ms -> max R1 of 0.5 Khz
        this->_true_bounds.second.push_back(1./2e-3);
        // Max T1 of 20 s -> min R1 of 0.05 Hz
        this->_true_bounds.first.push_back(1./20);
    }
    if(!this->_T2)
    {
        // Min T2 of 1 ms -> max R2 of 1 Khz
        this->_true_bounds.second.push_back(1./1e-3);
        // Max T1 of 10 s -> min R2 of 0.05 Hz
        this->_true_bounds.first.push_back(1./10);
    }
}

Problem::Vector
Problem
::get_true_dv(Vector const & scaled_dv) const
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

sycomore::Array<sycomore::Quantity>
Problem
::get_diffusion_tensor(Vector const & dv) const
{
    using namespace sycomore::units;
    
    auto const [theta, phi] = uniform_to_spherical(
        dv[Variables::u], dv[Variables::v]);
    auto const matrix = build_diffusion_tensor(
        theta, phi, dv[Variables::psi], 
        dv[Variables::lambda1], dv[Variables::lambda2], dv[Variables::lambda3]);
    sycomore::Array<sycomore::Quantity> D(9);
    for(unsigned int row=0; row<3; ++row)
    {
        for(unsigned int col=0; col<3; ++col)
        {
            D[3*row + col] = matrix(row, col) * m*m/s;
        }
    }
    return D;
}

sycomore::Quantity
Problem
::get_T1(Vector const & dv) const
{
    // NOTE: DV stores R1, not T1
    // If T1 is a variable, it is the previous-to-last item if T2 is also a
    // variable, otherwise the last one.
    return sycomore::units::s*(
        this->_T1 ? this->_T1.value() : 1./dv[dv.size()-(this->_T2?1:2)]);
}

sycomore::Quantity
Problem
::get_T2(Vector const & dv) const
{
    // NOTE: DV stores R2, not T2
    // If T2 is a variable, it is always the last item.
    return sycomore::units::s*(this->_T2 ? this->_T2.value(): 1./dv.back());
}

pagmo::vector_double
Problem
::fitness(Vector const & scaled_dv) const
{
    auto const true_dv = this->get_true_dv(scaled_dv);
    auto const D = this->get_diffusion_tensor(true_dv);
    auto const T1 = this->get_T1(true_dv);
    auto const T2 = this->get_T2(true_dv);
    
    auto const & reference_acquisition = this->_scheme[this->_reference_index];
    auto const reference_signal = this->_signals[this->_reference_index];
    if(reference_signal == 0.)
    {
        throw std::runtime_error("Non DW signal is null");
    }
    
    sycomore::Species const species(T1, T2, D);
    
    // NOTE: when e.g. T2 is too short, the threshold used in the simulator will
    // cause a return value of 0. Since we divide by this value, clamp it to a
    // non-0 value.
    auto const simulated_signal_reference = std::max(
        this->_simulator(species, reference_acquisition, this->_B1),
        1e-12);
    
    double residuals = 0;
    for(std::size_t i=0, end=this->_scheme.size(); i!=end; ++i)
    {
        if(i == this->_reference_index)
        {
            continue;
        }
        
        auto const & acquisition = this->_scheme[i];
        auto measured_ratio = this->_signals[i]/reference_signal;
        
        auto const simulated_signal = this->_simulator(
            species, acquisition, this->_B1);
        
        auto const simulated_ratio = simulated_signal/simulated_signal_reference;
        
        residuals += std::abs(simulated_ratio-measured_ratio);
    }
    
    // Normalize by number of acquisition.
    return {residuals/(this->_scheme.size()-1)};
}

std::pair<Problem::Vector, Problem::Vector>
Problem
::get_bounds() const
{
    // Start from the tensor, add 1 for T1/T2 when not specified by the user.
    auto const size = Variables::size + (this->_T1?0:1) + (this->_T2?0:1);
    pagmo::vector_double minimum(size, -1);
    pagmo::vector_double maximum(size, +1);
    
    return {minimum, maximum};
}
