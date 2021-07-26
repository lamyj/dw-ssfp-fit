#include "Acquisition.h"

#include <Eigen/Core>
#include <sycomore/Quantity.h>
#include <sycomore/sycomore.h>
#include <sycomore/TimeInterval.h>
#include <sycomore/units.h>

using namespace sycomore::units;

Acquisition
::Acquisition(
    sycomore::Quantity alpha, 
    sycomore::Quantity G_diffusion, sycomore::Quantity tau_diffusion,
        Eigen::Vector3d direction,
    sycomore::Quantity TE, sycomore::Quantity TR,
    sycomore::Quantity pixel_bandwidth, sycomore::Quantity resolution,
    sycomore::Quantity G_max)
: alpha(alpha),
    G_diffusion(G_diffusion), tau_diffusion(tau_diffusion), direction(direction),
    TE(TE), TR(TR),
    pixel_bandwidth(pixel_bandwidth), resolution(resolution), G_max(G_max)
{
    this->direction.normalize();
    this->update_intervals();
}

Acquisition
::Acquisition(
    double alpha, 
    double G_diffusion, double tau_diffusion, Eigen::Vector3d direction,
    double TE, double TR,
    double pixel_bandwidth, double resolution, double G_max)
: Acquisition(
    alpha*rad, 
    G_diffusion*T/m, tau_diffusion*s, direction,
    TE*s, TR*s,
    pixel_bandwidth*Hz, resolution*m, G_max*T/m)
{
    // Nothing else.
}

void
Acquisition
::update_intervals()
{
    using sycomore::units::s;
    
    // Frequency-encoding gradient for the whole readout
    // Handbook, eq. 8.13, p. 252, with Δν = nₓ Δνₚₚ / 2 (eq. 8.10), 
    // T_acq = 1/pixel_bandwidth
    auto const frequency_encoding_amplitude = 
        2*M_PI * this->pixel_bandwidth / (sycomore::gamma*this->resolution);
    if(frequency_encoding_amplitude > this->G_max)
    {
        throw std::runtime_error("G_max too low for frequency encoding");
    }
    
    auto const frequency_encoding_area =
        frequency_encoding_amplitude * 1/this->pixel_bandwidth;
    
    this->ro_plus = {
        0.5 * 1./this->pixel_bandwidth,
        {frequency_encoding_area/2, 0*T/m*s, 0*T/m*s}};
    this->ro_minus = {
        0.5 * frequency_encoding_area/this->G_max,
        -this->ro_plus.get_gradient_area()};
    
    this->idle = {
        0.5*(
            TE - tau_diffusion
            - this->ro_plus.get_duration() - this->ro_minus.get_duration())};
    if(idle.get_duration() < 0*s)
    {
        throw std::runtime_error("TE-readout short low to accomodate diffusion");
    }
    
    this->diffusion = {
        this->tau_diffusion, 
        {
            this->direction[0]*this->G_diffusion, 
            this->direction[1]*this->G_diffusion, 
            this->direction[2]*this->G_diffusion}};
    
    this->end_of_TR = {
            this->TR - this->TE
            - this->ro_plus.get_duration() - this->ro_minus.get_duration()};
    if(this->end_of_TR.get_duration() < 0*s)
    {
        throw std::runtime_error("TR too low to accomodate readout");
    }
    
    auto const total = 
        this->idle.get_duration() + this->diffusion.get_duration() + this->idle.get_duration()
        + this->ro_minus.get_duration() + this->ro_plus.get_duration()
        + this->ro_plus.get_duration() + this->ro_minus.get_duration()
        + this->end_of_TR.get_duration();
    if((total-this->TR)/this->TR > 1e-6)
    {
        throw std::runtime_error("Timing mismatch");
    }
}
