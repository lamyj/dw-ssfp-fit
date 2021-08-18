#include "Acquisition.h"

#include <map>
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
    sycomore::Quantity pixel_bandwidth,
    Eigen::Vector2i shape, sycomore::Array<sycomore::Quantity> FOV,
    unsigned int train_length, sycomore::Quantity G_max)
: alpha(alpha),
    G_diffusion(G_diffusion), tau_diffusion(tau_diffusion), direction(direction),
    TE(TE), TR(TR),
    pixel_bandwidth(pixel_bandwidth), shape(shape), FOV(FOV),
    train_length(train_length), G_max(G_max)
{
    this->direction.normalize();
    this->update_intervals();
}

Acquisition
::Acquisition(
    double alpha, 
    double G_diffusion, double tau_diffusion, Eigen::Vector3d direction,
    double TE, double TR,
    double pixel_bandwidth, Eigen::Vector2i shape, Eigen::Vector2d FOV,
    unsigned int train_length, double G_max)
: Acquisition(
    alpha*rad, 
    G_diffusion*T/m, tau_diffusion*s, direction,
    TE*s, TR*s,
    pixel_bandwidth*Hz, shape, {FOV[0]*m, FOV[1]*m}, train_length, G_max*T/m)
{
    // Nothing else.
}

void
Acquisition
::update_intervals()
{
    auto const readout_duration = 1/this->pixel_bandwidth;
    
    // Frequency-encoding gradient for the whole readout
    // Handbook, eq. 8.13, p. 252, with Δν = nₓ Δνₚₚ / 2 (eq. 8.10), 
    // T_acq = 1/pixel_bandwidth
    auto const frequency_encoding_amplitude = 
        (2*M_PI*this->shape[0]) / (sycomore::gamma*readout_duration*this->FOV[0]);
    if(frequency_encoding_amplitude > this->G_max)
    {
        throw std::runtime_error("G_max too low for frequency encoding");
    }
    auto const frequency_encoding_area =
        frequency_encoding_amplitude * readout_duration;
    
    // Maximum value of the phase-encoding gradient (border of the k-space).
    // Handbook, eq. 8.38, p. 262
    auto const max_phase_encoding_area = 
        M_PI * (this->shape[1]-1) / (sycomore::gamma*this->FOV[1]);
    
    // Readout preparation: go to the corner of the k-space in the shortest
    // time possible.
    auto const max_encoding_area = std::max(
        frequency_encoding_area/2, max_phase_encoding_area);
    this->readout_preparation = {
        max_encoding_area/this->G_max,
        {
            -frequency_encoding_area/2, 
            this->train_length>1 ? -max_phase_encoding_area : 0*T/m*s,
            0*T/m*s}};
    if(this->readout_preparation.get_gradient_amplitude()[1] > this->G_max)
    {
        throw std::runtime_error("G_max too low for phase encoding");
    }
    
    // Rewind: balance the spatial encoding gradients
    this->readout_rewind = {
        max_encoding_area/this->G_max,
        {
            ((this->train_length%2 == 0) ? +1 : -1)*frequency_encoding_area/2, 
            this->train_length>1 ? -max_phase_encoding_area : 0*T/m*s, 
            0*T/m*s}};
    
    // Half-readout lobes, positive and negative polarity
    this->half_readout = {
        {0, sycomore::TimeInterval(0*s)},
        {+1, sycomore::TimeInterval(
            0.5*readout_duration,
            {+frequency_encoding_area/2, 0*T/m*s, 0*T/m*s})},
        {-1, sycomore::TimeInterval(
            0.5*readout_duration,
            {-frequency_encoding_area/2, 0*T/m*s, 0*T/m*s})}};
    
    // With n echoes, there are n-1 phase blips to cover the whole k-space
    auto const phase_blip_area = 
        (train_length > 1)
        ? 2*max_phase_encoding_area/(train_length-1)
        : 0*T/m*s;
    auto const phase_blip_duration = phase_blip_area/this->G_max;
    this->phase_blip = {
        phase_blip_duration, {0*T/m*s, phase_blip_area, 0*T/m*s}};
    
    // The gradients are balanced ("canonical" design, with EPI readout), and
    // each readout is composed of (half_readout, half_readout, phase_blip). The
    // readouts are assumed to start with a positive amplitude.
    // 
    // For even and odd train lengths, we then have:
    // 
    // | Even train length       | Odd train length            |
    // | ----------------------- | --------------------------- |
    // |                     Start of TR                       |
    // |                       Pulse                           |
    // |                       Idle                            |
    // |                     Diffusion                         |
    // |                       Idle                            |
    // |                 Readout preparation                   |
    // | train_length/2 readouts | (train_length-1)/2 readouts |
    // |                         | half_readout                |
    // |                        TE                             |
    // |                         | half_readout                |
    // |                         | phase_blip                  |
    // | train_length/2 readouts | (train_length-1)/2 readouts |
    // |                  Readout rewind                       |
    // |                  End-of-TR idle                       |
    // |                      End of TR                        |
    
    auto const before_TE_readout = 
        (train_length-(train_length%2))/2*(readout_duration+phase_blip_duration)
        + (train_length%2==0 ? 0*s : 0.5*readout_duration);
    auto const after_TE_readout = 
        before_TE_readout
        + ((train_length%2 == 0) ? 0*s : phase_blip_duration);
    
    this->idle = {
        0.5*(
            TE - before_TE_readout - this->readout_preparation.get_duration()
            - tau_diffusion)};
    if(this->idle.get_duration() < 0*s)
    {
        throw std::runtime_error(
            "TE short low to accomodate diffusion and readout");
    }
    
    this->diffusion = {
        this->tau_diffusion, 
        {
            this->direction[0]*this->G_diffusion, 
            this->direction[1]*this->G_diffusion, 
            this->direction[2]*this->G_diffusion}};
    
    this->end_of_TR = {
        this->TR - this->TE - after_TE_readout
        - this->readout_rewind.get_duration()};
    if(this->end_of_TR.get_duration() < 0*s)
    {
        throw std::runtime_error("TR too low to accomodate readout");
    }
}
