#include "TimeIntervals.h"

#include <cmath>
#include <map>

#include <sycomore/Quantity.h>
#include <sycomore/sycomore.h>
#include <sycomore/TimeInterval.h>
#include <sycomore/units.h>

#include "Acquisition.h"

TimeIntervals
::TimeIntervals(
    Acquisition const & acquisition, sycomore::Quantity const & bin_width)
{
    using namespace sycomore::units;
    
    auto const readout_duration = 1/acquisition.pixel_bandwidth;
    
    // Frequency-encoding gradient for the whole readout
    // Handbook, eq. 8.13, p. 252, with Δν = nₓ Δνₚₚ / 2 (eq. 8.10), 
    // T_acq = 1/pixel_bandwidth
    auto const frequency_encoding_amplitude = 
        (2*M_PI*acquisition.shape[0]) 
        / (sycomore::gamma*readout_duration*acquisition.FOV[0]);
    if(frequency_encoding_amplitude > acquisition.G_max)
    {
        throw std::runtime_error("G_max too low for frequency encoding");
    }
    auto const frequency_encoding_area =
        frequency_encoding_amplitude * readout_duration;
    
    // Maximum value of the phase-encoding gradient (border of the k-space).
    // Handbook, eq. 8.38, p. 262
    // auto max_phase_encoding_area = 
    //     M_PI * (acquisition.shape[1]-1) / (sycomore::gamma*acquisition.FOV[1]);
    // Since we are using discrete EPG, the max phase encoding *bin* must be
    // divisible by number of lines-1 to avoid binning problems with the phase
    // blip.
    auto const max_phase_encoding_moment =
        M_PI*(acquisition.shape[1]-1)/acquisition.FOV[1];
    int max_phase_encoding_bin = std::round(max_phase_encoding_moment/bin_width);
    if(acquisition.train_length > 1)
    {
        max_phase_encoding_bin = 
            (acquisition.train_length-1)
            * std::div(max_phase_encoding_bin, acquisition.train_length-1).quot;
    }
    auto const max_phase_encoding_area =
        max_phase_encoding_bin*bin_width/sycomore::gamma;
    
    // Readout preparation: go to the corner of the k-space in the shortest
    // time possible.
    auto const max_encoding_area = std::max(
        frequency_encoding_area/2, max_phase_encoding_area);
    this->readout_preparation = {
        max_encoding_area/acquisition.G_max,
        {
            -frequency_encoding_area/2, 
            acquisition.train_length>1 ? -max_phase_encoding_area : 0*T/m*s,
            0*T/m*s}};
    if(this->readout_preparation.get_gradient_amplitude()[1] > acquisition.G_max)
    {
        throw std::runtime_error("G_max too low for phase encoding");
    }
    
    auto const even_train_length = (acquisition.train_length%2 == 0);
    auto const half_lines =
        (acquisition.train_length-(acquisition.train_length%2))/2;
    
    // Rewind: balance the spatial encoding gradients
    this->readout_rewind = {
        max_encoding_area/acquisition.G_max,
        {
            (even_train_length ? +1 : -1)*frequency_encoding_area/2, 
            acquisition.train_length>1 ? -max_phase_encoding_area : 0*T/m*s, 
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
        (acquisition.train_length > 1)
        ? 2*max_phase_encoding_area/(acquisition.train_length-1)
        : 0*T/m*s;
    auto const phase_blip_duration = phase_blip_area/acquisition.G_max;
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
        half_lines*(readout_duration+phase_blip_duration)
        + (even_train_length ? 0*s : 0.5*readout_duration);
    auto const after_TE_readout = 
        before_TE_readout
        + (even_train_length ? 0*s : phase_blip_duration);
    
    this->idle = {
        0.5*(
            acquisition.TE - before_TE_readout
            - this->readout_preparation.get_duration()
            - acquisition.tau_diffusion)};
    if(this->idle.get_duration() < 0*s)
    {
        throw std::runtime_error(
            "TE short low to accomodate diffusion and readout");
    }
    
    this->diffusion = {
        acquisition.tau_diffusion, 
        {
            acquisition.direction[0]*acquisition.G_diffusion, 
            acquisition.direction[1]*acquisition.G_diffusion, 
            acquisition.direction[2]*acquisition.G_diffusion}};
    
    this->end_of_TR = {
        acquisition.TR - acquisition.TE - after_TE_readout
        - this->readout_rewind.get_duration()};
    if(this->end_of_TR.get_duration() < 0*s)
    {
        throw std::runtime_error("TR too low to accomodate readout");
    }
}