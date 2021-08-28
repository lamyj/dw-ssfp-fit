#ifndef _b1677a88_d26b_4084_a3f9_177db08f70e9
#define _b1677a88_d26b_4084_a3f9_177db08f70e9

#include <map>

#include <sycomore/Quantity.h>
#include <sycomore/TimeInterval.h>

#include "Acquisition.h"

/// @brief Time intervals associated with the discrete EPG models.
struct TimeIntervals
{
    sycomore::TimeInterval diffusion;
    sycomore::TimeInterval idle;
    sycomore::TimeInterval readout_preparation;
    std::map<int, sycomore::TimeInterval> half_readout;
    sycomore::TimeInterval phase_blip;
    sycomore::TimeInterval readout_rewind;
    sycomore::TimeInterval end_of_TR;
    
    TimeIntervals(
        Acquisition const & acquisition, sycomore::Quantity const & bin_width);
};

#endif // _b1677a88_d26b_4084_a3f9_177db08f70e9
