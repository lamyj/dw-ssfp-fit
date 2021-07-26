#ifndef _1f4a493f_51c3_4425_a75e_e98a13fff20b
#define _1f4a493f_51c3_4425_a75e_e98a13fff20b

#include <sycomore/Species.h>
#include "Acquisition.h"
#include "Problem.h"

/// @file

/// @brief Return the average runtime of a simulator.
double benchmark(
    sycomore::Species const & species, Acquisition const & acquisition,
    Problem::Simulator simulator, std::size_t count);

#endif // _1f4a493f_51c3_4425_a75e_e98a13fff20b
