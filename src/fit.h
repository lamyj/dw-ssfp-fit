#ifndef _24f03df8_13d8_489a_b56b_fc8171c71309
#define _24f03df8_13d8_489a_b56b_fc8171c71309

#include <pagmo/algorithm.hpp>
#include "Problem.h"

/// @file

/**
 * @brief Fit a single voxel, store the result in user-provided array.
 *
 * @warning D_array must be large enough to store the whole population.
 */
void fit_single(
    Problem const & problem, pagmo::algorithm const & algorithm,
    unsigned int population, unsigned int generations, int verbosity,
    double * D_array);

#endif // _24f03df8_13d8_489a_b56b_fc8171c71309
