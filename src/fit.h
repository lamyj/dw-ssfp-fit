#ifndef _24f03df8_13d8_489a_b56b_fc8171c71309
#define _24f03df8_13d8_489a_b56b_fc8171c71309

#include <vector>
#include <boost/mpi/communicator.hpp>
#include <pagmo/algorithm.hpp>
#include "Problem.h"

/// @file

/**
 * @brief Dispatch fitting jobs to an MPI communicator.
 *
 * @warning result must be large enough to store the populations of all jobs.
 */
void fit(
    std::vector<Acquisition> const & scheme, unsigned int non_dw, 
    double const * DW_SSFP, double const * T1_map, double const * T2_map,
    double const * B1_map, boost::mpi::communicator communicator,
    unsigned int population, unsigned int generations, std::size_t blocks_count,
    int block_size, double * individuals, double * champions);

/**
 * @brief Fit a single voxel, store the result in user-provided array.
 *
 * @warning result array must be large enough to store the whole population.
 */
void fit(
    Problem const & problem, pagmo::algorithm const & algorithm,
    unsigned int population, unsigned int generations,
    double * individuals, double * champion);

#endif // _24f03df8_13d8_489a_b56b_fc8171c71309
