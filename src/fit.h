#ifndef _24f03df8_13d8_489a_b56b_fc8171c71309
#define _24f03df8_13d8_489a_b56b_fc8171c71309

#include <span>
#include <vector>

#include <boost/mpi/communicator.hpp>

#include "Acquisition.h"

/// @file

/**
 * @brief Dispatch fitting jobs to an MPI communicator.
 *
 * @warning result structures must be allocated by the caller.
 */
void fit(
    std::vector<Acquisition> const & scheme, unsigned int reference,
    std::span<double const> DW_SSFP, std::span<double const> B1_map,
    std::span<double const> T1_map, std::span<double const> T2_map,
    unsigned int population, unsigned int generations,
    std::span<double> champions_D, std::span<double> champions_T1,
        std::span<double> champions_T2,
    std::span<double> individuals_D, std::span<double> individuals_T1,
        std::span<double> individuals_T2,
    boost::mpi::communicator communicator);

/**
 * @brief Fit a region in a single process, store the result in user-provided
 * array.
 *
 * @warning result structures must be allocated by the caller.
 */
void fit(
    std::vector<Acquisition> const & scheme, unsigned int reference,
    std::span<double const> DW_SSFP, std::span<double const> B1_map,
    std::span<double const> T1_map, std::span<double const> T2_map,
    unsigned int population, unsigned int generations,
    std::span<double> champions_D, std::span<double> champions_T1,
        std::span<double> champions_T2,
    std::span<double> individuals_D, std::span<double> individuals_T1,
        std::span<double> individuals_T2);

#endif // _24f03df8_13d8_489a_b56b_fc8171c71309
