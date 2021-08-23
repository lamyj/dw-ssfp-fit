#ifndef _81671a30_0dfd_4395_8d3c_95155d5c0d02
#define _81671a30_0dfd_4395_8d3c_95155d5c0d02

#include <span>
#include <utility>
#include <vector>

#include <boost/mpi/communicator.hpp>

/// @file

/**
 * @brief Scatter the blocks fairly on all ranks of the communicator.
 *
 * @warning the return value must be de-allocated by the user, i.e. 
 * `delete[] span.data()`.
 *
 * @warning source.size() must be divisible by block_size.
 */
std::span<double> scatter_blocks(
    boost::mpi::communicator const & communicator, 
    std::span<double const> const & source, unsigned int block_size=1);

/**
 * @brief Gather the blocks matching the fair distribution of scatter_blocks.
 *
 * @warning source.size() must be divisible by block_size.
 */
void gather_blocks(
    boost::mpi::communicator const & communicator,
    std::span<double const> const & source, std::span<double> & destination,
    unsigned int block_size=1);

/**
 * @brief Compute the chunk size so that blocks are fairly dispatched on all
 * ranks of the communicator.
 */
std::pair<std::vector<int>, std::vector<int>>
compute_chunks(
    boost::mpi::communicator const & communicator, int blocks_count, 
    int block_size=1);

#endif // _81671a30_0dfd_4395_8d3c_95155d5c0d02
