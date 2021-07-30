#include "dispatch.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#if BOOST_VERSION <= 106900
// Compile error due to scoping error.
using namespace boost::mpi::detail;
#endif

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>

std::pair<std::vector<int>, std::vector<int>>
compute_chunks(
    boost::mpi::communicator const & communicator, 
    int blocks_count, int block_size)
{
    std::vector<int> subset_sizes(communicator.size());
    
    // R ranks, N elements
    // N - (N%R) can be split evenly in chunks of size (N - (N%R))/R. 
    // Add 1 to size for the first N%R chunks.
    auto const div = std::div(blocks_count, communicator.size());
    std::fill(subset_sizes.begin(), subset_sizes.end(), div.quot);
    for(std::size_t i=0; i != div.rem; ++i)
    {
        ++subset_sizes[i];
    }
    
    // std::cout << "Subset sizes:";
    // for(auto && x: subset_sizes) { std::cout << " " << x; }
    // std::cout << std::endl;
    
    std::vector<int> offsets(subset_sizes.size());
    offsets[0] = 0;
    std::partial_sum(
        subset_sizes.begin(), subset_sizes.end()-1, 1+offsets.begin());
    
    for(auto && x: subset_sizes)
    {
        x *= block_size;
    }
    for(auto && x: offsets)
    {
        x *= block_size;
    }
    
    return {subset_sizes, offsets};
}

std::vector<double>
scatter_blocks(
    boost::mpi::communicator const & communicator,
    double const * data, std::size_t blocks_count, int block_size)
{
    std::vector<int> subset_sizes(communicator.size());
    std::vector<int> offsets(communicator.size());
    if(communicator.rank() == 0)
    {
        std::tie(subset_sizes, offsets) = 
            compute_chunks(communicator, blocks_count, block_size);
    }
    boost::mpi::broadcast(
        communicator, subset_sizes.data(), subset_sizes.size(), 0);
    boost::mpi::broadcast(communicator, offsets.data(), offsets.size(), 0);
    
    std::vector<double> subset(subset_sizes[communicator.rank()]);
    // NOTE: if data is empty (the case on ranks ≠ 0), then sizes must also be
    // empty
    boost::mpi::scatterv(
        communicator, data,
        (data==nullptr)?std::vector<int>():subset_sizes, offsets, 
        subset.data(), subset.size(), 0);
    
    return subset;
}

void gather_blocks(
    boost::mpi::communicator const & communicator,
    std::vector<double> const & subset, int blocks_count, int block_size, 
    double * result)
{
    std::vector<int> subset_sizes(communicator.size());
    std::vector<int> offsets(communicator.size());
    if(communicator.rank() == 0)
    {
        std::tie(subset_sizes, offsets) = 
            compute_chunks(communicator, blocks_count, block_size);
    }
    boost::mpi::broadcast(
        communicator, subset_sizes.data(), subset_sizes.size(), 0);
    boost::mpi::broadcast(communicator, offsets.data(), offsets.size(), 0);
    
    boost::mpi::gatherv(communicator, subset, result, subset_sizes, offsets, 0);
}
