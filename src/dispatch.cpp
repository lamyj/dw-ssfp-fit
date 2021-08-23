#include "dispatch.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

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

std::span<double> scatter_blocks(
    boost::mpi::communicator const & communicator,
    std::span<double> const & source, unsigned int block_size)
{
    if(source.empty())
    {
        return std::span<double>();
    }
    
    std::vector<int> subset_sizes(communicator.size());
    std::vector<int> offsets(communicator.size());
    if(communicator.rank() == 0)
    {
        if(source.size() % block_size != 0)
        {
            throw std::runtime_error(
                "source size must be divisible by block size");
        }
        std::tie(subset_sizes, offsets) = 
            compute_chunks(communicator, source.size()/block_size, block_size);
    }
    boost::mpi::broadcast(
        communicator, subset_sizes.data(), subset_sizes.size(), 0);
    boost::mpi::broadcast(communicator, offsets.data(), offsets.size(), 0);
    
    auto const size = subset_sizes[communicator.rank()];
    std::span<double> span(new double[size], size);
    // NOTE: if source is empty (the case on ranks ≠ 0), then sizes must also be
    // empty
    boost::mpi::scatterv(
        communicator, source.data(),
        source.empty()?decltype(subset_sizes)():subset_sizes, span.data(), 0);
    return span;
}

void gather_blocks(
    boost::mpi::communicator const & communicator,
    std::span<double> const & source, std::span<double> & destination,
    unsigned int block_size)
{
    std::vector<int> subset_sizes(communicator.size());
    std::vector<int> offsets(communicator.size());
    if(communicator.rank() == 0)
    {
        if(source.size() % block_size != 0)
        {
            throw std::runtime_error(
                "source size must be divisible by block size");
        }
        std::tie(subset_sizes, offsets) = 
            compute_chunks(communicator, source.size()/block_size, block_size);
    }
    boost::mpi::broadcast(
        communicator, subset_sizes.data(), subset_sizes.size(), 0);
    boost::mpi::broadcast(communicator, offsets.data(), offsets.size(), 0);
    
    boost::mpi::gatherv(
        communicator, source.data(), source.size(), destination.data(),
        subset_sizes, 0);
}
