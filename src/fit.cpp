#include "fit.h"

#include <algorithm>
#include <vector>
#include <boost/mpi/communicator.hpp>
#include <Eigen/SVD>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de1220.hpp>
#include <pagmo/island.hpp>

#include "dispatch.h"
#include "models.h"
#include "Problem.h"

void fit(
    std::vector<Acquisition> const & scheme, unsigned int non_dw, 
    double const * DW_SSFP, double const * T1_map, double const * T2_map,
    double const * B1_map, boost::mpi::communicator communicator,
    unsigned int population, unsigned int generations, std::size_t blocks_count,
    int block_size, double * result)
{
    // Dispatch the chunks to the workers. 
    auto DW_SSFP_subset = scatter_blocks(
        communicator, DW_SSFP, blocks_count, block_size);
    auto T1_subset = scatter_blocks(
        communicator, T1_map, blocks_count, block_size);
    auto T2_subset = scatter_blocks(
        communicator, T2_map, blocks_count, block_size);
    auto B1_subset = scatter_blocks(
        communicator, B1_map, blocks_count, block_size);
    
    auto const subset_blocks_count = DW_SSFP_subset.size()/block_size;
    
    // NOTE: data/model do not allow to easily specify ftol/xtol
    pagmo::algorithm algorithm{pagmo::de1220{generations}};
    
    std::vector<double> D(9*population*subset_blocks_count);
    for(std::size_t i=0; i<subset_blocks_count; ++i)
    {
        std::vector<double> signals(
            DW_SSFP_subset.data()+block_size*i, 
            DW_SSFP_subset.data()+block_size*(i+1));
        Problem problem{
            scheme, non_dw, signals, T1_subset[i], T2_subset[i], B1_subset[i],
            freed};
        fit(problem, algorithm, population, generations, D.data()+9*population*i);
    }
    
    gather_blocks(communicator, D, blocks_count, 9*population, result);
}

void fit(
    Problem const & problem, pagmo::algorithm const & algorithm,
    unsigned int population, unsigned int generations, double * D_array)
{
    pagmo::island island{algorithm, problem, population};
    island.evolve();
    island.wait_check();
    
    auto const final_population = island.get_population();
    for(std::size_t index=0, end=final_population.size(); index!=end; ++index)
    {
        auto const & scaled_dv = final_population.get_x()[index];
        auto const true_dv = Problem::get_true_dv(scaled_dv);
        auto const D = Problem::get_diffusion_tensor(true_dv);
        std::copy(D.data(), D.data()+D.size(), D_array+9*index);
    }
}
