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
    int block_size, bool return_tensor, double * individuals, double * champions)
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
    auto const item_size = return_tensor?9:6;
    
    // NOTE: data/model do not allow to easily specify ftol/xtol
    pagmo::algorithm algorithm{pagmo::de1220{generations}};
    
    std::vector<double> local_individuals(item_size*population*subset_blocks_count);
    std::vector<double> local_champions(item_size*subset_blocks_count);
    for(std::size_t i=0; i<subset_blocks_count; ++i)
    {
        std::vector<double> signals(
            DW_SSFP_subset.data()+block_size*i, 
            DW_SSFP_subset.data()+block_size*(i+1));
        Problem problem{
            scheme, non_dw, signals, T1_subset[i], T2_subset[i], B1_subset[i],
            freed};
        fit(
            problem, algorithm, population, generations, return_tensor,
            local_individuals.data()+item_size*population*i,
            local_champions.data()+item_size*i);
    }
    
    gather_blocks(
        communicator, local_individuals, blocks_count, item_size*population, 
        individuals);
    gather_blocks(
        communicator, local_champions, blocks_count, item_size, champions);
}

void fit(
    Problem const & problem, pagmo::algorithm const & algorithm,
    unsigned int population, unsigned int generations, bool return_tensor,
    double * individuals, double * champion)
{
    pagmo::island island{algorithm, problem, population};
    island.evolve();
    island.wait_check();
    
    auto const final_population = island.get_population();
    for(std::size_t index=0, end=final_population.size(); index!=end; ++index)
    {
        auto const & scaled_dv = final_population.get_x()[index];
        auto const true_dv = Problem::get_true_dv(scaled_dv);
        if(return_tensor)
        {
            auto const D = Problem::get_diffusion_tensor(true_dv);
            std::copy(D.data(), D.data()+D.size(), individuals+9*index);
        }
        else
        {
            std::copy(
                scaled_dv.data(), scaled_dv.data()+scaled_dv.size(),
                individuals+6*index);
        }
    }
    
    auto const & champion_dv = final_population.champion_x();
    if(return_tensor)
    {
        auto const true_dv = Problem::get_true_dv(champion_dv);
        auto const D = Problem::get_diffusion_tensor(true_dv);
        std::copy(D.data(), D.data()+D.size(), champion);
    }
    else
    {
        std::copy(
            champion_dv.data(), champion_dv.data()+champion_dv.size(),
            champion);
    }
}
