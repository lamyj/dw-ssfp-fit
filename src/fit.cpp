#include "fit.h"

#include <algorithm>
#include <iostream>
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
    int block_size, double * individuals, double * champions)
{
    // Dispatch the chunks to the workers. 
    auto const DW_SSFP_subset = scatter_blocks(
        communicator, DW_SSFP, blocks_count, block_size);
    auto const T1_subset = scatter_blocks(
        communicator, T1_map, blocks_count, 1);
    auto const T2_subset = scatter_blocks(
        communicator, T2_map, blocks_count, 1);
    auto const B1_subset = scatter_blocks(
        communicator, B1_map, blocks_count, 1);
    
    auto const subset_blocks_count = DW_SSFP_subset.size()/block_size;
    auto const item_size = 9;
    
    auto const return_individuals = (individuals!=nullptr);
    auto const return_champions = (champions!=nullptr);

    auto const base_size = item_size * subset_blocks_count;
    std::vector<double> local_individuals(
        return_individuals?base_size*population:0, 0.);
    std::vector<double> local_champions(return_champions?base_size:0, 0.);
    for(std::size_t i=0; i<subset_blocks_count; ++i)
    {
        try
        {
            std::vector<double> signals(
                DW_SSFP_subset.data()+block_size*i, 
                DW_SSFP_subset.data()+block_size*(i+1));
            Problem problem{
                scheme, non_dw, signals, T1_subset[i], T2_subset[i], B1_subset[i],
                epg_discrete_1d};
            pagmo::algorithm algorithm{pagmo::de1220{generations}};
            std::cout 
                << communicator.rank() << ": "
                << "item " << i << "/" << subset_blocks_count
                << " starting fit" << std::endl;
            fit(
                problem, algorithm, population, generations,
                return_individuals?local_individuals.data()+item_size*population*i:nullptr,
                return_champions?local_champions.data()+item_size*i:nullptr,
                communicator);
            std::cout 
                << communicator.rank() << ": "
                << "item " << i << "/" << subset_blocks_count
                << " fit OK" << std::endl;
        }
        catch(std::exception & e)
        {
            std::cerr 
                << communicator.rank() << ": "
                << "Could not fit problem: " << e.what() << std::endl;
        }
        catch(...)
        {
            std::cerr 
                << communicator.rank() << ": "
                << "Could not fit problem (unknown exception)" << std::endl;
        }
    }
    
    std::cout 
        << communicator.rank() << ": "
        << "Done fitting subset" << std::endl;
    
    // if(individuals != nullptr)
    // {
    //     gather_blocks(
    //         communicator, local_individuals, blocks_count, item_size*population, 
    //         individuals);
    // }
    // if(champions != nullptr)
    // {
    //     gather_blocks(
    //         communicator, local_champions, blocks_count, item_size, champions);
    // }
    communicator.barrier();
}

void fit(
    Problem const & problem, pagmo::algorithm const & algorithm,
    unsigned int population, unsigned int generations,
    double * individuals, double * champion,
    boost::mpi::communicator communicator)
{
    pagmo::island island{algorithm, problem, population};
    
    bool has_error = false;
    try
    {
        island.evolve();
        island.wait_check();
        std::cout 
            << communicator.rank() << ": "
            << "evolution OK" << std::endl;
    }
    catch(std::exception const & e)
    {
        std::cerr << "Error during evolution: " << e.what() << std::endl;
        has_error = true;
    }
    catch(...)
    {
        std::cerr << "Error during evolution (unknown)" << std::endl;
        has_error = true;
    }
    
    if(!has_error)
    {
        auto const final_population = island.get_population();
        
        if(individuals != nullptr)
        {
            for(std::size_t index=0, end=final_population.size(); index!=end; ++index)
            {
                auto const & scaled_dv = final_population.get_x()[index];
                auto const true_dv = Problem::get_true_dv(scaled_dv);
                auto const D = Problem::get_diffusion_tensor(true_dv);
                // std::copy(D.data(), D.data()+D.size(), individuals+9*index);
            }
        }
        
        if(champion != nullptr)
        {
            auto const champion_dv = final_population.champion_x();
            auto const true_dv = Problem::get_true_dv(champion_dv);
            auto const D = Problem::get_diffusion_tensor(true_dv);
            // std::copy(D.data(), D.data()+D.size(), champion);
        }
    }
}
