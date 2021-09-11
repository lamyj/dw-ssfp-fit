#include "fit.h"

#include <algorithm>
#include <optional>
#include <span>
#include <vector>

#include <boost/mpi/communicator.hpp>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de1220.hpp>
#include <pagmo/island.hpp>

#include "Acquisition.h"
#include "dispatch.h"
#include "models.h"
#include "Problem.h"

void fit(
    std::vector<Acquisition> const & scheme, unsigned int reference,
    std::span<double const> DW_SSFP, std::span<double const> B1_map,
    std::span<double const> T1_map, std::span<double const> T2_map,
    unsigned int population, unsigned int generations,
    std::span<double> champions_D, std::span<double> champions_T1,
        std::span<double> champions_T2,
    std::span<double> individuals_D, std::span<double> individuals_T1,
        std::span<double> individuals_T2,
    boost::mpi::communicator communicator)
{
    // Dispatch the chunks to the workers. 
    auto const size = B1_map.size();
    auto DW_SSFP_subset = scatter_blocks(communicator, DW_SSFP, scheme.size());
    auto B1_map_subset = scatter_blocks(communicator, B1_map);
    auto T1_map_subset = scatter_blocks(communicator, T1_map);
    auto T2_map_subset = scatter_blocks(communicator, T2_map);
    
    auto champions_D_subset = scatter_blocks(communicator, champions_D, 9);
    auto champions_T1_subset = scatter_blocks(communicator, champions_T1);
    auto champions_T2_subset = scatter_blocks(communicator, champions_T2);
    
    auto individuals_D_subset = scatter_blocks(communicator, individuals_D, 9);
    auto individuals_T1_subset = scatter_blocks(communicator, individuals_T1);
    auto individuals_T2_subset = scatter_blocks(communicator, individuals_T2);
    
    fit(
        scheme, reference,
        DW_SSFP_subset, B1_map_subset, T1_map_subset, T2_map_subset,
        population, generations, 
        champions_D_subset, champions_T1_subset, champions_T2_subset,
        individuals_D_subset, individuals_T1_subset, individuals_T2_subset);
    
    bool const fit_T1 = T1_map_subset.empty();
    bool const fit_T2 = T2_map_subset.empty();
    bool const return_individuals = !individuals_D_subset.empty();
    
    // Gather champions (D is always present, T1 and T2 are optional).
    gather_blocks(communicator, champions_D_subset, champions_D, 9);
    if(fit_T1)
    {
        gather_blocks(communicator, champions_T1_subset, champions_T1);
    }
    if(fit_T2)
    {
        gather_blocks(communicator, champions_T2_subset, champions_T2);
    }
    
    // Gather individuals (optional; if present, same semantics as for champions)
    if(return_individuals)
    {
        gather_blocks(communicator, individuals_D_subset, individuals_D, 9);
        if(fit_T1)
        {
            gather_blocks(communicator, individuals_T1_subset, individuals_T1);
        }
        if(fit_T2)
        {
            gather_blocks(communicator, individuals_T2_subset, individuals_T2);
        }
    }
    
    for(auto && subset: {
        DW_SSFP_subset, T1_map_subset, T2_map_subset,
        champions_D_subset, champions_T1_subset, champions_T2_subset,
        individuals_D_subset, individuals_T1_subset, individuals_T2_subset})
    {
        if(!subset.empty())
        {
            delete[] subset.data();
        }
    }
}

void fit(
    std::vector<Acquisition> const & scheme, unsigned int reference,
    std::span<double const> DW_SSFP, std::span<double const> B1_map,
    std::span<double const> T1_map, std::span<double const> T2_map,
    unsigned int population, unsigned int generations,
    std::span<double> champions_D, std::span<double> champions_T1,
        std::span<double> champions_T2,
    std::span<double> individuals_D, std::span<double> individuals_T1,
        std::span<double> individuals_T2)
{
    bool const fit_T1 = T1_map.empty();
    bool const fit_T2 = T2_map.empty();
    bool const return_individuals = !individuals_D.empty();
    
    for(int index=0; index != B1_map.size(); ++index)
    {
        std::vector<double> signals(
            DW_SSFP.data()+scheme.size()*index, 
            DW_SSFP.data()+scheme.size()*(index+1));
        Problem problem{
            scheme, reference, signals, 
            fit_T1?std::optional<double>():T1_map[index],
            fit_T2?std::optional<double>():T2_map[index],
            B1_map[index], epg_discrete_1d};
        pagmo::algorithm algorithm{pagmo::de1220{generations}};
        
        pagmo::island island{algorithm, problem, population};
        island.evolve();
        island.wait_check();
        
        auto const final_population = island.get_population();
        
        auto const dv = problem.get_true_dv(final_population.champion_x());
        auto const D = problem.get_diffusion_tensor(dv);
        std::transform(
            D.begin(), D.end(), champions_D.data()+9*index,
            [](auto const & q) { return q.magnitude; } );
        if(fit_T1)
        {
            champions_T1[index] = problem.get_T1(dv).magnitude;
        }
        if(fit_T2)
        {
            champions_T2[index] = problem.get_T2(dv).magnitude;
        }
        
        if(return_individuals)
        {
            for(std::size_t i=0; i != final_population.size(); ++i)
            {
                auto const dv = problem.get_true_dv(final_population.get_x()[i]);
                
                auto const destination = population*index+i;
                
                auto const D = problem.get_diffusion_tensor(dv);
                std::transform(
                    D.begin(), D.end(), individuals_D.data()+9*destination,
                    [](auto const & q) { return q.magnitude; } );
                if(fit_T1)
                {
                    individuals_T1[destination] = problem.get_T1(dv).magnitude;
                }
                if(fit_T2)
                {
                    individuals_T2[destination] = problem.get_T2(dv).magnitude;
                }
            }
        }
    }
}
