#include "fit.h"

#include <algorithm>
#include <Eigen/SVD>
#include <pagmo/algorithm.hpp>
#include <pagmo/island.hpp>

#include "Problem.h"

void fit_single(
    Problem const & problem, pagmo::algorithm const & algorithm,
    unsigned int population, unsigned int generations, int verbosity,
    double * D_array)
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
