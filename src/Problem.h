#ifndef _b62f1e90_b315_4d41_940c_152159958f9e
#define _b62f1e90_b315_4d41_940c_152159958f9e

#include <functional>
#include <utility>
#include <pagmo/types.hpp>

#include "Acquisition.h"

/// @brief Description of the optimization problem in Pagmo2.
class Problem
{
public:
    /**
     * @brief Callable simulating the magnitude of the DW-SSFP signal from
     * @f$T_1@f$, @f$T_2@f$, @f$D@f$, a diffusion-weighted acquisition, and a
     * non-diffusion-weighted acquisition.
     */
    using Simulator = double(*)(
            double, double, Eigen::Matrix3d const &, 
            Acquisition const &, Acquisition const &);
    
    /// @brief Parameters of the set of acquisitions to fit.
    std::vector<Acquisition> acquisitions;
    
    /// @brief Index of the non-diffusion-weighted acquisition.
    std::size_t reference_acquisition_index;
    
    /// @brief Magnitude of the signal for each acquisition.
    std::vector<double> signals;
    
    /// @brief @f$T_1@f$, in seconds.
    double T1;
    
    /// @brief @f$T_2@f$, in seconds.
    double T2;
    
    /// @brief Simulation function (see @ref Simulator for parameters).
    Simulator simulator;
    
    /**
     * @brief Evaluate the fitness of a decision vector with respect to the
     * acquired and simulated signals.
     */
    pagmo::vector_double fitness(pagmo::vector_double const & dv) const;
    
    /// @brief Return the bounds of each problem variable.
    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const;

private:
    /// @brief Problem variables.
    enum class Variables
    {
        /// @brief First parameter of radius vector, @f$u \in [0,1]@f$
        u=0, 
        /// @brief Second parameter of radius vector, @f$v \in [0,1]@f$
        v, 
        /// @brief Axis of rotation around radius vector, @f$\psi \in [-\pi,\pi]@f$
        psi, 
        /// @brief Largest eigenvalue, in @f$\mu m^2/s@f$
        lambda1, 
        /// @brief Second eigenvalue, in @f$\mu m^2/s@f$
        lambda2, 
        /// @brief Largest eigenvalue, in @f$\mu m^2/s@f$
        lambda3,
        
        /// @brief Number of variables
        size
    };

};

#endif // _b62f1e90_b315_4d41_940c_152159958f9e
