#ifndef _b62f1e90_b315_4d41_940c_152159958f9e
#define _b62f1e90_b315_4d41_940c_152159958f9e

#include <functional>
#include <utility>
#include <vector>
#include <pagmo/types.hpp>
#include <sycomore/Species.h>

#include "Acquisition.h"

/// @brief Description of the optimization problem in Pagmo2.
class Problem
{
public:
    /**
     * @brief Callable simulating the magnitude of the DW-SSFP signal from
     * @f$T_1@f$, @f$T_2@f$, @f$D@f$, and acquisition parameters.
     */
    using Simulator = double(*)(
        sycomore::Species const &, Acquisition const &, double);
    
    /// @brief Parameters of the set of acquisitions to fit.
    std::vector<Acquisition> scheme;
    
    /// @brief Index of the non-diffusion-weighted acquisition.
    std::size_t non_dw_index;
    
    /// @brief Magnitude of the signal for each acquisition.
    std::vector<double> signals;
    
    /// @brief Species @f$T_1@f$, in s.
    double T1;
    
    /// @brief Species @f$T_2@f$, in s.
    double T2;
    
    /// @brief Relative B1, unitless.
    double B1;
    
    /// @brief Simulation function (see @ref Simulator for parameters).
    Simulator simulator;
    
    /// @brief Rescale a dv from [-1, 1] to its true range.
    static pagmo::vector_double get_true_dv(pagmo::vector_double const & scaled_dv);
    
    /// @brief Extract the diffusion tensor from the decision vector.
    static Eigen::Matrix3d get_diffusion_tensor(pagmo::vector_double const & dv);
    
    /**
     * @brief Evaluate the fitness of a decision vector with respect to the
     * acquired and simulated signals.
     */
    pagmo::vector_double fitness(pagmo::vector_double const & dv) const;
    
    /// @brief Return the bounds of each problem variable.
    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const;

private:
    /// @brief Problem variables.
    enum Variables: std::size_t
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
    
    static std::pair<pagmo::vector_double, pagmo::vector_double> const _true_bounds;
};

#endif // _b62f1e90_b315_4d41_940c_152159958f9e
