#ifndef _b62f1e90_b315_4d41_940c_152159958f9e
#define _b62f1e90_b315_4d41_940c_152159958f9e

#include <optional>
#include <utility>
#include <vector>

#include <pagmo/types.hpp>
#include <sycomore/Array.h>
#include <sycomore/Quantity.h>
#include <sycomore/Species.h>

#include "Acquisition.h"

/// @brief Description of the optimization problem in Pagmo2.
class Problem
{
public:
    using Vector = pagmo::vector_double;
    
    /**
     * @brief Callable simulating the magnitude of the DW-SSFP signal from
     * @f$T_1@f$, @f$T_2@f$, @f$D@f$, and acquisition parameters.
     */
    using Simulator = double(*)(
        sycomore::Species const &, Acquisition const &, double);
    
    Problem() = default;
    Problem(
        std::vector<Acquisition> const & scheme, unsigned int reference_index,
        std::vector<double> const & signals,
        std::optional<double> T1, std::optional<double> T2, double B1,
        Simulator simulator);
    Problem(Problem const &) = default;
    Problem(Problem &&) = default;
    Problem & operator=(Problem const &) = default;
    ~Problem() = default;
    
    /// @brief Rescale a dv from [-1, 1] to its true range.
    Vector get_true_dv(Vector const & scaled_dv) const;
    
    /// @brief Extract the diffusion tensor from the decision vector.
    sycomore::Array<sycomore::Quantity> 
    get_diffusion_tensor(Vector const & dv) const;
    
    /// @brief Extract the T1 value from the decision vector or the Problem.
    sycomore::Quantity get_T1(Vector const & dv) const;
    
    /// @brief Extract the T2 value from the decision vector or the Problem.
    sycomore::Quantity get_T2(Vector const & dv) const;
    
    /**
     * @brief Evaluate the fitness of a decision vector with respect to the
     * acquired and simulated signals.
     */
    Vector fitness(Vector const & dv) const;
    
    /// @brief Return the bounds of each problem variable.
    std::pair<Vector, Vector> get_bounds() const;

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
    
    /// @brief Parameters of the set of acquisitions to fit.
    std::vector<Acquisition> _scheme;
    
    /// @brief Index of the reference acquisition.
    unsigned int _reference_index;
    
    /// @brief Magnitude of the signal for each acquisition.
    std::vector<double> _signals;
    
    /// @brief Species @f$T_1@f$, in s.
    std::optional<double> _T1;
    
    /// @brief Species @f$T_2@f$, in s.
    std::optional<double> _T2;
    
    /// @brief Relative B1, unitless.
    double _B1;
    
    /// @brief Simulation function (see @ref Simulator for parameters).
    Simulator _simulator;
    
    std::pair<Vector, Vector> _true_bounds;
};

#endif // _b62f1e90_b315_4d41_940c_152159958f9e
