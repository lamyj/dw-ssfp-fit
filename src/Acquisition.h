#ifndef _656a810d_dca3_413b_9208_d1924c3aea92
#define _656a810d_dca3_413b_9208_d1924c3aea92

#include <Eigen/Core>
#include <sycomore/Quantity.h>
#include <sycomore/TimeInterval.h>

/// @brief Parameters of a DW-SSFP acquisition
struct Acquisition
{
    /// @brief Flip angle.
    sycomore::Quantity alpha;
    
    /// @brief Amplitude of the diffusion-sensitization gradient.
    sycomore::Quantity G_diffusion;
    
    /// @brief Duration of the diffusion-sensitization gradient.
    sycomore::Quantity tau_diffusion;
    
    /// @brief Direction of the diffusion-sensitization gradient, normalized.
    Eigen::Vector3d direction;
    
    /// @brief Echo time.
    sycomore::Quantity TE;
    
    /// @brief Repetition time.
    sycomore::Quantity TR;
    
    /// @brief Per-pixel bandwidth.
    sycomore::Quantity pixel_bandwidth;
    
    /// @brief In-plane resolution assumed isotropic.
    sycomore::Quantity resolution;
    
    /// @brief Maximum amplitude of the gradient system.
    sycomore::Quantity G_max;
    
    sycomore::TimeInterval diffusion;
    sycomore::TimeInterval idle;
    sycomore::TimeInterval ro_plus;
    sycomore::TimeInterval ro_minus;
    sycomore::TimeInterval end_of_TR;
    
    Acquisition(
        sycomore::Quantity alpha, 
        sycomore::Quantity G_diffusion, sycomore::Quantity tau_diffusion, 
            Eigen::Vector3d direction,
        sycomore::Quantity TE, sycomore::Quantity TR,
        sycomore::Quantity pixel_bandwidth, sycomore::Quantity resolution,
        sycomore::Quantity G_max);
    
    Acquisition(
        double alpha, 
        double G_diffusion, double tau_diffusion, Eigen::Vector3d direction,
        double TE, double TR,
        double pixel_bandwidth, double resolution, double G_max);
    
    void update_intervals();
};

#endif // _656a810d_dca3_413b_9208_d1924c3aea92
