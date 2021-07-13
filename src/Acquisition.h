#ifndef _656a810d_dca3_413b_9208_d1924c3aea92
#define _656a810d_dca3_413b_9208_d1924c3aea92

#include <Eigen/Core>

/// @brief Parameters of a DW-SSFP acquisition
struct Acquisition
{
    /// @brief Flip angle in rad.
    double alpha;
    
    /// @brief Amplitude of the diffusion-sensitization gradient, in T/m.
    double G_diffusion;
    
    /// @brief Duration of the diffusion-sensitization gradient, in s.
    double tau_diffusion;
    
    /// @brief Direction of the diffusion-sensitization gradient, normalized.
    Eigen::Vector3d direction;
    
    /// @brief Echo time, in seconds.
    double TE;
    
    /// @brief Repetition time, in s.
    double TR;
    
    /// @brief Per-pixel bandwidth, in Hz.
    double pixel_bandwidth;
    
    /// @brief In-plane resolution, in m, assumed isotropic.
    double resolution;
    
    /// @brief Maximum amplitude of the gradient system, in T/m.
    double G_max;
};

#endif // _656a810d_dca3_413b_9208_d1924c3aea92
