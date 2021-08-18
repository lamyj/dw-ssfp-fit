#ifndef _656a810d_dca3_413b_9208_d1924c3aea92
#define _656a810d_dca3_413b_9208_d1924c3aea92

#include <map>
#include <Eigen/Core>
#include <sycomore/Array.h>
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
    
    /// @brief Shape of the k-space matrix as @f$(k_x, k_y)@f$.
    Eigen::Vector2i shape;
    
    /// @brief Field-of-view along @f$(k_x, k_y)@f$.
    sycomore::Array<sycomore::Quantity> FOV;
    
    /// @brief Number of echoes
    unsigned int train_length;
    
    /// @brief Maximum amplitude of the gradient system.
    sycomore::Quantity G_max;
    
    sycomore::TimeInterval diffusion;
    sycomore::TimeInterval idle;
    sycomore::TimeInterval readout_preparation;
    std::map<int, sycomore::TimeInterval> half_readout;
    sycomore::TimeInterval phase_blip;
    sycomore::TimeInterval readout_rewind;
    sycomore::TimeInterval end_of_TR;
    
    Acquisition(
        sycomore::Quantity alpha, 
        sycomore::Quantity G_diffusion, sycomore::Quantity tau_diffusion, 
            Eigen::Vector3d direction,
        sycomore::Quantity TE, sycomore::Quantity TR,
        sycomore::Quantity pixel_bandwidth, 
        Eigen::Vector2i shape, sycomore::Array<sycomore::Quantity> FOV,
        unsigned int train_length, sycomore::Quantity G_max);
    
    Acquisition(
        double alpha, 
        double G_diffusion, double tau_diffusion, Eigen::Vector3d direction,
        double TE, double TR,
        double pixel_bandwidth, Eigen::Vector2i shape, Eigen::Vector2d FOV,
        unsigned int train_length, double G_max);
    
    Acquisition(Acquisition const &) = default;
    Acquisition(Acquisition &&) = default;
    Acquisition & operator=(Acquisition const &) = default;
    ~Acquisition() = default;
    
    void update_intervals();
};

#endif // _656a810d_dca3_413b_9208_d1924c3aea92
