#ifndef _96a3fddd_ce2d_434d_b798_2ee72aba3278
#define _96a3fddd_ce2d_434d_b798_2ee72aba3278

#include <Eigen/Core>
#include "Acquisition.h"

/// @file

/**
 * @brief Analytical model of the DW-SSFP signal relative to a 
 * non-diffusion-weighted (or low-diffusion-weighted) signal.
 *
 * See @ref freed(double, double, Eigen::Matrix3d const &, Acquisition const &)
 * "other definition" for details.
 *
 * @param T1: species @f$T_1@f$, in s.
 * @param T2: species @f$T_2@f$, in s.
 * @param D: species diffusion tensor, in m²/s.
 * @param dw: diffusion-weighted acquisition parameters (@f$\alpha@f$, TR, 
 *   @f$G_{\text{diffusion}}@f$, @f$\tau_{\text{diffusion}}@f$, direction)
 * @param non_dw: non-diffusion-weighted acquisition parameters (@f$\alpha@f$,
 *   TR, @f$G_{\text{diffusion}}@f$, @f$\tau_{\text{diffusion}}@f$, direction)
 */
double freed(
    double T1, double T2, Eigen::Matrix3d const & D, 
    Acquisition const & dw, Acquisition const & non_dw);

/**
 * @brief Analytical model of the DW-SSFP signal.
 * 
 * Reference: *Steady-state free precession experiments and exact treatment of
 * diffusion in a uniform gradient*. Freed et al. The Journal of Chemical 
 * Physics 115(9), pp. 4249-4258. 2001.
 * [doi:10.1063/1.1389859](https://doi.org/10.1063/1.1389859)
 *
 * @param T1: species @f$T_1@f$, in s.
 * @param T2: species @f$T_2@f$, in s.
 * @param D: species diffusion tensor, in m²/s.
 * @param acquisition: acquisition parameters (@f$\alpha@f$, TR, 
 *   @f$G_{\text{diffusion}}@f$, @f$\tau_{\text{diffusion}}@f$, direction)
 */
double freed(
    double T1, double T2, Eigen::Matrix3d const & D, 
    Acquisition const & acquisition);

#endif // _96a3fddd_ce2d_434d_b798_2ee72aba3278
