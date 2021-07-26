#ifndef _96a3fddd_ce2d_434d_b798_2ee72aba3278
#define _96a3fddd_ce2d_434d_b798_2ee72aba3278

#include <Eigen/Core>
#include <sycomore/Species.h>
#include "Acquisition.h"

/// @file

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
 *   @f$G_{\text{diffusion}}@f$, @f$\tau_{\text{diffusion}}@f$, direction).
 * @param B1: relative B1, unitless.
 */
double freed(
    sycomore::Species const & species,
    Acquisition const & acquisition, double B1);

/**
 * @brief 1D, discrete EPG model of the DW-SSFP signal, based on the "canonical"
 * sequence diagram.
 * 
 * Reference: *Steady-state diffusion-weighted imaging: theory, acquisition and
 * analysis*. McNab & Miller. NMR in Biomedicine 23(7). 2010.
 * [doi:10.1002/nbm.1509](https://doi.org/10.1002/nbm.1509)
 * 
 * @param T1: species @f$T_1@f$, in s.
 * @param T2: species @f$T_2@f$, in s.
 * @param D: species diffusion tensor, in m²/s.
 * @param acquisition: acquisition parameters.
 * @param B1: relative B1, unitless.
 */
double epg_discrete_1d(
    sycomore::Species const & species,
    Acquisition const & acquisition, double B1);

/**
 * @brief 3D, discrete EPG model of the DW-SSFP signal, based on the "canonical"
 * sequence diagram.
 * 
 * Reference: *Steady-state diffusion-weighted imaging: theory, acquisition and
 * analysis*. McNab & Miller. NMR in Biomedicine 23(7). 2010.
 * [doi:10.1002/nbm.1509](https://doi.org/10.1002/nbm.1509)
 * 
 * @param T1: species @f$T_1@f$, in s.
 * @param T2: species @f$T_2@f$, in s.
 * @param D: species diffusion tensor, in m²/s.
 * @param acquisition: acquisition parameters.
 * @param B1: relative B1, unitless.
 */
double epg_discrete_3d(
    sycomore::Species const & species,
    Acquisition const & acquisition, double B1);

#endif // _96a3fddd_ce2d_434d_b798_2ee72aba3278
