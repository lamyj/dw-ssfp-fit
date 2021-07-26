#include "models.h"

#include <cmath>
#include <Eigen/Core>
#include <sycomore/Species.h>
#include <sycomore/sycomore.h>

#include "Acquisition.h"

double freed(
    sycomore::Species const & species,
    Acquisition const & acquisition, double B1)
{
    auto const cos_alpha = std::cos(B1*acquisition.alpha.magnitude);
    auto const sin_alpha = std::sin(B1*acquisition.alpha.magnitude);
    
    auto const truncation_level = 6;
    
    auto const TR = acquisition.TR.magnitude;
    auto const R1 = species.get_R1().magnitude;
    auto const R2 = species.get_R2().magnitude;
    
    Eigen::Matrix3d D_;
    for(unsigned int row=0; row<3; ++row)
    {
        for(unsigned int col=0; col<3; ++col)
        {
            D_(row, col) = species.get_D()[3*row + col].magnitude;
        }
    }
    double const ADC = 
        acquisition.direction.transpose() * D_ * acquisition.direction;
    
    auto const G_tau = 
        acquisition.G_diffusion.magnitude * acquisition.tau_diffusion.magnitude;
    
    // Diffusion rate, at the end of the first paragraph of p. 4251
    auto const R_D = ADC * std::pow(sycomore::gamma.magnitude * G_tau, 2);
    
    // Equation 5, using definition of R_D
    auto const E1 = [&](int p) { 
        return std::exp(-TR * (R1 + R_D * std::pow(p, 2))); };
    
    // Equation 4, using definition of R_D
    auto const E2 = [&](int p) { 
        return std::exp(-TR * (R2 + R_D*(std::pow(p, 2) + p + 1./3.))); };
    
    // Equation 9
    auto const A = [&](int p) { return 0.5 * (E1(p) - 1.) * (1. + cos_alpha); };
    
    // Equation 10
    auto const B = [&](int p) { return 0.5 * (E1(p) + 1.) * (1. - cos_alpha); };
    
    // Equation 11
    auto const C = [&](int p) { return E1(p) - cos_alpha; };
    
    // Equation 24
    auto const n = [&](int p) { 
        return -E2(-p) * E2(p-1) * std::pow(A(p), 2) * B(p-1) / B(p); };

    // Equation 25
    auto const d = [&](int p) { 
        return A(p)-B(p) + E2(p) * E2(-p-1) * B(p) * C(p+1) / B(p+1); };
    
    // Equation 26 for p=l
    int l = truncation_level;
    auto const e_l = -E2(l) * E2(-l-1) * B(l) * C(l+1) / B(l+1);
    // Equation 23
    auto x_1 = n(truncation_level) / (d(truncation_level) + e_l);
    for(int l=truncation_level-1; l>0; --l)
    {
        x_1 = n(l) / (d(l)+x_1);
    }
    
    // Equation 22
    auto const r_1 = 1/(E2(-1) * B(0)) * x_1 + E2(0) * C(1) / B(1);
    
    // Equation 27
    auto const b_0 = 
        -sin_alpha * (1 - E1(0)) / (A(0) - B(0) + r_1 * E2(-1) * C(0));
    // Equation 27, with the TR modification of equation 2 in "Quantitative In
    // Vivo Diffusion Imaging of Cartilage Using Double Echo Steady-State Free
    // Precession", Bieri & al. Magnetic Resonance in Medicine 68(3). 2012.
    auto const b_minus_1 = -r_1 * E2(-1) * b_0;
    
    // b_0 is the FID mode, b_{-1} is the pre-FID mode.
    return -b_minus_1;
}
