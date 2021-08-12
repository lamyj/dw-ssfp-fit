#include "models.h"

#include <cmath>
#include <iostream>

#include <Eigen/Core>
#include <sycomore/epg/Discrete.h>
#include <sycomore/epg/Discrete3D.h>
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

double epg_discrete_1d(
    sycomore::Species const & species,
    Acquisition const & acquisition, double B1)
{
    using namespace sycomore::units;

    bool stable=false;
    
    Eigen::Matrix3d D_;
    for(unsigned int row=0; row<3; ++row)
    {
        for(unsigned int col=0; col<3; ++col)
        {
            D_(row, col) = species.get_D()[3*row + col].convert_to(std::pow(m, 2)/s);
        }
    }
    double const ADC = 
        acquisition.direction.transpose() * D_ * acquisition.direction;
    sycomore::Species const isotropic_species(
        species.get_R1(), species.get_R2(), ADC*std::pow(m, 2)/s);    
    
    int const repetitions = std::max<int>(1, 5*species.get_T1()/acquisition.TR);
    std::vector<double> signal;
    signal.reserve(repetitions);
    
    sycomore::epg::Discrete model(isotropic_species, {0,0,1}, 1e-6*rad/m);
    model.threshold = 1e-4;
    
    try
    {
        while(!stable && signal.size() < repetitions)
        {
            model.apply_pulse(acquisition.alpha*B1);
            
            model.apply_time_interval(acquisition.idle);
            model.apply_time_interval(
                acquisition.tau_diffusion, acquisition.G_diffusion);
            model.apply_time_interval(acquisition.idle);
            
            model.apply_time_interval(acquisition.ro_minus);
            model.apply_time_interval(acquisition.ro_plus);
            signal.push_back((model.size() > 0) ? std::abs(model.echo()) : 0);
            model.apply_time_interval(acquisition.ro_plus);
            model.apply_time_interval(acquisition.ro_minus);

            model.apply_time_interval(acquisition.end_of_TR);

            if(signal.size() > 20)
            {
                auto const begin = signal.end()-20;
                auto const end = signal.end();
                auto const mean = 1./20. * std::accumulate(begin, end, 0.);
                auto const range = std::minmax_element(begin, end);
            
                if((*(range.second)-*(range.first))/mean < 1e-2)
                {
                    stable = true;
                }
            }
        }
    }
    catch(std::exception const & e)
    {
        std::cerr
            << "Error in simulation: " << e.what() << ". "
            << "alpha=" << acquisition.alpha << ", "
            << "G_diffusion=" << acquisition.G_diffusion << ", "
            << "tau_diffusion=" << acquisition.tau_diffusion << ", "
            << "direction=" << acquisition.direction << ", "
            << "TE=" << acquisition.TE << ", "
            << "TR=" << acquisition.TR << ", "
            << "pixel_bandwidth=" << acquisition.pixel_bandwidth << ", "
            << "resolution=" << acquisition.resolution << ", "
            << "G_max=" << acquisition.G_max << ", "
            << "T1=" << species.get_T1() << ", "
            << "T2=" << species.get_T2() << ", "
            << "B1=" << B1 << ", "
            << "D=" << D_
            << std::endl;
        return 1e9;
    }
    catch(...)
    {
        std::cerr
            << "Error in simulation (unknown). "
            << "alpha=" << acquisition.alpha << ", "
            << "G_diffusion=" << acquisition.G_diffusion << ", "
            << "tau_diffusion=" << acquisition.tau_diffusion << ", "
            << "direction=" << acquisition.direction << ", "
            << "TE=" << acquisition.TE << ", "
            << "TR=" << acquisition.TR << ", "
            << "pixel_bandwidth=" << acquisition.pixel_bandwidth << ", "
            << "resolution=" << acquisition.resolution << ", "
            << "G_max=" << acquisition.G_max << ", "
            << "T1=" << species.get_T1() << ", "
            << "T2=" << species.get_T2() << ", "
            << "B1=" << B1 << ", "
            << "D=" << D_
            << std::endl;
        return 1e9;
    }
    
    auto const size = std::min(signal.size(), 20UL);
    if(size == 0)
    {
        std::cerr
            << "Signal is empty. "
            << "Error in simulation (unknown). "
            << "alpha=" << acquisition.alpha << ", "
            << "G_diffusion=" << acquisition.G_diffusion << ", "
            << "tau_diffusion=" << acquisition.tau_diffusion << ", "
            << "direction=" << acquisition.direction << ", "
            << "TE=" << acquisition.TE << ", "
            << "TR=" << acquisition.TR << ", "
            << "pixel_bandwidth=" << acquisition.pixel_bandwidth << ", "
            << "resolution=" << acquisition.resolution << ", "
            << "G_max=" << acquisition.G_max << ", "
            << "T1=" << species.get_T1() << ", "
            << "T2=" << species.get_T2() << ", "
            << "B1=" << B1 << ", "
            << "D=" << D_
            << std::endl;
        return 1e9;
    }
    auto const begin = signal.end()-size;
    auto const end = signal.end();
    auto const mean = 1./size * std::accumulate(begin, end, 0.);
    
    return mean;
}

double epg_discrete_3d(
    sycomore::Species const & species,
    Acquisition const & acquisition, double B1)
{
    using namespace sycomore::units;

    bool stable=false;
    
    int const repetitions = 5*species.get_T1()/acquisition.TR;

    std::vector<double> signal;
    signal.reserve(repetitions);

    sycomore::epg::Discrete3D model(species, {0,0,1}, 1e-6*rad/m);

    while(signal.size() < repetitions)
    {
        model.apply_pulse(acquisition.alpha*B1);
        
        model.apply_time_interval(acquisition.idle);
        model.apply_time_interval(acquisition.diffusion);
        model.apply_time_interval(acquisition.idle);
        
        model.apply_time_interval(acquisition.ro_minus);
        model.apply_time_interval(acquisition.ro_plus);
        signal.push_back(std::abs(model.echo()));
        model.apply_time_interval(acquisition.ro_plus);
        model.apply_time_interval(acquisition.ro_minus);

        model.apply_time_interval(acquisition.end_of_TR);
    }
    
    auto const begin = signal.end()-20;
    auto const end = signal.end();
    auto const mean = 1./20. * std::accumulate(begin, end, 0.);
    
    return mean;
}
