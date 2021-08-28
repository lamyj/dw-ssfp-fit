#include "Acquisition.h"

#include <map>
#include <Eigen/Core>
#include <sycomore/Quantity.h>
#include <sycomore/sycomore.h>
#include <sycomore/TimeInterval.h>
#include <sycomore/units.h>

using namespace sycomore::units;

Acquisition
::Acquisition(
    sycomore::Quantity alpha, 
    sycomore::Quantity G_diffusion, sycomore::Quantity tau_diffusion,
        Eigen::Vector3d direction,
    sycomore::Quantity TE, sycomore::Quantity TR,
    sycomore::Quantity pixel_bandwidth,
    Eigen::Vector2i shape, sycomore::Array<sycomore::Quantity> FOV,
    unsigned int train_length, sycomore::Quantity G_max)
: alpha(alpha),
    G_diffusion(G_diffusion), tau_diffusion(tau_diffusion), direction(direction),
    TE(TE), TR(TR),
    pixel_bandwidth(pixel_bandwidth), shape(shape), FOV(FOV),
    train_length(train_length), G_max(G_max)
{
    this->direction.normalize();
}

Acquisition
::Acquisition(
    double alpha, 
    double G_diffusion, double tau_diffusion, Eigen::Vector3d direction,
    double TE, double TR,
    double pixel_bandwidth, Eigen::Vector2i shape, Eigen::Vector2d FOV,
    unsigned int train_length, double G_max)
: Acquisition(
    alpha*rad, 
    G_diffusion*T/m, tau_diffusion*s, direction,
    TE*s, TR*s,
    pixel_bandwidth*Hz, shape, {FOV[0]*m, FOV[1]*m}, train_length, G_max*T/m)
{
    // Nothing else.
}
