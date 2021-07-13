#ifndef _c499e45b_1d74_45de_85e3_ab89df0f6de4
#define _c499e45b_1d74_45de_85e3_ab89df0f6de4

#include <utility>
#include <Eigen/Core>

/**
 * @brief Uniform density parameterization of a sphere from parameters in [0,1] 
 * to @f$\theta@f$ @f$\phi@f$.
 *
 * Since the area element is @f$\sin \phi\ d\theta\ d\phi@f$, uniform sampling
 * of @f$\theta@f$ and @f$\phi@f$ will yield a higher density around the poles
 * of the sphere (i.e. @f$\phi=0@f$ and @f$\phi=\pi@f$)
 */
std::pair<double, double> uniform_to_spherical(double u, double v);

/**
 * @brief Build a second-order diffusion tensor from a representation in 
 *   spherical coordinates.
 *
 * The azimuthal angle @f$\theta@f$ and the polar angle \f$\phi\f$ define the 
 * radius vector \f$\mathbf{r} = \left[
 *   \cos\theta \sin\phi, \sin\theta \sin\phi, \cos\theta \right]\f$. Given an
 * additional rotation angle \f$\psi\f$, we can define a rotation matrix aroud
 * \f$\mathbf{r}\f$, \f$\mathcal{R}(\mathbf{r}, \psi)\f$, and, given the 
 * eigenvalues \f$\lambda_1\f$, \f$\lambda_2\f$, and \f$\lambda_3\f$, a 
 * diffusion tensor is created by \f$D = 
 *   \mathcal{R}(\mathbf{r}, \psi) 
 *     \text{diag}(\lambda_1, \lambda_2, \lambda_3)
 *   \mathcal{R}(\mathbf{r}, \psi)^T\f$.
 *
 * @param theta azimuthal angle in the @f$(x,y)@f$ plane from the @f$x@f$ axis 
 *   (@f$0 \le \theta \le 2\pi@f$)
 * @param phi polar angle from the positive @f$z@f$ axis, also known as 
 *   colatitude or inclination angle (@f$0 \le \phi \le \pi@f$)
 * @param psi rotation angle around the radius vector defined by @f$\theta@f$ 
 *   and @f$\phi@f$ (@f$-\pi \le \psi \le \pi@f$)
 * @param lambda1 largest eigenvalue
 * @param lambda2 second eigenvalue
 * @param lambda3 smallest eigenvalue
 */
Eigen::Matrix3d build_diffusion_tensor(
    double theta, double phi, double psi, 
    double lambda1, double lambda2, double lambda3);

/**
 * @brief Build a second-order diffusion tensor from a representation in 
 *   spherical coordinates, store it in pre-allocated array.
 * 
 * See @ref build_diffusion_tensor(double, double, double, double, double, double)
 * "other definition" for details.
 * 
 * @param D Array of size 9, managed by caller
 */
void build_diffusion_tensor(
    double theta, double phi, double psi, 
    double lambda1, double lambda2, double lambda3,
    double * D);

#endif // _c499e45b_1d74_45de_85e3_ab89df0f6de4
