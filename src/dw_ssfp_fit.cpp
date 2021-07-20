#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de1220.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problem.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sycomore/Quantity.h>
#include <sycomore/units.h>

#include "Acquisition.h"
#include "diffusion_tensor.h"
#include "models.h"
#include "Problem.h"

pybind11::array_t<double> fit(
    pybind11::sequence scheme_py, unsigned int non_dw, 
    pybind11::array_t<double> signals_py, double T1, double T2,
    unsigned int population, unsigned int generations, unsigned int jobs,
    unsigned int verbosity)
{
    using namespace pybind11;
    using namespace sycomore;
    using namespace sycomore::units;
    
    std::vector<Acquisition> scheme;
    for(auto && item: scheme_py)
    {
        auto & acquisition = scheme.emplace_back();
        acquisition.alpha = item["alpha"].cast<double>();
        acquisition.G_diffusion = item["G_diffusion"].cast<double>();
        acquisition.tau_diffusion = item["tau_diffusion"].cast<double>();
        acquisition.direction = Eigen::Map<Eigen::VectorXd>{
            item["direction"].cast<array_t<double>>().mutable_data(), 3};
        acquisition.TR = item["TR"].cast<double>();
        acquisition.TR = item["TE"].cast<double>();
        acquisition.pixel_bandwidth = item["pixel_bandwidth"].cast<double>();
        acquisition.resolution = item["resolution"].cast<double>();
        acquisition.G_max = item["G_max"].cast<double>();
    }
    
    if(signals_py.ndim() != 1 || signals_py.shape()[0] != scheme.size())
    {
        throw std::runtime_error(
            "Scheme and signals size mismatch: "
            + std::to_string(scheme.size()) + " != " 
            + std::to_string(signals_py.shape()[0]));
    }
    std::vector<double> signals(signals_py.shape()[0]);
    for(std::size_t i=0, end=signals.size(); i!=end; ++i)
    {
        signals[i] = signals_py.at(i);
    }
    
    Problem problem{scheme, non_dw, signals, T1, T2, freed};
    
    // NOTE: data/model do not allow to easily specify ftol/xtol
    pagmo::algorithm algorithm{pagmo::de1220{generations}};
    algorithm.set_verbosity(verbosity);
    
    pagmo::archipelago archipelago{jobs, algorithm, problem, population};
    archipelago.evolve();
    archipelago.wait_check();
    
    long const total_population = std::accumulate(
        archipelago.begin(), archipelago.end(), 0,
        [](auto a, auto island) { return a+island.get_population().size(); });
    
    pybind11::array_t<double> D_array{{total_population, 3L, 3L}};
    std::size_t index = 0;
    for(auto && island: archipelago)
    {
        auto const population = island.get_population();
        for(auto && scaled_dv: population.get_x())
        {
            auto const true_dv = Problem::get_true_dv(scaled_dv);
            auto const D = Problem::get_diffusion_tensor(true_dv);
            std::copy(D.data(), D.data()+D.size(), D_array.mutable_data(index));
            
            ++index;
        }
    }
    
    return D_array;
}

PYBIND11_MODULE(_dw_ssfp_fit, _dw_ssfp_fit)
{
    using namespace pybind11;
    
    class_<Acquisition>(_dw_ssfp_fit, "Acquisition")
        .def(init(
            [](
                double alpha, double G_diffusion, double tau_diffusion, 
                Eigen::Vector3d direction, double TE, double TR, 
                double pixel_bandwidth, double resolution, double G_max) 
            { 
                return Acquisition{
                    alpha, G_diffusion, tau_diffusion, direction, TE, TR,
                    pixel_bandwidth, resolution, G_max};
            }),
            arg("alpha")=0, arg("G_diffusion")=0, arg("tau_diffusion")=0, 
            arg("direction")=Eigen::Vector3d{1,0,0}, arg("TE")=0, arg("TR")=0,
            arg("pixel_bandwidth")=0, arg("resolution")=0, arg("G_max")=0)
        .def_readwrite("alpha", &Acquisition::alpha)
        .def_readwrite("G_diffusion", &Acquisition::G_diffusion)
        .def_readwrite("tau_diffusion", &Acquisition::tau_diffusion)
        .def_readwrite("direction", &Acquisition::direction)
        .def_readwrite("TE", &Acquisition::TE)
        .def_readwrite("TR", &Acquisition::TR)
        .def_readwrite("pixel_bandwidth", &Acquisition::pixel_bandwidth)
        .def_readwrite("resolution", &Acquisition::resolution)
        .def_readwrite("G_max", &Acquisition::G_max);
    
    _dw_ssfp_fit.def(
        "fit", &fit, 
        arg("scheme"), arg("non_dw"), arg("signals"), arg("T1"), arg("T2"),
        arg("population")=100, arg("generations")=100, arg("jobs")=1,
        arg("verbosity")=0);
}
