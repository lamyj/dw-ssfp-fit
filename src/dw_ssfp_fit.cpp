#include <stdexcept>
#include <vector>

#include <boost/mpi/communicator.hpp>
#include <mpi4py/mpi4py.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Acquisition.h"
#include "benchmark.h"
#include "diffusion_tensor.h"
#include "fit.h"
#include "models.h"

namespace pybind11 { namespace detail {

template<>
struct type_caster<boost::mpi::communicator>
{
public:
    PYBIND11_TYPE_CASTER(boost::mpi::communicator, _("communicator"));
    bool load(handle src, bool)
    {
        auto communicator_pointer = PyMPIComm_Get(src.ptr());
        if(!communicator_pointer)
        {
            return false;
        }
        value = boost::mpi::communicator(
            *communicator_pointer, boost::mpi::comm_attach);
        return true;
    }
};

} }

pybind11::tuple
fit_wrapper(
    std::vector<Acquisition> const & scheme, unsigned int non_dw, 
    pybind11::array_t<double> DW_SSFP, pybind11::array_t<double> B1_map,
    pybind11::array_t<double> T1_map, pybind11::array_t<double> T2_map,
    boost::mpi::communicator communicator,
    unsigned int population, unsigned int generations,
    bool return_individuals, bool return_champions)
{
    using namespace std::string_literals;
    
    pybind11::array_t<double> champions_D, champions_T1, champions_T2;
    pybind11::array_t<double> individuals_D, individuals_T1, individuals_T2;
    
    if(communicator.rank() == 0)
    {
        std::vector<int> const champions_shape(
            DW_SSFP.shape(), DW_SSFP.shape()+DW_SSFP.ndim()-1);
        
        auto champions_shape_D{champions_shape};
        champions_shape_D.push_back(3);
        champions_shape_D.push_back(3);
        champions_D = pybind11::array_t<double>(champions_shape_D);
        
        if(T1_map.size() == 0)
        {
            champions_T1 = pybind11::array_t<double>(champions_shape);
        }
        
        if(T2_map.size() == 0)
        {
            champions_T2 = pybind11::array_t<double>(champions_shape);
        }
        
        if(return_individuals)
        {
            auto individuals_shape(champions_shape);
            individuals_shape.push_back(population);
            
            auto individuals_shape_D(individuals_shape);
            individuals_shape_D.push_back(3);
            individuals_shape_D.push_back(3);
            individuals_D = pybind11::array_t<double>(individuals_shape_D);
            
            if(T1_map.size() == 0)
            {
                individuals_T1 = pybind11::array_t<double>(individuals_shape);
            }
            
            if(T2_map.size() == 0)
            {
                individuals_T2 = pybind11::array_t<double>(individuals_shape);
            }
        }
    }
    
    fit(
        scheme, non_dw, 
        {DW_SSFP.data(), DW_SSFP.size()}, {B1_map.data(), B1_map.size()},
        {T1_map.data(), T1_map.size()}, {T2_map.data(), T2_map.size()},
        population, generations, 
        {champions_D.mutable_data(), champions_D.size()},
        {champions_T1.mutable_data(), champions_T1.size()},
        {champions_T2.mutable_data(), champions_T2.size()},
        {individuals_D.mutable_data(), individuals_D.size()},
        {individuals_T1.mutable_data(), individuals_T1.size()},
        {individuals_T2.mutable_data(), individuals_T2.size()},
        communicator);
    
    return pybind11::make_tuple(
        champions_D, champions_T1, champions_T2,
        individuals_D, individuals_T1, individuals_T2);
}

PYBIND11_MODULE(_dw_ssfp_fit, _dw_ssfp_fit)
{
    using namespace pybind11::literals;
    using namespace sycomore::units;
    
    if(import_mpi4py() < 0)
    {
        throw pybind11::error_already_set();
    }
    
    pybind11::class_<Acquisition>(_dw_ssfp_fit, "Acquisition")
        .def(pybind11::init<
                sycomore::Quantity, sycomore::Quantity, sycomore::Quantity, 
                Eigen::Vector3d, sycomore::Quantity, sycomore::Quantity, 
                sycomore::Quantity, Eigen::Vector2i,
                sycomore::Array<sycomore::Quantity>, 
                unsigned int, sycomore::Quantity>(),
            "alpha"_a=0*rad, "G_diffusion"_a=0*T/m, "tau_diffusion"_a=0*s,
            "direction"_a=Eigen::Vector3d{1,0,0}, "TE"_a=0*s, "TR"_a=0*s,
            "pixel_bandwidth"_a=0*Hz, "shape"_a=Eigen::Vector2i{0,0},
            "FOV"_a=sycomore::Array<sycomore::Quantity>{0*m,0*m},
            "train_length"_a=0, "G_max"_a=0*T/m)
        .def(pybind11::init<
                double, double, double,
                Eigen::Vector3d, double, double, 
                double, Eigen::Vector2i, 
                Eigen::Vector2d, unsigned int, double>(),
            "alpha"_a=0, "G_diffusion"_a=0, "tau_diffusion"_a=0,
            "direction"_a=Eigen::Vector3d{1,0,0}, "TE"_a=0, "TR"_a=0,
            "pixel_bandwidth"_a=0, "shape"_a=Eigen::Vector2i{0,0}, 
            "FOV"_a=Eigen::Vector2d{0,0}, "train_length"_a=0, "G_max"_a=0)
        .def_readwrite("alpha", &Acquisition::alpha)
        .def_readwrite("G_diffusion", &Acquisition::G_diffusion)
        .def_readwrite("tau_diffusion", &Acquisition::tau_diffusion)
        .def_readwrite("direction", &Acquisition::direction)
        .def_readwrite("TE", &Acquisition::TE)
        .def_readwrite("TR", &Acquisition::TR)
        .def_readwrite("pixel_bandwidth", &Acquisition::pixel_bandwidth)
        .def_readwrite("shape", &Acquisition::shape)
        .def_readwrite("FOV", &Acquisition::FOV)
        .def_readwrite("train_length", &Acquisition::train_length)
        .def_readwrite("G_max", &Acquisition::G_max)
        .def_readwrite("diffusion", &Acquisition::diffusion)
        .def_readwrite("idle", &Acquisition::idle)
        .def_readwrite("readout_preparation", &Acquisition::readout_preparation)
        .def_readwrite("half_readout", &Acquisition::half_readout)
        .def_readwrite("phase_blip", &Acquisition::phase_blip)
        .def_readwrite("readout_rewind", &Acquisition::readout_rewind)
        .def_readwrite("end_of_TR", &Acquisition::end_of_TR);
    
    _dw_ssfp_fit.def("freed", &freed, "species"_a, "acquisition"_a, "B1"_a);
    _dw_ssfp_fit.def(
        "epg_discrete_1d", &epg_discrete_1d,
        "species"_a, "acquisition"_a, "B1"_a);
    _dw_ssfp_fit.def(
        "epg_discrete_3d", &epg_discrete_3d,
        "species"_a, "acquisition"_a, "B1"_a);
    
    _dw_ssfp_fit.def(
        "benchmark_freed", 
        [](
            sycomore::Species const & species, Acquisition const & acquisition,
            std::size_t count)
        {
            return benchmark(species, acquisition, &freed, count);
        }, 
        "species"_a, "acquisition"_a, "count"_a);
    _dw_ssfp_fit.def(
        "benchmark_epg_discrete_1d", 
        [](
            sycomore::Species const & species, Acquisition const & acquisition,
            std::size_t count)
        {
            return benchmark(species, acquisition, &epg_discrete_1d, count);
        }, 
        "species"_a, "acquisition"_a, "count"_a);
    _dw_ssfp_fit.def(
        "benchmark_epg_discrete_3d", 
        [](
            sycomore::Species const & species, Acquisition const & acquisition,
            std::size_t count)
        {
            return benchmark(species, acquisition, &epg_discrete_3d, count);
        }, 
        "species"_a, "acquisition"_a, "count"_a);
    
    _dw_ssfp_fit.def(
        "fit", &fit_wrapper, 
        "scheme"_a, "non_dw"_a, "DW_SSFP"_a, "B1_map"_a, "T1_map"_a, "T2_map"_a,
        "communicator"_a, "population"_a=100, "generations"_a=100,
        "return_individuals"_a=true, "return_champions"_a=true);
}
