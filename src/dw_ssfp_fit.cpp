#include <vector>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <mpi4py/mpi4py.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Acquisition.h"
#include "diffusion_tensor.h"
#include "fit.h"
#include "models.h"
#include "Problem.h"

namespace pybind11 { namespace detail {

template<>
struct type_caster<boost::mpi::communicator> {
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

std::pair<pybind11::array_t<double>, pybind11::array_t<double>>
fit_wrapper(
    std::vector<Acquisition> const & scheme, unsigned int non_dw, 
    pybind11::array_t<double> DW_SSFP, pybind11::array_t<double> T1_map,
    pybind11::array_t<double> T2_map, pybind11::array_t<double> B1_map,
    boost::mpi::communicator communicator,
    unsigned int population, unsigned int generations, bool return_tensor)
{
    std::size_t blocks_count;
    int block_size;
    pybind11::array_t<double> individuals, champions;
    
    if(communicator.rank() == 0)
    {
        if(DW_SSFP.ndim() != 1+T1_map.ndim())
        {
            throw std::runtime_error("DW_SSFP and T1_map dimensions don't match");
        }
        if(DW_SSFP.ndim() != 1+T2_map.ndim())
        {
            throw std::runtime_error("DW_SSFP and T2_map dimensions don't match");
        }
        if(DW_SSFP.ndim() != 1+B1_map.ndim())
        {
            throw std::runtime_error("DW_SSFP and B1_map dimensions don't match");
        }
        
        for(int i=0, end=T1_map.ndim(); i!=end; ++i)
        {
            if(DW_SSFP.shape(i) != T1_map.shape(i))
            {
                throw std::runtime_error("DW_SSFP and T1_map shapes don't match");
            }
            if(DW_SSFP.shape(i) != T2_map.shape(i))
            {
                throw std::runtime_error("DW_SSFP and T2_map shapes don't match");
            }
            if(DW_SSFP.shape(i) != B1_map.shape(i))
            {
                throw std::runtime_error("DW_SSFP and B1_map shapes don't match");
            }
        }
        
        if(!(DW_SSFP.flags() & pybind11::array::c_style))
        {
            throw std::runtime_error("DW_SSFP is not contiguous");
        }
        if(!(T1_map.flags() & pybind11::array::c_style))
        {
            throw std::runtime_error("T1_map is not contiguous");
        }
        if(!(T2_map.flags() & pybind11::array::c_style))
        {
            throw std::runtime_error("T2_map is not contiguous");
        }
        if(!(B1_map.flags() & pybind11::array::c_style))
        {
            throw std::runtime_error("B1_map is not contiguous");
        }
        
        block_size = DW_SSFP.shape(DW_SSFP.ndim()-1);
        blocks_count = DW_SSFP.size() / block_size;
        
        std::vector<int> shape(DW_SSFP.shape(), DW_SSFP.shape()+DW_SSFP.ndim()-1);
        shape.push_back(population);
        if(return_tensor)
        {
            shape.push_back(3);
            shape.push_back(3);
        }
        else
        {
            shape.push_back(6);
        }
        
        individuals = pybind11::array_t<double>(shape);
        
        shape = std::vector<int>(DW_SSFP.shape(), DW_SSFP.shape()+DW_SSFP.ndim()-1);
        if(return_tensor)
        {
            shape.push_back(3);
            shape.push_back(3);
        }
        else
        {
            shape.push_back(6);
        }
        
        champions = pybind11::array_t<double>(shape);
    }
    
    boost::mpi::broadcast(communicator, blocks_count, 0);
    boost::mpi::broadcast(communicator, block_size, 0);
    
    fit(
        scheme, non_dw, DW_SSFP.data(), T1_map.data(), T2_map.data(),
        B1_map.data(), communicator, population, generations, blocks_count, 
        block_size, return_tensor, 
        individuals.mutable_data(), champions.mutable_data());
    
    return {individuals, champions};
}

pybind11::array_t<double>
build_diffusion_tensor(pybind11::array_t<double> dv_array)
{
    std::vector<int> shape(dv_array.shape(), dv_array.shape()+dv_array.ndim()-1);
    shape.push_back(3);
    shape.push_back(3);
    pybind11::array_t<double> D_array(shape);
    
    auto dv_it = dv_array.data();
    auto dv_end = dv_array.data()+dv_array.size();
    std::vector<double> scaled_dv(6);
    auto D_it = D_array.mutable_data();
    Eigen::Matrix3d D;
    
    while(dv_it != dv_end)
    {
        std::copy(dv_it, dv_it+6, scaled_dv.begin());
        dv_it += 6;
        
        auto const true_dv = Problem::get_true_dv(scaled_dv);
        D = Problem::get_diffusion_tensor(true_dv);
        
        std::copy(D.data(), D.data()+D.size(), D_it);
        D_it += 9;
    }
    
    return D_array;
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
                sycomore::Quantity, 
                sycomore::Quantity, sycomore::Quantity, Eigen::Vector3d, 
                sycomore::Quantity, sycomore::Quantity, 
                sycomore::Quantity, sycomore::Quantity, sycomore::Quantity>(),
            "alpha"_a=0*rad,
            "G_diffusion"_a=0*T/m, "tau_diffusion"_a=0*s,
                "direction"_a=Eigen::Vector3d{1,0,0},
            "TE"_a=0*s, "TR"_a=0*s,
            "pixel_bandwidth"_a=0*Hz, "resolution"_a=0*m, "G_max"_a=0*T/m)
        .def(pybind11::init<
                double, 
                double, double, Eigen::Vector3d, 
                double, double, 
                double, double, double>(),
            "alpha"_a=0,
            "G_diffusion"_a=0, "tau_diffusion"_a=0,
                "direction"_a=Eigen::Vector3d{1,0,0},
            "TE"_a=0, "TR"_a=0,
            "pixel_bandwidth"_a=0, "resolution"_a=0, "G_max"_a=0)
        .def_readwrite("alpha", &Acquisition::alpha)
        .def_readwrite("G_diffusion", &Acquisition::G_diffusion)
        .def_readwrite("tau_diffusion", &Acquisition::tau_diffusion)
        .def_readwrite("direction", &Acquisition::direction)
        .def_readwrite("TE", &Acquisition::TE)
        .def_readwrite("TR", &Acquisition::TR)
        .def_readwrite("pixel_bandwidth", &Acquisition::pixel_bandwidth)
        .def_readwrite("resolution", &Acquisition::resolution)
        .def_readwrite("G_max", &Acquisition::G_max)
        .def_readwrite("diffusion", &Acquisition::diffusion)
        .def_readwrite("ro_plus", &Acquisition::ro_plus)
        .def_readwrite("ro_minus", &Acquisition::ro_minus)
        .def_readwrite("end_of_TR", &Acquisition::end_of_TR);
    
    _dw_ssfp_fit.def("freed", &freed, "species"_a, "acquisition"_a, "B1"_a);
    _dw_ssfp_fit.def(
        "epg_discrete_1d", &epg_discrete_1d,
        "species"_a, "acquisition"_a, "B1"_a);
    _dw_ssfp_fit.def(
        "epg_discrete_3d", &epg_discrete_3d,
        "species"_a, "acquisition"_a, "B1"_a);
    
    _dw_ssfp_fit.def(
        "uniform_to_spherical", &uniform_to_spherical, "u"_a, "v"_a);
    
    _dw_ssfp_fit.def(
        "build_diffusion_tensor", 
        pybind11::overload_cast<double, double, double, double, double, double>(
            &build_diffusion_tensor), 
        "theta"_a, "phi"_a, "psi"_a, "lambda1"_a, "lambda2"_a, "lambda3"_a);
    
    _dw_ssfp_fit.def(
        "build_diffusion_tensor", 
        pybind11::overload_cast<pybind11::array_t<double>>(
            &build_diffusion_tensor),
        "array"_a);
    
    _dw_ssfp_fit.def(
        "fit", &fit_wrapper, 
        "scheme"_a, "non_dw"_a, "DW_SSFP"_a, "T1_map"_a, "T2_map"_a, "B1_map"_a,
        "communicator"_a, "population"_a=100, "generations"_a=100,
        "return_tensor"_a=true);
}
