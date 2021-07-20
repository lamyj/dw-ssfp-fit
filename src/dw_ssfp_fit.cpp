#include <vector>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <mpi4py/mpi4py.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Acquisition.h"
#include "fit.h"

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

pybind11::array_t<double>
fit_wrapper(
    std::vector<Acquisition> const & scheme, unsigned int non_dw, 
    pybind11::array_t<double> DW_SSFP, pybind11::array_t<double> T1_map,
    pybind11::array_t<double> T2_map, boost::mpi::communicator communicator,
    unsigned int population, unsigned int generations)
{
    std::size_t blocks_count;
    int block_size;
    pybind11::array_t<double> result;
    
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
        
        block_size = DW_SSFP.shape(DW_SSFP.ndim()-1);
        blocks_count = DW_SSFP.size() / block_size;
        
        std::vector<int> shape(DW_SSFP.shape(), DW_SSFP.shape()+DW_SSFP.ndim()-1);
        shape.push_back(population);
        shape.push_back(9);
        
        
        result = pybind11::array_t<double>(shape);
    }
    
    boost::mpi::broadcast(communicator, blocks_count, 0);
    boost::mpi::broadcast(communicator, block_size, 0);
    
    fit(
        scheme, non_dw, DW_SSFP.data(), T1_map.data(), T2_map.data(),
        communicator, population, generations, blocks_count, block_size, 
        result.mutable_data());
    
    return result;
}

PYBIND11_MODULE(_dw_ssfp_fit, _dw_ssfp_fit)
{
    using namespace pybind11;
    
    if(import_mpi4py() < 0)
    {
        throw pybind11::error_already_set();
    }
    
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
        "fit", &fit_wrapper, 
        arg("scheme"), arg("non_dw"), arg("DW_SSFP"), arg("T1_map"), 
        arg("T2_map"), arg("communicator"), arg("population")=100, 
        arg("generations")=100);
}
