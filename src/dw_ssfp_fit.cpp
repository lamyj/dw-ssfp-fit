#include <vector>

#include <boost/mpi/collectives.hpp>
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

pybind11::tuple
fit_wrapper(
    std::vector<Acquisition> const & scheme, unsigned int non_dw, 
    pybind11::array_t<double> DW_SSFP, pybind11::array_t<double> T1_map,
    pybind11::array_t<double> T2_map, pybind11::array_t<double> B1_map,
    boost::mpi::communicator communicator,
    unsigned int population, unsigned int generations,
    bool return_individuals, bool return_champions)
{
    using namespace std::string_literals;
    
    std::size_t blocks_count;
    int block_size;
    pybind11::array_t<double> individuals, champions;
    
    if(communicator.rank() == 0)
    {
        for(auto && item: {
            std::make_pair("T1_map", T1_map), std::make_pair("T2_map", T2_map),
            std::make_pair("B1_map", B1_map)})
        {
            if(DW_SSFP.ndim() != 1+item.second.ndim())
            {
                throw std::runtime_error("ndim mismatch: "s + item.first);
            }
            for(int i=0, end=item.second.ndim(); i!=end; ++i)
            {
                if(DW_SSFP.shape(i) != item.second.shape(i))
                {
                    throw std::runtime_error("shape mismatch: "s + item.first);
                }
            }
        }
        
        block_size = DW_SSFP.shape(DW_SSFP.ndim()-1);
        blocks_count = DW_SSFP.size() / block_size;
        
        if(return_individuals)
        {
            std::vector<int> shape(
                DW_SSFP.shape(), DW_SSFP.shape()+DW_SSFP.ndim()-1);
            shape.push_back(population);
            shape.push_back(3);
            shape.push_back(3);
            
            individuals = pybind11::array_t<double>(shape);
        }
        else
        {
            individuals = pybind11::array_t<double>();
        }
        
        if(return_champions)
        {
            std::vector<int> shape(
                DW_SSFP.shape(), DW_SSFP.shape()+DW_SSFP.ndim()-1);
            shape.push_back(3);
            shape.push_back(3);
            
            champions = pybind11::array_t<double>(shape);
        }
        else
        {
            champions = pybind11::array_t<double>();
        }
    }
    
    boost::mpi::broadcast(communicator, blocks_count, 0);
    boost::mpi::broadcast(communicator, block_size, 0);
    
    fit(
        scheme, non_dw, DW_SSFP.data(), T1_map.data(), T2_map.data(),
        B1_map.data(), communicator, population, generations, blocks_count, 
        block_size, 
        return_individuals?individuals.mutable_data():nullptr,
        return_champions?champions.mutable_data():nullptr);
    
    return pybind11::make_tuple(
        return_individuals?individuals.cast<pybind11::object>():pybind11::none(),
        return_champions?champions.cast<pybind11::object>():pybind11::none());
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
        "scheme"_a, "non_dw"_a, "DW_SSFP"_a, "T1_map"_a, "T2_map"_a, "B1_map"_a,
        "communicator"_a, "population"_a=100, "generations"_a=100,
        "return_individuals"_a=true, "return_champions"_a=true);
}
