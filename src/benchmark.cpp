#include "benchmark.h"

#include <algorithm>
#include <chrono>
#include <vector>

#include <sycomore/Species.h>

#include "Acquisition.h"
#include "Problem.h"

double benchmark(
    sycomore::Species const & species, Acquisition const & acquisition,
    Problem::Simulator simulator, std::size_t count)
{
    using Clock = std::chrono::high_resolution_clock;
    constexpr auto const ticks_per_second = 
        double(Clock::period::den) / double(Clock::period::num);
    
    std::vector<double> duration(count, 0.);
    for(std::size_t i=0; i!=count; ++i)
    {
        auto const now = Clock::now();
        simulator(species, acquisition, 1.0);
        duration[i] = (Clock::now()-now).count()/ticks_per_second;
    }
    return *std::min_element(duration.begin(), duration.end());
}
