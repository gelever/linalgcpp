#include "timer.hpp"

namespace linalgcpp
{

Timer::Timer(Start start)
{
    if (static_cast<bool>(start))
    {
        Click();
    }
}

void Timer::Click()
{
    times_.push_back(std::chrono::high_resolution_clock::now());
}

double Timer::TotalTime() const
{
    if (times_.size() < 2)
    {
        return 0.0;
    }

    std::chrono::duration<double> elapsed_time = times_.back() - times_.front();
    return elapsed_time.count();
}

double Timer::operator[](int index) const
{
    assert(index >= 0);
    assert(index < static_cast<int>(times_.size()) - 1);

    std::chrono::duration<double> elapsed_time = times_[index + 1] - times_[index];
    return elapsed_time.count();
}

} // namespace linalgcpp

