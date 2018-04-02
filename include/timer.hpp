/*! @file */

#ifndef TIMER_HPP__
#define TIMER_HPP__

#include <chrono>
#include <vector>
#include <assert.h>

namespace linalgcpp
{

/*! @brief Timer implemented as stop watch */

class Timer
{
    public:
        enum class Start : bool { True = true, False = false };

        Timer(Start start = Start::False);
        void Click();
        double TotalTime() const;
        double operator[](int index) const;

    private:
        std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> times_;

};

} // namespace linalgcpp

#endif // TIMER_HPP
