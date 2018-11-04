/*! @file */

#ifndef UTILITIES_HPP__
#define UTILITIES_HPP__

namespace linalgcpp
{

/*! @brief Throw if false in debug mode only */
inline
void linalgcpp_assert(bool expression, const std::string& message = "linalgcpp assertion failed")
{
#ifndef NDEBUG
    if (!expression)
    {
        throw std::runtime_error(message);
    }
#endif
}

/*! @brief Throw if false unconditionally */
inline
void linalgcpp_verify(bool expression, const std::string& message = "linalgcpp verification failed")
{
    if (!expression)
    {
        throw std::runtime_error(message);
    }
}

} //namespace linalgcpp

#endif // UTILITIES_HPP__

