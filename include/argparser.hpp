/*! @file */

#ifndef ARGPARSER_HPP__
#define ARGPARSER_HPP__

#include <map>
#include <vector>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <assert.h>

namespace linalgcpp
{

/*! @class Super simple command line argument parser
    
    Accepts flags in the form "-flag"
*/
class ArgParser
{
    public:
        /*! @brief Default Constructor */
        ArgParser() = default;

        /*! @brief Constructor from command line arguments
            @param argc number of arguments, including progam name
            @param argv command line arguments
        */
        ArgParser(int argc, const char* const* argv);

        /*! @brief Set argument if flag specified
            @param arg input argument
            @param flag flag to check
            @param help description of flag
        */
        template <typename T>
        void Parse(T& arg, const std::string& flag, const std::string& help = "") const;

        /*! @brief Show any errors that occured during parsing
            @param out output stream
        */
        void ShowErrors(std::ostream& out = std::cout) const;

        /*! @brief Show program help information
            @param out output stream
        */
        void ShowHelp(std::ostream& out = std::cout) const;

        /*! @brief Show current program parameters
            @param out output stream
        */
        void ShowOptions(std::ostream& out = std::cout) const;

        /*! @brief Check if errors occured or help is requested
            @retval true if no errors or help occured
        */
        bool IsGood() const;

    private:
        mutable std::map<std::string, std::string> help_;
        mutable std::vector<std::string> errors_;
        mutable std::map<std::string, std::string> values_;
        mutable std::map<std::string, std::string> seen_values_;

        bool need_help_ = false;
};

template <>
void ArgParser::Parse(bool& arg, const std::string& flag, const std::string& help) const;

template <typename T>
void ArgParser::Parse(T& arg, const std::string& flag, const std::string& help) const
{
    help_[flag] = help;

    if (values_.find(flag) != values_.end())
    {
        std::stringstream(values_[flag]) >> arg;
    }

    if (seen_values_.find(flag) == seen_values_.end())
    {
        std::stringstream ss;
        ss << arg;

        seen_values_[flag] = ss.str();
    }
    else
    {
        errors_.push_back(flag + " requested to be parsed multiple times!");
    }
}

} // namespace linalgcpp

#endif // ARGPARSER_HPP
