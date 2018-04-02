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

/*! @brief Super simple command line argument parser */

class ArgParser
{
    public:
        ArgParser() = default;
        ArgParser(int argc, const char* const* argv);

        template <typename T>
        void Parse(T& arg, const std::string& flag, const std::string& help = "") const;

        void ShowErrors(std::ostream& out = std::cout) const;
        void ShowHelp(std::ostream& out = std::cout) const;
        void ShowOptions(std::ostream& out = std::cout) const;

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
