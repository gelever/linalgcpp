#include "utilities.hpp"

namespace linalgcpp
{

void SetMarker(std::vector<int>& marker, const std::vector<int>& indices)
{
    const int size = indices.size();

    for (int i = 0; i < size; ++i)
    {
        assert(indices[i] < static_cast<int>(marker.size()));

        marker[indices[i]] = i;
    }
}

void ClearMarker(std::vector<int>& marker, const std::vector<int>& indices)
{
    const int size = indices.size();

    for (int i = 0; i < size; ++i)
    {
        assert(indices[i] < static_cast<int>(marker.size()));

        marker[indices[i]] = -1;
    }
}

} // namespace linalgcpp
