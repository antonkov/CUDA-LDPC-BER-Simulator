#include "filesystem.h"

#include <boost/filesystem.hpp>
#include <algorithm>

namespace fs = boost::filesystem;

std::vector<std::string> filesInDirectory(std::string const & dirName)
{
    std::vector<std::string> res;
    fs::path dirPath(dirName);
    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(dirPath); iter != end_iter; iter++)
    {
        if (fs::is_regular_file(iter->status()))
            res.push_back(iter->path().string());
    }
    std::sort(res.begin(), res.end());
    return res;
}

bool isDirectory(std::string const & name)
{
    fs::path p(name);
    return fs::exists(p) && fs::is_directory(p);
}
