#pragma once

#include "dem_constant.h"
#include <cuda_runtime.h>
#include<boost/filesystem.hpp>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Native little-endian field blocks, matching the supplied SaveParticles_Bin
// and SaveContInfo_Bin layouts. No record padding or additional file header.
namespace dem_binary
{
static_assert(CHAR_BIT == 8 && sizeof(int) == 4, "Binary output requires int32.");
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "Binary output requires IEEE-754 float64.");
static_assert(sizeof(double3) == 24, "Unexpected double3 padding.");
static_assert(sizeof(double4) == 32, "Unexpected double4 padding.");

inline std::size_t Count(int count)
{
    if (count < 0) throw std::runtime_error("Negative binary output count.");
    return static_cast<std::size_t>(count);
}

inline std::size_t Multiply(std::size_t count, std::size_t elementSize)
{
    if (elementSize != 0 && count > (std::numeric_limits<std::size_t>::max)() / elementSize)
        throw std::runtime_error("Binary output size overflow.");
    return count * elementSize;
}

inline int NextStep(int step)
{
    if (step < -1 || step == (std::numeric_limits<int>::max)())
        throw std::runtime_error("Invalid binary output step.");
    return step + 1;
}

inline std::string FileName(const char* directory, const char* prefix, int step)
{
    Count(step);
    boost::filesystem::create_directories(directory);
    std::ostringstream name;
    name << directory << '/' << prefix << '.' << std::setw(WidthOutput)
         << std::setfill('0') << step << ".bin";
    return name.str();
}

class Writer
{
public:
    explicit Writer(const std::string& path) : path_(path)
    {
        const std::uint32_t endian = 1;
        if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
            throw std::runtime_error("Binary output requires a little-endian host.");
        file_.open(path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
        if (!file_) throw std::runtime_error("Cannot open binary output: " + path_);
    }

    template <typename T> void Write(const T* data, std::size_t count)
    {
        const std::size_t bytes = Multiply(count, sizeof(T));
        if (bytes == 0) return;
        if (data == NULL) throw std::runtime_error("Null data in binary output: " + path_);
        if (bytes > static_cast<std::size_t>((std::numeric_limits<std::streamsize>::max)()))
            throw std::runtime_error("Binary output exceeds stream size: " + path_);
        file_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        if (!file_) throw std::runtime_error("Cannot write binary output: " + path_);
    }

    template <typename T> void Write(const std::vector<T>& data)
    {
        Write(data.data(), data.size());
    }

    template <typename T> void Scalar(const T& value) { Write(&value, 1); }

    void Close()
    {
        file_.close();
        if (!file_) throw std::runtime_error("Cannot close binary output: " + path_);
    }

private:
    Writer(const Writer&);
    Writer& operator=(const Writer&);
    std::string path_;
    std::ofstream file_;
};

inline void SaveContacts(int step, int particleCount, int branchLength,
                         const int* neighbors, const double3* forces, const double3* points)
{
    const std::size_t count = Multiply(Count(particleCount), Count(branchLength));
    Writer nei(FileName("./OutCnt", "Nei", step));
    nei.Scalar(particleCount);
    nei.Scalar(branchLength);
    nei.Write(neighbors, count);
    nei.Close();

    Writer fcn(FileName("./OutCnt", "Fcn", step));
    fcn.Scalar(particleCount);
    fcn.Scalar(branchLength);
    fcn.Write(forces, count);
    fcn.Close();

    Writer cpt(FileName("./OutCnt", "Cpt", step));
    cpt.Scalar(particleCount);
    cpt.Scalar(branchLength);
    cpt.Write(points, count);
    cpt.Close();
}

inline void SaveForceHistory(int step, int interval, const double3* force, const double3* torque)
{
    Count(interval);
    const int skip = 100;
    const int count = interval / skip + (interval % skip != 0 ? 1 : 0);
    if (count != 0 && (force == NULL || torque == NULL))
        throw std::runtime_error("Null force history data.");
    std::vector<int> samples(count);
    std::vector<double3> forces(count), torques(count);
    for (int i = 0; i < count; ++i)
    {
        samples[i] = i * skip;
        forces[i] = force[samples[i]];
        torques[i] = torque[samples[i]];
    }
    Writer file(FileName("./OutForce", "fc", step));
    file.Scalar(count);
    file.Write(samples);
    file.Write(forces);
    file.Write(torques);
    file.Close();
}

// All clumps are stored as field blocks; the optional intruder is ID -1 and
// occupies the first entry in every block. Its unavailable property is zero.
inline void SaveClumps(const std::string& path, int clumpCount,
                       const double3* pos, const double3* vel, const double4* quat,
                       const double3* angSpd, const double4* property,
                       const double3* intruderPos = NULL, const double3* intruderVel = NULL,
                       const double4* intruderQuat = NULL, const double3* intruderAngSpd = NULL)
{
    const std::size_t n = Count(clumpCount);
    const bool hasIntruder = intruderPos != NULL;
    if ((hasIntruder && (!intruderVel || !intruderQuat || !intruderAngSpd)) ||
        (!hasIntruder && (intruderVel || intruderQuat || intruderAngSpd)))
        throw std::runtime_error("Incomplete intruder output data.");
    if (hasIntruder && clumpCount == (std::numeric_limits<int>::max)())
        throw std::runtime_error("Binary clump count overflow.");
    const int count = clumpCount + (hasIntruder ? 1 : 0);
    std::vector<int> ids(Count(count));
    if (hasIntruder) ids[0] = -1;
    for (int i = 0; i < clumpCount; ++i) ids[static_cast<std::size_t>(i) + (hasIntruder ? 1 : 0)] = i;
    const double4 zeroProperty = { 0.0, 0.0, 0.0, 0.0 };
    Writer file(path);
    file.Scalar(count);
    file.Write(ids);
    if (hasIntruder) file.Write(intruderPos, 1);
    file.Write(pos, n);
    if (hasIntruder) file.Write(intruderVel, 1);
    file.Write(vel, n);
    if (hasIntruder) file.Write(intruderQuat, 1);
    file.Write(quat, n);
    if (hasIntruder) file.Write(intruderAngSpd, 1);
    file.Write(angSpd, n);
    if (hasIntruder) file.Scalar(zeroProperty);
    file.Write(property, n);
    file.Close();
}
}
