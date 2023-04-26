#ifndef SRMD_NCNN_VULKAN_SRMD_WRAPPED_H
#define SRMD_NCNN_VULKAN_SRMD_WRAPPED_H

#include "srmd.h"
#include "pybind11/include/pybind11/pybind11.h"
#include <locale>
#include <codecvt>
#include <utility>
#include <iostream>

// wrapper class of ncnn::Mat
class SRMDImage {
public:
    std::string d;
    int w;
    int h;
    int c;

    SRMDImage(std::string d, int w, int h, int c);

    void set_data(std::string data);

    pybind11::bytes get_data() const;
};

class SRMDWrapped : public SRMD {
public:
    SRMDWrapped(int gpuid, bool tta_mode);

    int get_tilesize() const;

    // SRMD parameters
    void set_parameters(int _noise, int _scale, int _prepadding, int _tilesize);

    int load(const std::string &parampath, const std::string &modelpath);

    int process(const SRMDImage &inimage, SRMDImage &outimage) const;

private:
    int gpuid;
};

int get_gpu_count();

void destroy_gpu_instance();

#endif // SRMD_NCNN_VULKAN_SRMD_WRAPPED_H
