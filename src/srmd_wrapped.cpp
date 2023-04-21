#include "srmd_wrapped.h"

// Image Data Structure
Image::Image(std::string d, int w, int h, int c) {
    this->d = std::move(d);
    this->w = w;
    this->h = h;
    this->c = c;
}

void Image::set_data(std::string data) {
    this->d = std::move(data);
}

pybind11::bytes Image::get_data() const {
    return pybind11::bytes(this->d);
}

// SRMDWrapped
SRMDWrapped::SRMDWrapped(int gpuid, bool tta_mode)
        : SRMD(gpuid, tta_mode) {
    this->gpuid = gpuid;
}

int SRMDWrapped::get_tilesize() const {
    int tilesize = 0;

    uint32_t heap_budget = ncnn::get_gpu_device(this->gpuid)->get_heap_budget();

    if (heap_budget > 2600)
        tilesize = 400;
    else if (heap_budget > 740)
        tilesize = 200;
    else if (heap_budget > 250)
        tilesize = 100;
    else
        tilesize = 32;

    return tilesize;
}


void SRMDWrapped::set_parameters(int _noise, int _scale, int _prepadding, int _tilesize) {
    SRMD::noise = _noise;
    SRMD::scale = _scale;
    SRMD::tilesize = _tilesize ? _tilesize : SRMDWrapped::get_tilesize();
    SRMD::prepadding = _prepadding;
}

int SRMDWrapped::load(const std::string &parampath,
                      const std::string &modelpath) {
#if _WIN32
    // convert string to wstring
    auto to_wide_string = [&](const std::string& input) {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        return converter.from_bytes(input);
    };
    return SRMD::load(to_wide_string(parampath), to_wide_string(modelpath));
#else
    return SRMD::load(parampath, modelpath);
#endif
}

int SRMDWrapped::process(const Image &inimage, Image &outimage) const {
    int c = inimage.c;
    ncnn::Mat inimagemat =
            ncnn::Mat(inimage.w, inimage.h, (void *) inimage.d.data(), (size_t) c, c);
    ncnn::Mat outimagemat =
            ncnn::Mat(outimage.w, outimage.h, (void *) outimage.d.data(), (size_t) c, c);
    return SRMD::process(inimagemat, outimagemat);
}

int get_gpu_count() { return ncnn::get_gpu_count(); }

void destroy_gpu_instance() { ncnn::destroy_gpu_instance(); }

PYBIND11_MODULE(srmd_ncnn_vulkan_wrapper, m) {
    pybind11::class_<SRMDWrapped>(m, "SRMDWrapped")
            .def(pybind11::init<int, bool>())
            .def("load", &SRMDWrapped::load)
            .def("process", &SRMDWrapped::process)
            .def("set_parameters", &SRMDWrapped::set_parameters);

    pybind11::class_<Image>(m, "Image")
            .def(pybind11::init<std::string, int, int, int>())
            .def("get_data", &Image::get_data)
            .def("set_data", &Image::set_data);

    m.def("get_gpu_count", &get_gpu_count);

    m.def("destroy_gpu_instance", &destroy_gpu_instance);
}
