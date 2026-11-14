#include "chimera/chimera_index.cuh"
#include "chimera/kernels.cuh"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using Chimera::check_cuda;
template<class T> struct Device {
    T* ptr = nullptr;
    explicit Device(size_t count) { check_cuda(cudaMalloc(&ptr, std::max(count,size_t(1))*sizeof(T))); }
    explicit Device(const std::vector<T>& data):Device(data.size()) {
        if(!data.empty())check_cuda(cudaMemcpy(ptr,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice));
    }
    ~Device() { if(ptr)(void)cudaFree(ptr); }
    Device(const Device&)=delete;
};
template<class Kernel>
void report_occupancy(const char* name, Kernel kernel, size_t dynamic_shared,
                      const cudaDeviceProp& device) {
    constexpr int threads = 256;
    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(&attributes, kernel));
    int blocks = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kernel, threads, dynamic_shared));
    const int warps_per_block = (threads + device.warpSize - 1) / device.warpSize;
    const int max_warps = device.maxThreadsPerMultiProcessor / device.warpSize;
    std::cout << "[OCCUPANCY] scope=kernel type=theoretical kernel=" << name
              << " threads_per_block=" << threads
              << " regs_per_thread=" << attributes.numRegs
              << " local_bytes_per_thread=" << attributes.localSizeBytes
              << " static_shared_bytes=" << attributes.sharedSizeBytes
              << " dynamic_shared_bytes=" << dynamic_shared
              << " max_active_blocks_per_sm=" << blocks
              << " max_active_warps_per_sm=" << blocks * warps_per_block
              << " hardware_max_warps_per_sm=" << max_warps
              << " theoretical_occupancy_pct=" << (100.0 * blocks * warps_per_block / max_warps)
              << '\n';
}

int main() try {
    constexpr size_t corpus=257, guard=11;
    constexpr float sentinel=-12345.25f;
    std::mt19937 rng(217);
    size_t cases=0, compared=0;
    cudaStream_t stream=nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    const size_t shared_bytes=(DOCUMENT_SCORING_LUT_SMEM_FLOATS+DOCUMENT_SCORING_LUT_TILE_Q)*sizeof(float);
    check_cuda(cudaFuncSetAttribute(Chimera::one_bit_token_score_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
    check_cuda(cudaFuncSetAttribute(Chimera::one_bit_token_score_compact_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
    int ordinal = 0;
    cudaDeviceProp device{};
    check_cuda(cudaGetDevice(&ordinal));
    check_cuda(cudaGetDeviceProperties(&device, ordinal));
    std::cout << "[OCCUPANCY] scope=device ordinal=" << ordinal
              << " sm_count=" << device.multiProcessorCount
              << " warp_size=" << device.warpSize
              << " regs_per_sm=" << device.regsPerMultiprocessor
              << " shared_bytes_per_sm=" << device.sharedMemPerMultiprocessor
              << " max_threads_per_sm=" << device.maxThreadsPerMultiProcessor
              << " max_blocks_per_sm=" << device.maxBlocksPerMultiProcessor
              << '\n';
    report_occupancy("original", Chimera::one_bit_token_score_kernel, shared_bytes, device);
    report_occupancy("compact", Chimera::one_bit_token_score_compact_kernel, shared_bytes, device);

    for(int fixture=0;fixture<8;++fixture) {
        std::vector<uint64_t> codes(corpus*2);
        for(auto& v:codes)v=(static_cast<uint64_t>(rng())<<32)|rng();
        if(fixture==1)std::fill(codes.begin(),codes.end(),uint64_t(0));
        if(fixture==2)std::fill(codes.begin(),codes.end(),~uint64_t(0));
        if(fixture==3)std::fill(codes.begin(),codes.end(),uint64_t(0x0123456789abcdef));
        std::vector<float> lut(32*LUT_ENTRIES_PER_QUERY),cb(32),factors(corpus);
        for(auto& v:lut)v=float(int(rng()%2049)-1024)/512.0f;
        for(auto& v:cb)v=float(int(rng()%257)-128)/256.0f;
        for(auto& v:factors)v=float(int(rng()%1025)-512)/1024.0f;
        if(fixture==4)for(auto& v:lut)v=-std::abs(v);
        if(fixture==5)std::fill(factors.begin(),factors.end(),0.0f);
        if(fixture==6)std::fill(lut.begin(),lut.end(),0.0f);
        if(fixture==7)for(size_t i=0;i<lut.size();++i)lut[i]=(i&1)?-0.0f:0.0f;
        Device<uint64_t> dc(codes);Device<float> dl(lut),db(cb),df(factors);
        for(size_t n:{0,1,31,32,33,127,128,255,256,257,513,1023})
        for(int grid_mode=0;grid_mode<2;++grid_mode) {
            const size_t offset=fixture%2?17:0,total=n+offset+19;
            std::vector<uint32_t> positions(std::max(n,size_t(1)));
            for(size_t t=0;t<positions.size();++t)
                positions[t]=fixture==1?uint32_t(7):uint32_t((t*73+19)%corpus);
            std::vector<float> initial(guard+32*total+guard,sentinel);
            Device<uint32_t> dp(positions);Device<float> control(initial),compact(initial);
            const int blocks=grid_mode?std::max(size_t(1),(n+255)/256):1;
            Chimera::one_bit_token_score_kernel<<<blocks,256,shared_bytes,stream>>>(
                dl.ptr,reinterpret_cast<const char*>(dc.ptr),df.ptr,db.ptr,dp.ptr,
                control.ptr+guard+offset,total,n);
            check_cuda(cudaGetLastError());
            Chimera::one_bit_token_score_compact_kernel<<<blocks,256,shared_bytes,stream>>>(
                dl.ptr,reinterpret_cast<const char*>(dc.ptr),df.ptr,db.ptr,dp.ptr,
                compact.ptr+guard+offset,total,n);
            check_cuda(cudaGetLastError());
            check_cuda(cudaStreamSynchronize(stream));
            std::vector<float> a(initial.size()),b(initial.size());
            check_cuda(cudaMemcpy(a.data(),control.ptr,a.size()*sizeof(float),cudaMemcpyDeviceToHost));
            check_cuda(cudaMemcpy(b.data(),compact.ptr,b.size()*sizeof(float),cudaMemcpyDeviceToHost));
            if(std::memcmp(a.data(),b.data(),a.size()*sizeof(float))) {
                for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],sizeof(float))) {
                    std::cerr<<"fixture="<<fixture<<" count="<<n<<" gridmode="<<grid_mode<<" idx="<<i<<'\n';break;
                }
                throw std::runtime_error("Token score FP32 matrix differs from original");
            }
            std::vector<bool> written(initial.size(),false);
            for(size_t q=0;q<32;++q)for(size_t t=0;t<n;++t)written[guard+offset+q*total+t]=true;
            for(size_t i=0;i<a.size();++i)if(!written[i]&&std::memcmp(&a[i],&sentinel,sizeof(float)))
                throw std::runtime_error("Token score write escaped its output slice");
            compared+=32*n;++cases;
        }
    }
    check_cuda(cudaStreamDestroy(stream));
    std::cout<<"PASS: "<<cases<<" original/compact GPU token matrices bitwise equal; "<<compared
             <<" scores; count0/partialCTA/gridstride/offsets/repeated+reorderedpositions/LUTs/factors/negative/zero/ties/guards\n";
    return 0;
} catch(const std::exception& e) {
    std::cerr<<"[FAILED] "<<e.what()<<'\n';return 1;
}
