#include "chimera/cpu_onebit_fused.hpp"
#include <cfloat>
#include <cstddef>
#include <cstring>
#include <cmath>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace Chimera {
void build_cpu_onebit_lut(const float* query,CpuOnebitLut& lut) {
    for(int c=0;c<32;++c) for(int nib=0;nib<16;++nib) for(int q=0;q<32;++q) {
        float sum=0.0f;
        for(int bit=0;bit<4;++bit) if((nib>>bit)&1) sum=sum+query[q*128+c*4+bit];
        lut.values[(c*16+nib)*32+q]=sum;
    }
}
float score_cpu_onebit_document(const uint8_t* codes,const float* factors,
    const uint32_t* inverse,const int* ptrs,int doc,
    const CpuOnebitLut& lut,const float* cb) {
    float maxima[32];
#if defined(__aarch64__)
    float32x4_t acc[8];for(int j=0;j<8;++j)acc[j]=vdupq_n_f32(-FLT_MAX);
#else
    for(float& x:maxima)x=-FLT_MAX;
#endif
    for(int token=ptrs[doc];token<ptrs[doc+1];++token) {
        const uint32_t position=inverse[token];
        uint64_t words[2];std::memcpy(words,codes+size_t(position)*16,16);
#if defined(__aarch64__)
        float32x4_t ip[8];for(int j=0;j<8;++j)ip[j]=vdupq_n_f32(0.0f);
        for(int c=0;c<32;++c) {
            const unsigned nib=(words[c/16]>>(4*(c%16)))&15;
            const float* row=lut.values.data()+(c*16+nib)*32;
            for(int j=0;j<8;++j)ip[j]=vaddq_f32(ip[j],vld1q_f32(row+j*4));
        }
        const auto factor=vdupq_n_f32(factors[position]);
        for(int j=0;j<8;++j) {
            const auto score=vmulq_f32(vsubq_f32(ip[j],vld1q_f32(cb+j*4)),factor);
            acc[j]=vmaxnmq_f32(acc[j],score);
        }
#else
        for(int q=0;q<32;++q) {
            float ip=0.0f;
            for(int c=0;c<32;++c) {
                const unsigned nib=(words[c/16]>>(4*(c%16)))&15;
                ip=ip+lut.values[(c*16+nib)*32+q];
            }
            maxima[q]=std::fmax(maxima[q],(ip-cb[q])*factors[position]);
        }
#endif
    }
#if defined(__aarch64__)
    for(int j=0;j<8;++j)vst1q_f32(maxima+j*4,acc[j]);
#endif
    // Original 128-thread document reduction, including 96 zero lanes.
    float lanes[128]{};
    for(int q=0;q<32;++q)lanes[q]=0.0f+maxima[q];
    for(int stride=64;stride;stride>>=1)
        for(int q=0;q<stride;++q)lanes[q]=lanes[q]+lanes[q+stride];
    return lanes[0];
}
}
