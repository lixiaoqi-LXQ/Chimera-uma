// Standalone CPU-only mathematical fixture; not registered in production targets.
#include "chimera/cpu_onebit_fused.hpp"
#include <array>
#include <vector>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iostream>
int main() {
    constexpr int N=233,D=13;
    std::array<int,D+1> ptrs{};
    // Includes an empty document; total is exactly N.
    for(int d=1;d<D;++d)ptrs[d]=d==1?0:(d-1)*17;
    ptrs[D]=N;
    std::vector<uint8_t> doc_codes(N*16),cluster_codes(N*16);
    std::vector<float> doc_factors(N),cluster_factors(N);
    std::vector<uint32_t> inverse(N);
    uint32_t random=71;
    for(auto& x:doc_codes){random=1664525u*random+1013904223u;x=uint8_t(random>>17);}
    for(int t=0;t<N;++t){inverse[t]=(t*37)%N;doc_factors[t]=float(t%127+1)/128;
        std::memcpy(cluster_codes.data()+inverse[t]*16,doc_codes.data()+t*16,16);
        cluster_factors[inverse[t]]=doc_factors[t];}
    float query[32*128],cb[32];
    for(float& x:query){random=1664525u*random+1013904223u;
        uint32_t bits=((112u+((random>>24)&15))<<23)|(random&0x807fffffu);std::memcpy(&x,&bits,4);}
    for(int q=0;q<32;++q)cb[q]=float(q-17)/32;
    Chimera::CpuOnebitLut lut;Chimera::build_cpu_onebit_lut(query,lut);
    for(int d=0;d<D;++d){float lanes[128]{};
        for(int q=0;q<32;++q){float maximum=-FLT_MAX;
            for(int t=ptrs[d];t<ptrs[d+1];++t){float ip=0;
                for(int c=0;c<32;++c){unsigned nib=(doc_codes[t*16+c/2]>>(4*(c%2)))&15;float entry=0;
                    for(int b=0;b<4;++b)if((nib>>b)&1)entry=entry+query[q*128+c*4+b];ip=ip+entry;}
                maximum=std::fmax(maximum,(ip-cb[q])*doc_factors[t]);}
            lanes[q]=0.0f+maximum;}
        for(int stride=64;stride;stride>>=1)for(int q=0;q<stride;++q)lanes[q]=lanes[q]+lanes[q+stride];
        float got=Chimera::score_cpu_onebit_document(cluster_codes.data(),cluster_factors.data(),
            inverse.data(),ptrs.data(),d,lut,cb);
        if(std::memcmp(&got,lanes,4))throw std::runtime_error("inverse fixture score bits differ");
    }
    std::cout<<"{\"passed\":true,\"docs\":13,\"tokens\":233,\"GPU_used\":false}\n";
}
