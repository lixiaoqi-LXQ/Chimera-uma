#pragma once
#include "tensor_scalar_half_oracle.cuh"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Chimera { namespace tensor_half_oracle_detail {
// Called only for explicit Tensor QA, after the existing completion event.
// Oracle buffers belong to this lane, and IDs map all three score arrays to
// the same refined candidates. FP32 score differences are diagnostic only.
inline std::string verify_document_results(
    const float* tensor_query_maxima,
    const double* half_reference_maxima,
    const double* half_error_bounds,
    const float* tensor_scores,
    const float* original_scores,
    const size_t* offsets,
    const int* document_ids,
    size_t docs) {
    double maximum_query_ratio=0,maximum_doc_ratio=0,maximum_absolute=0,maximum_relative=0;
    size_t maximum_absolute_doc=0,maximum_relative_doc=0,empty_docs=0,finite_docs=0;
    for(size_t doc=0;doc<docs;++doc) {
        const bool empty=offsets[doc]==offsets[doc+1];
        double reference_sum=0;
        for(size_t q=0;q<32;++q) {
            const size_t index=doc*32+q;
            const double actual=tensor_query_maxima[index];
            const double reference=half_reference_maxima[index];
            const double bound=half_error_bounds[index];
            if(!std::isfinite(actual)||!std::isfinite(reference)||!std::isfinite(bound)||bound<0)
                throw std::runtime_error("Nonfinite Tensor querymax/scalar-half oracle or invalid bound");
            if(empty) {
                if(actual!=-double(FLT_MAX)||reference!=-double(FLT_MAX)||bound!=0)
                    throw std::runtime_error("Tensor empty-document querymax sentinel differs");
            } else {
                const double error=std::abs(actual-reference);
                if(error>bound)
                    throw std::runtime_error("Tensor querymax exceeds scalar-half roundoff bound at candidate "+
                                             std::to_string(doc)+" query "+std::to_string(q));
                if(bound>0)maximum_query_ratio=std::max(maximum_query_ratio,error/bound);
                reference_sum+=reference;
            }
        }
        if(empty) {
            ++empty_docs;
            if(tensor_scores[doc]!=-std::numeric_limits<float>::infinity()||
               original_scores[doc]!=tensor_scores[doc])
                throw std::runtime_error("Tensor empty-document score sentinel differs");
            continue;
        }
        if(!std::isfinite(tensor_scores[doc])||!std::isfinite(original_scores[doc]))
            throw std::runtime_error("Nonfinite nonempty Tensor/original document score");
        const double bound=finalsum_error_bound(half_reference_maxima+doc*32,half_error_bounds+doc*32);
        const double doc_error=std::abs(double(tensor_scores[doc])-reference_sum);
        if(!std::isfinite(bound)||bound<0||doc_error>bound)
            throw std::runtime_error("Tensor docscore exceeds scalar-half roundoff bound at candidate "+std::to_string(doc));
        if(bound>0)maximum_doc_ratio=std::max(maximum_doc_ratio,doc_error/bound);
        if(finite_docs++==0){maximum_absolute_doc=doc;maximum_relative_doc=doc;}
        const double absolute=std::abs(double(tensor_scores[doc])-original_scores[doc]);
        const double relative=absolute/std::max(std::abs(double(original_scores[doc])),1e-12);
        if(absolute>maximum_absolute){maximum_absolute=absolute;maximum_absolute_doc=doc;}
        if(relative>maximum_relative){maximum_relative=relative;maximum_relative_doc=doc;}
    }
    auto top=[&](const float* scores){
        std::vector<size_t> order(docs);std::iota(order.begin(),order.end(),size_t(0));
        std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return scores[a]!=scores[b]?scores[a]>scores[b]:a<b;});
        order.resize(std::min(size_t(20),docs));return order;
    };
    const auto reference_top=top(original_scores),actual_top=top(tensor_scores);
    size_t set_matches=0,position_matches=0;
    for(size_t i=0;i<reference_top.size();++i){
        if(reference_top[i]==actual_top[i])++position_matches;
        if(std::find(actual_top.begin(),actual_top.end(),reference_top[i])!=actual_top.end())++set_matches;
    }
    const double count=double(reference_top.size());
    std::ostringstream out;out.precision(12);
    out<<"[TENSOR_HALF_ORACLE_VERIFY] candidates="<<docs<<" querymax_checked="<<docs*32
       <<" querymax_bound=passed docscore_bound=passed empty_documents="<<empty_docs
       <<" max_querymax_error_to_bound="<<maximum_query_ratio<<" max_docscore_error_to_bound="<<maximum_doc_ratio
       <<" onebit_input=fp16 accumulation=fp32 final_fullbit=original_fp32\n";
    out<<"[TENSOR_APPROX_DIAGNOSTIC] candidates="<<docs<<" max_abs_error="<<maximum_absolute
       <<" max_abs_error_doc="<<(finite_docs?document_ids[maximum_absolute_doc]:-1)
       <<" max_rel_error="<<maximum_relative<<" max_rel_error_doc="<<(finite_docs?document_ids[maximum_relative_doc]:-1)
       <<" relative_denominator_floor=1e-12 topk="<<reference_top.size()
       <<" top20_set_overlap="<<(count?set_matches/count:1)
       <<" top20_position_match="<<(count?position_matches/count:1)
       <<" top20_set_exact="<<(set_matches==reference_top.size()?"true":"false")
       <<" top20_order_exact="<<(position_matches==reference_top.size()?"true":"false")<<'\n';
    return out.str();
}
} }  // namespace Chimera::tensor_half_oracle_detail
