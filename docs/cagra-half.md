# CAGRA half 路由实验

本候选只把 CAGRA 的 dataset/query 输入改为 FP16。默认 `SearchOptions::cagra_half=false`；`--cagra-half` 只在 `--stage custom` 启用，要求原 CAGRA device graph 和 `--uma-query-buffers`。原 graph、行号、搜索参数、FP32 查询缓冲与最终 fullbit 评分保留。

```text
原float index ──保留dataset及graph owner──> 原uint32 graph
      │                                         │ 只借指针
      └─初始化GPU RN转换─> half dataset ──> half index

每lane CPU rotated FP32 query ──CPU RN转换──> 独立8192B ATS half query
                │                                 │
                └─原LUT/评分/fullbit                └─原参数CAGRA search
```

初始化阶段：

1. 验证原 dense D128 / L2Expanded dataset、graph shape，拒绝 remap、磁盘 FD 和 CAGRA ATS placement 组合。
2. 分配一份 half dataset，GPU 逐元素检查有限范围 `[-65504,65504]` 并 `__float2half_rn`。同 stream 的 domain flag 回读及同步完成后才构造、发布 half index。
3. 用 `non_owning_dataset<half,int64_t>` 绑定新 dataset，用 `update_graph` 绑定原 float index 的同一 graph。检查 dataset/graph 指针、shape和借用状态，打印一次 `[CAGRA_HALF]`。
4. 每个 Workspace 单独分配 256 B 对齐、8192 B 的 ATS query buffer。CPU first touch，`h_` 为唯一 owner，`d_` 是别名；打印 `[CAGRA_HALF_QUERY]`。Workspace 禁止复制。

每 query 在原 CPU rotation 后、第一次 GPU 提交前，检查并 RN 转换全部4096元素到该 lane 的 half buffer；转换函数不调用 CUDA API。原 float query 保留给 LUT、候选评分与 fullbit。原事件和完成等待不变，无新增 query host synchronization。

half view 先销毁，再释放 half dataset，最后销毁原 float graph/data owner；reload 使用同一顺序。原 FP32 `save()` 路径不变。owner 初始化完成后，batch workers只借同一只读 PG/index，各自持有 query/RAFT resource/工作缓冲。

LoTTE 当前 `2097152×128 / degree64` 下，half dataset 为536870912 B，原 FP32 dataset为1073741824 B，原 graph为536870912 B。搜索活跃数组由1.5 GiB降至1 GiB；保留原 owner 后总路由数组驻留2 GiB，另有8192 B/lane query。没有 graph copy、graph build 或自动释放原 float dataset。

接口及输出：

- PG/IVF 四个 `search_centroids` 入口在末尾新增默认空的 `const __half*` query，保持旧调用源码兼容；half index与half query必须同时启用。hash/width/beam/team/block等原字段原样传递。
- `chimera_index::cagra_route_storage_stats()` 返回实际 dtype状态、dataset/graph alias、保留 owner、实际行/维度/degree和字节数；benchmark JSON据此记录。`cagra_route_math=fp16-input/cuvs-native`，不推断库内累加精度或默认team。
- `onebit_score_math` 仍按原布局记录，`final_fullbit_math=original-fp32`。本实验不启用 TF32，也不更改原 `kernels.cu`。

编译及 GPU 验证由持 EX 锁的父任务执行；本源码提交只做静态审查。现有 benchmark 使用同一组 endpoint 参数，配对半精度 arm 仅添加：

```sh
--stage custom --uma-query-buffers --cagra-half
```

先32 query的现有 native/batch ID及scorebits资格检查，加 `--verify-full-bit` 核同一候选集的原FP32 fullbit；再全2931条GT，固定B32/W4和原nprobe/beam/hash/width等，要求实际R10≥0.90。FP16路由可改变候选集，原/新route labels不要求逐位相同。全量质量通过后才进入正式性能比较；CPU转换计入端到端时间，load转换另计。这是独立精度/工作集候选。
