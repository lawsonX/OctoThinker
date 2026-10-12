# OctoThinker 复现指南（本机版）

本指南面向这台机器（单节点、6 张 NVIDIA A100-PCIE-40GB、64 核 CPU、没有 Slurm 调度系统），说明如何从头到尾复现论文《OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling》（arXiv:2506.20512）的核心实验。论文要回答的问题是：什么样的基座模型适合用强化学习（RL，让模型通过试错和奖励信号来学习）继续训练。论文的答案是：先在中训练阶段（mid-training，即在预训练之后、正式使用之前，再用特定数据做一轮续训）注入高质量数学数据和长推理链数据，可以让原本不适合 RL 的 Llama 模型变得适合。

注意：任何时候开始跑任务前，先用 `nvidia-smi` 看一下有几张卡空闲。这台机器上常有两个 GPU 被一个无关的 vLLM 服务占用，所有脚本都可以通过 `CUDA_VISIBLE_DEVICES` 指定用哪几张卡，不要把 6 张卡全当作可用。

## 一、论文核心实验一览

论文的实验分四部分：

1. **第 2 节，Zero-RL 动力学对比**：直接对 Llama-3.2-3B-Base 和 Qwen2.5-3B-Base 做 R1-Zero 式 RL。Qwen 越练越好，Llama 会提前报答案然后陷入重复。这是全文的动机实验。
2. **第 3 节，受控的 20B token 中训练消融**：固定 3B 模型和 20B token 预算，逐个改变中训练数据（数学语料、问答数据、长推理链数据、指令数据）和数据格式，再对每个续训出的模型做同样的 RL，观察哪种数据让 RL 收益最大。
3. **第 4 节，两阶段中训练（Stable-then-Decay）**：第一阶段用恒定学习率在 MegaMath-Web-Pro-Max 上续训 200B token（stable 阶段）；第二阶段分三个分支（Long / Short / Hybrid，即长推理链、短推理链、混合）各训 20B token 并衰减学习率（decay 阶段）。三个分支对应三类推理行为。
4. **第 5 节，OctoThinker-Zero 系列**：把所有 OctoThinker base 模型（1B、3B，各分支）再过一遍第 2 节的 RL 流程，得到 *-Zero 模型，验证它们追平 Qwen 同规模模型的 RL 水平。

## 二、资产地图

```
OctoThinker/                        # 本仓库
├── models/                         # 基座和已发布检查点（download_models.sh 下载）
├── midtraining/
│   ├── nanotron/                   # Nanotron 中训练框架（原版快照，勿改）
│   ├── nanotron-runner/            # 原始的 OctoThinker 训练配置（空壳子模块，已丢失）
│   ├── run/                        # 【本复现新增】数据准备脚本 + 训练配置 + 启动器
│   ├── env_setup.sh                # 【本复现新增】nanotron conda 环境安装
│   └── data/                       # 【运行时生成】jsonl 中间产物 + tokenized 数据
├── rl/                             # vendored verl 0.2.0.dev + GRPO 脚本 + MATH8K 数据
│   ├── env_setup.sh                # 【本复现新增】verl conda 环境安装
│   └── scripts/run_grpo_3b_local.sh# 【本复现新增】单机 6 卡 GRPO 启动脚本
├── eval/                           # 13 个数学评测基准 + vLLM 推理评测脚本
│   ├── env_setup.sh                # 【本复现新增】matheval conda 环境安装
│   └── scripts/run_{fewshot_base,zeroshot_rl}.sh  # 【本复现新增】评测启动脚本
├── plot/                           # 论文图表的绘图脚本和数据
└── REPRODUCTION.md                 # 本文件

/hdd2/xiaolirui/octoThinker-Datasets/   # 仓库外：8 个中训练数据集（论文同款，正在上传）
    FineMath-4plus  MegaMath-QA  MegaMath-Web-Pro  MegaMath-Web-Pro-Max
    OpenR1-Math-220K  TULU3-sft  UltraChat-220K  WildChat-1M
```

## 三、环境搭建顺序

总共需要三个 conda 环境。按下面的顺序做，前两步只依赖网络：

```bash
# 1. 模型（约 57GB，见 models/README.md；脚本自动适配本机网络：基座走 ModelScope、OctoThinker 检查点走 hf-mirror）
python3 models/download_models_v2.py

# 2. 三个 conda 环境（python 3.10，装在 /hdd2/xiaolirui/conda_envs/ 下）
bash rl/env_setup.sh          # /hdd2/xiaolirui/conda_envs/verl
bash eval/env_setup.sh        # /hdd2/xiaolirui/conda_envs/matheval
bash midtraining/env_setup.sh # /hdd2/xiaolirui/conda_envs/nanotron
```

三个环境可以并行装。**本机网络质量差时的快速路径**（已经验证过）：先装 matheval，另外两个用克隆方式秒建（复用已装好的 torch/cu12 全家桶，只补各自的增量包）：

```bash
bash eval/env_setup.sh                 # 先把这个装完（或网络好时）
bash rl/env_setup_from_clone.sh        # conda clone + RL 增量
bash midtraining/env_setup_from_clone.sh
```

环境装好后做三个冒烟测试：

```bash
conda activate /hdd2/xiaolirui/conda_envs/verl
cd rl && python -c "import verl.trainer.main_ppo" && cd ..   # verl 是源码内置的，不用 pip 安装

conda activate /hdd2/xiaolirui/conda_envs/matheval
python -c "import vllm, math_verify; print('matheval ok')"

python -c "import pyarrow.parquet as pq; print(pq.read_table('rl/data/math.8k/train.parquet').num_rows, 'train rows')"  # 应为 8407
```

本仓库搭建时已经跑通过的验证（2026-10-10）：评测链路用 Llama-3.2-1B-Base 在单卡上跑通 MATH500 few-shot，得 5.0%，与论文表 6 的 4.60 基本一致；mid-training 配置与 nanotron 源码逐个核对过；RL 脚本 46 个 hydra 参数全部对照 vendored verl 验证。

## 四、数据准备

- **RL 数据**：`rl/data/math.8k/` 就是 SimpleRL-Zoo 的 MATH8K（8407 条训练、272 条验证，其中最后 18 条是填充行）。如果要干净的验证指标，先跑一次 `python rl/scripts/clean_val_data.py`，会生成 254 条的 `test.clean.parquet`。
- **中训练数据**：论文同款 8 个数据集正在上传到 `/hdd2/xiaolirui/octoThinker-Datasets/`（共 400GB+，注意确认上传完成后再跑预处理）。预处理管线（`midtraining/run/`）分三步：
  1. `python midtraining/run/prepare_data.py --datasets-root /hdd2/xiaolirui/octoThinker-Datasets --output-root midtraining/data/jsonl` —— 把每个数据集转成统一的 `{"text": ...}` jsonl，并按论文第 3 节规则格式化（OpenR1 加 `<think></think>` 包裹、指令数据格式化为 `User:/Assistant:` 多轮、WildChat 只保留 GPT-4 对话）。脚本会自动探测 schema，某数据集没传完时会明确报错并继续处理其它数据集。
  2. `python midtraining/run/shard_data.py --n-shards 32` —— 把每个数据集的 `data.jsonl` 切成 32 份 `data.shard-*.jsonl` 并删除原文件。不可省略：datatrove 按文件分片，不切的话整个数据集只由一个 tokenize 任务处理，其余任务全部空转。
  3. `bash midtraining/run/tokenize_data.sh --n-tasks 32` —— 调用 nanotron 的 `tools/preprocess_data.py` 用 Llama-3 分词器把 jsonl 切成 `.ds` 格式（加权混合直接在训练配置里指定目录和权重）。`--n-tasks` 不应超过切分份数；换任务数或重切后要先删掉 `tokenized/<Name>/` 和 `tokenized/logs/datatrove_<Name>/` 再跑，否则旧的完成标记会让部分分片被跳过。
- **评测数据**：`eval/datasets/` 已齐备 13 个基准，包括 `math500/test.jsonl`（500 条）。

## 五、中训练启动命令

`midtraining/run/configs/` 下已按论文重建好全部训练配置（原始配置所在的 `nanotron-runner` 子模块是空的，见“偏差与风险”）：

| 配置 | 对应论文实验 |
|---|---|
| `ablation_3b_20B_cosine.yaml` | 第 3 节 20B token 受控消融（默认 Web-Pro:QA = 9:1，注释里有 89:10:1 和其它混合） |
| `scale_3b_100B_pro_max.yaml` | 第 3.5 节 Pro-Max 100B 扩规模（在 20B/70B/100B 处存检查点） |
| `two_stage_stable_3b.yaml` | 第 4.1 节 stable 阶段（200B token、恒定 LR、表 2 混合比例） |
| `two_stage_decay_3b_{long,short,hybrid}.yaml` | 第 4.2 节 decay 阶段三个分支（5000 步、cosine 衰减、表 5 混合比例） |
| `smoke_3b_tiny.yaml` | 20 步冒烟测试，验证管线 |
| `llama-3.2-1b.yaml` / `llama-3.2-3b.yaml` / `llama-3.1-8b.yaml` | 三个模型尺寸的骨架配置（并行策略已按 6 卡调好） |

运行流程（详见 `midtraining/run/README.md`）：

```bash
conda activate /hdd2/xiaolirui/conda_envs/nanotron

# 0) 冒烟测试（20 步，先确认环境、数据、并行策略都通）
bash midtraining/run/run_midtraining.sh midtraining/run/configs/smoke_3b_tiny.yaml

# 1) 把 HF 格式的 Llama base 转成 nanotron 格式（一次性，约几分钟）
bash midtraining/run/convert_hf_to_nanotron.sh models/Llama-3.2-3B-Base models/Llama-3.2-3B-Base-nanotron

# 2) 启动 20B 消融（配置里 init_method 指向第 1 步转好的目录）
bash midtraining/run/run_midtraining.sh midtraining/run/configs/ablation_3b_20B_cosine.yaml
```

说明：本机 6 卡时全局 batch 取 528 条序列（micro_batch 2 × accum 44 × dp 6，约 4.33M token，比论文的 512 多 3%，20B token 对应 4624 步，已写进配置注释）。这是 40GB 显存的适配结果：3B 模型必须开 ZeRO-1（`zero_stage: 1`）+ 关闭 fp32 梯度累积（`accumulate_grad_in_fp32: false`）+ 激活重算（`recompute_layer: true`）+ micro_batch 降到 2（128256 词表的 fp32 交叉熵 logits 在 mbs 8 时会独占 15GB+ 显存），全套设置已应用到 `midtraining/run/configs/` 下所有 3B/1B/8B 配置并经过冒烟验证；配置头注释里有完整的显存账本。训练中断后用 `RESUME=1 bash midtraining/run/run_midtraining.sh <同一个配置>` 续训。decay 阶段（第 4 节）从 stable 阶段的最终检查点出发，三个分支各跑 5000 步（3B 学习率 2e-5 → 2e-6）。训完用 `bash midtraining/run/convert_nanotron_to_hf.sh <checkpoint> <输出目录> <tokenizer路径>` 转成 HF 格式，交给 RL 和评测（注意：转换脚本自带的生成验证需要空闲 GPU，显存被别的任务占满时验证步骤会 OOM，但转换产物已完整落盘）。

注意：`run_midtraining.sh` 按配置里 dp×tp×pp 决定用几张卡，必须和 `CUDA_VISIBLE_DEVICES` 可见卡数一致。比如只有 3、5 两张卡空闲时，先复制一份配置把 parallelism 改成 dp: 2（micro_batch_size 可保持 8），再 `CUDA_VISIBLE_DEVICES=3,5 bash midtraining/run/run_midtraining.sh <副本>.yaml`；速度约为 6 卡的 1/3。

decay 阶段的完整数据配方还需要 DCLM-Baseline、OpenMathInstruct-2、NuminaMath-1.5、AM-DeepSeek-Distilled-40M 等配置注释里列出的数据集（不在上传的 8 个数据集之内）；如果只想用已上传的 8 个数据集，可以把这些权重临时并回 MegaMath-Web-Pro / OpenR1 / MegaMath-QA。第 3 节的核心消融**不需要**这些额外数据集。

## 六、RL 启动命令（每个分支）

RL 用 vendored verl 跑 GRPO（组相对策略优化，PPO 的一种简化变体）。仓库里的 `rl/scripts/run_grpo_3b_local.sh` 已经按本机 6 卡改好，默认配置对齐论文第 2.1 节：每条提示采样 16 条回答、学习率 1e-6、温度 1.0、KL 系数 0，默认约 13 个 epoch（约 416 步，对应论文图中的 400 步）。

```bash
conda activate /hdd2/xiaolirui/conda_envs/verl

# 基线 1：Llama-3.2-3B-Base（论文图 1/2 的 Zero-RL 基线，短分支固定 4096）
MODEL_PATH=models/Llama-3.2-3B-Base BRANCH=short \
bash rl/scripts/run_grpo_3b_local.sh

# 基线 2：Qwen2.5-3B-Base（对比基线）
MODEL_PATH=models/Qwen2.5-3B-Base BRANCH=short \
bash rl/scripts/run_grpo_3b_local.sh

# 核心实验：OctoThinker Long 分支（前 200 步限长 2048，200-320 步 4096，之后 8192）
MODEL_PATH=models/OctoThinker-3B-Long-Base BRANCH=long \
bash rl/scripts/run_grpo_3b_local.sh

# Hybrid / Short 分支同理：BRANCH=hybrid 或 BRANCH=short

# 用自己中训练出的检查点：
MODEL_PATH=outputs/midtrain/3b_long/hf/final BRANCH=long \
bash rl/scripts/run_grpo_3b_local.sh
```

卡不全空闲时这样限制（下面例子自动识别为 2 卡）：

```bash
CUDA_VISIBLE_DEVICES=3,5 MODEL_PATH=models/OctoThinker-3B-Long-Base bash rl/scripts/run_grpo_3b_local.sh
```

运行须知（本机踩过的坑，脚本已内置处理）：
- **不要同时跑两个 RL 任务**：每个脚本启动时会 `ray stop --force` 清掉本机所有 ray 进程（包括别人的任务）。两人协作时先互相确认。
- 系统 redis 长期占用 6379 端口，脚本的 Ray 默认用 `RAY_PORT` 环境变量（缺省 26379+ 随机偏移），不用 6379。
- 训练进程打印的 "Flash Attention 2.0 only supports torch.float16 and torch.bfloat16... current dype is torch.float32" 是 FSDP mixed-precision 加载的正常警告，不影响训练。
- 显存占用：rollout 阶段 vLLM 按 `gpu_memory_utilization=0.75`（约 30GB/卡）占用，训练阶段 FSDP 把参数/优化器 offload 到内存（64GB+ 内存余量即可），两者分时复用同一块 GPU。

监控方式：控制台日志每步打印 reward 和回答长度（没配 wandb key 时只打控制台）；每步的原始采样在 `outputs/rl/<branch>/samples/<run>/train/step_*.json`；检查点在 `outputs/rl/<branch>/ckpts/<run>`，每 10 步存一次，崩溃后自动续训。

## 六点五、训练监控看板（loss / ppl / grad norm / MFU / lr）

`monitor/` 目录提供基于 TensorBoard 的监控，无需改动训练代码（解析训练日志）：

```bash
# 方式一：启动训练时自动挂监控（推荐）
MONITOR=1 bash midtraining/run/run_midtraining.sh midtraining/run/configs/ablation_3b_20B_cosine.yaml
MONITOR=1 MODEL_PATH=models/OctoThinker-3B-Long-Base bash rl/scripts/run_grpo_3b_local.sh

# 方式二：对已运行的任务手动挂载
bash monitor/run_monitor.sh outputs/midtraining/logs/<训练日志>.log <run名称>

# 打开看板（默认 6006 端口，浏览器访问 http://<本机IP>:6006）
bash monitor/serve.sh
```

中训练可看：lm_loss、ppl(=exp(loss))、grad_norm、实际 lr、tokens/s（总量与每卡）、MFU（=model_tflops/312，A100 bf16 峰值）、CUDA 显存、预计剩余时间；RL 可看：reward、response length、KL、actor loss/grad_norm、验证集准确率等（指标清单见 `monitor/README.md`）。所有指标同时落盘 `outputs/tb/<run>/metrics.jsonl`。

## 七、评测命令

```bash
conda activate /hdd2/xiaolirui/conda_envs/matheval

# base / 中训练后的模型：few-shot，13 个基准
MODEL_PATH=models/Llama-3.2-3B-Base bash eval/scripts/run_fewshot_base.sh
MODEL_PATH=models/OctoThinker-3B-Long-Base bash eval/scripts/run_fewshot_base.sh

# RL 之后的模型：zero-shot，用 RL 训练时的同一个提示模板，贪心解码
MODEL_PATH=models/OctoThinker-3B-Long-Zero bash eval/scripts/run_zeroshot_rl.sh
```

结果汇总在 `eval/outputs/summary/evaluation_results.csv`（每个模型一行，每个基准一列）。

## 七·五、训练监控看板

`monitor/` 目录提供基于 TensorBoard 的监控：直接解析 nanotron / verl 的控制台日志，无需改动训练代码。

```bash
# 方式一：训练启动时自动挂载（推荐）
MONITOR=1 bash midtraining/run/run_midtraining.sh <配置>.yaml     # mid-training
MONITOR=1 bash rl/scripts/run_grpo_3b_local.sh                    # RL

# 方式二：手动挂到一个正在运行的任务
bash monitor/run_monitor.sh <训练日志文件> <run名字>

# 打开看板（默认 6006 端口，浏览器访问 http://<本机IP>:6006）
bash monitor/serve.sh
```

看板指标：mid-training 有 lm_loss、ppl（由 loss 推算）、grad_norm、实际 lr、tokens/s（每卡）、MFU（model_tflops÷312，A100 bf16 峰值）、ETA、显存；RL 有 reward、KL、response length、loss、grad_norm、验证集指标等。明细见 `monitor/README.md`。

## 八、参考数字（用来对照复现结果）

论文图 1 在 MATH500（500 道竞赛数学题）上给出的关键数字：

| 模型 | MATH500（4-shot / zero-shot RL 后） |
|---|---|
| Llama-3.2-3B-Base | 7.4 |
| Llama-3.2-3B-Zero（RL 后） | 10.0 |
| OctoThinker-3B-Base（stable 阶段后） | 22.4 |
| OctoThinker-3B-Long-Base | 25.8 |
| OctoThinker-3B-Long-Zero（RL 后） | 38.2 |
| Qwen2.5-3B-Base | 7.4 |
| Qwen2.5-3B-Zero（RL 后） | 约 65.2–66.4 |

论文表 6/7/8 是 1B/3B/8B 在 13 个基准上的完整 few-shot 成绩（core 四项是 GSM8K、MATH500、OlympiadBench、AMC23）。3B 系列的对照点：Llama-3.2-3B 四项平均 10.61，OctoThinker-3B-Long-Base 23.54，Short 25.49，Hybrid 27.29；8B 系列：Llama-3.1-8B 平均 21.12，OctoThinker-8B-Long-Base 31.78，Hybrid 32.44，Short 33.32。注意论文**没有**给出 RL 后 *-Zero 模型的完整 13 基准总表——RL 结果只出现在图 12–14 的训练曲线里（纵轴就是那 4 个 core 基准的 zero-shot 准确率），所以复现 RL 时对照图 1 的 MATH500 数字和图 12–14 的曲线形态（准确率随步数上升、正确回答长度平稳增长）即可。如果复现结果和这些数字差距很大，先检查评测模板和答案提取是否正常，再怀疑训练。

## 九、按算力排期的建议（现实估计）

这台机器只有 6 张 40GB 卡，按论文规模全量复现不现实，建议按下面的顺序做：

1. **先跑 Zero-RL 基线**（Llama-3.2-3B 和 Qwen2.5-3B 各一次 RL，各约 1–2 天）。这是论文的动机实验，最快验证整条 RL 链路。
2. **挑 1–3 个关键的 20B token 中训练消融**（每个约 3.5–5 天），比如 MegaMath-Web-Pro vs FineMath-4plus 这一对，论文最核心的数据结论就来自这里。
3. **长程中训练（100B–200B）期间，用已发布的 OctoThinker-3B-*-Base 检查点跑完整的三分支 RL 对比**。100B 约需 2.5–3.5 周，完整 200B stable 阶段约 5–7 周，可以和其他实验并行排期。

经验法则：3B 模型每 20B token 约 3.5–5 天；RL 每次约 1–2 天（短分支）到 2–3 天（long 分支，因为后期要生成长回答）。

## 十、已知偏差与风险

- **vLLM 版本**：eval 原版要求 0.6.3.post1，本机统一用 **0.6.3**（与 RL 一致；post1 只是打包修复，不影响行为）。
- **transformers/tokenizers 版本**：nanotron 侧原建议 transformers==4.39.3，但它硬依赖 tokenizers<0.19，无法解析 unsloth 镜像 Llama-3.2 的 tokenizer.json（含 `ignore_merges` 字段）。nanotron 环境实际装 **transformers==4.46.3 + tokenizers==0.20.3**（与 eval 环境一致），HF↔nanotron 权重转换脚本兼容。
- **40GB 显存的训练配置**：3B + 128k 词表在 40GB 卡上跑全量 AdamW 很紧（参数 6.2GB + 梯度 6.2GB + 优化器态 24.8GB + all-gather 缓冲 6GB）。本机配置已开 `optimizer.zero_stage: 1`（ZeRO-1 把优化器态按 dp 切分）+ `recompute_layer: true`（激活检查点）；6 卡 dp6 时 3B 可训（每卡约 25-28GB）。冒烟用的是 1B 配置 `smoke_1b_tiny.yaml`。**注意**：`fp32 梯度累加`（`accumulate_grad_in_fp32: true`）会再加 12GB/卡，40GB 卡上建议保持 false、80GB 卡可开。
- **eval 代码修复**：原仓库评测链路有几处 bug 已修：CLI 参数名下划线/连字符不匹配导致 few-shot 静默失效；单卡时 worker 写的 metrics 文件名与主程序读取不一致导致 "n samples = 0"；worker GPU 映射不遵守外部 CUDA_VISIBLE_DEVICES；summarize 对 HF 风格模型名排序崩溃；新增 MATH500 配置与 RL 模型的 zero-shot 配置（`configs/en_math_zero_shot_test_configs.json`）。
- **RL 脚本适配**：`run_grpo_3b_local.sh` 已改为单机 6 卡默认（可用 CUDA_VISIBLE_DEVICES 限制），数据和模型路径全部绝对化（hydra 会改工作目录），支持 TRAIN_BATCH_SIZE/VAL_BATCH_SIZE 环境变量；2 卡冒烟时建议 VAL_BATCH_SIZE=128。
- **网络环境（重要）**：这台机器的出口有 SANGFOR TLS 拦截代理——huggingface.co 的 API 可以访问，但 LFS/CDN 大文件流量会被拦截（SSL 证书错误/空响应）；pypi.nvidia.com、download.pytorch.org 也不可用；公司内网 pip 镜像只有较新的包版本。本复现的所有下载/安装脚本已适配：pip 走阿里云/清华镜像（历史版本全），大 wheel 用 `models/fetch_remaining_wheels.py` 多线程分段下载，基座模型走 ModelScope，OctoThinker 检查点走 hf-mirror.com。如果你的网络环境变了，这些 workaround 可以全部去掉。
- **Llama 权重来源**：这台机器没有 HuggingFace 的 meta-llama 授权 token，`download_models_v2.py` 用的是 unsloth 的 Llama 镜像（ModelScope 上的同权重镜像）。镜像权重应当一致，但如果你要求严格对齐官方发布，请换用 meta-llama 官方仓库。
- **RL batch 口径**：论文写 global batch 128 条提示、PPO mini-batch 64；仓库脚本沿用代码库默认的 256 / 16（脚本头部有注释说明对应关系）。想完全对齐论文可以改 `train_batch_size` 和 `ppo_mini_batch_size`。
- **环境的少量冻结偏差**（搭建时记录，均不影响主链路）：verl 环境补装了 `torchdata==0.8.0`（vendored verl 的 StatefulDataLoader 需要，原 requirements 漏了；用 --no-deps 装以免它的 torch==2.4.0 精确pin覆盖我们的 +cu121 构建）；eval 环境的 `antlr4-python3-runtime` 用 4.13.2（与 RL 环境一致；原文件钉的 4.11.1 与 math-verify 0.5.2 依赖的 latex2sympy2_extended 1.0.6 冲突）；`pyairports` 降为 0.0.1（2.1.1 在任何镜像都不存在）；nanotron 环境的 datatrove 钉 0.3.0（nanotron 的 numpy<2 与 datatrove 新版 numpy>=2 冲突）；eval 的 vllm 用 0.6.3（原钉 0.6.3.post1 在镜像缺失，与 RL 统一为 0.6.3）。
- **中训练配置丢失**：`midtraining/nanotron-runner/` 是空的，原始训练配置没有保留下来。`midtraining/run/configs/` 里的配置是按论文正文（第 3.1、4.1、4.2 节和表 2–5）重建的，所有配置键都逐个核对过本仓库 nanotron 快照的源码；但数据配比里的两个细节论文没有写明，只能合理假设：（1）第 3.4 节 89:10:1 混合中 1% 指令数据在三个指令数据集的子划分（默认三等分）；（2）decay 阶段缺失数据集的权重处理方式。这两点不影响第 3 节核心消融的复现。
- **中训练数据**：8 个数据集上传完成后再跑 `prepare_data.py`；它会对每个数据集打印 schema 报告，格式对不上时会明确报错而不是悄悄跳过。之后按顺序跑 `shard_data.py`（切分 jsonl）和 `tokenize_data.sh`（分词），两步缺一不可：datatrove 按文件分片，不切分则整个数据集只由一个 tokenize 任务处理。
- **vLLM 显存占用**：RL 脚本里 `gpu_memory_utilization=0.75` 是为 40GB 卡调的（原脚本 0.8 是给 80GB 卡的），卡上还有别的服务时记得降低或换卡。
