# OctoThinker 训练监控（monitor/）

把两个训练框架的**控制台日志**实时转成 TensorBoard 曲线，训练过程中随时打开
`http://<机器IP>:6006` 查看 loss / ppl / 梯度范数 / MFU / 实际学习率 / 吞吐 / ETA。

- 只读日志文件，不碰训练进程；运行在 miniforge **base python** 下
  （`/home/lawson/miniforge3/bin/python3`，已装好 `tensorboard`，训练环境无需安装任何东西）。
- 每个 run 一个子目录：`outputs/tb/<run_name>/`，内含 TB 事件文件和 `metrics.jsonl`（逐 step 的 JSON 备份）。

## 各阶段可用指标

**mid-training（nanotron）** — 解析 `LoggerWriter` 每步一行（`iteration: N / M | lm_loss: ...`）：

- 原始：`lm_loss`、`lr`、`grad_norm`、`tokens_per_sec(_per_gpu)`、`consumed_tokens`、
  `model_tflops_per_gpu`、`hardware_tflops_per_gpu`、`elapsed_time_per_iteration_ms`、
  `global_batch_size`，偶尔还有 `cuda_memory_allocated`、`hd_*_memory_tb` 等
- 派生（自动计算）：
  - `train/ppl = exp(lm_loss)`
  - `train/mfu = model_tflops_per_gpu / 312.0`（A100-PCIE-40GB bf16 稠密峰值）
  - `train/eta_hours`、`train/progress`（按 token 数估算剩余时间，见下文）
  - `train/tokens_per_sec_ema`（EMA α=0.1 平滑）
- 全部挂在 `train/` 分组下。

**RL（verl / RayPPOTrainer）** — 解析 `LocalLogger` 的 console 行
（`step:N - critic/kl:0.005 - actor/pg_loss:-0.012 - ...`，见
`rl/verl/utils/logger/aggregate_logger.py` 的 `concat_dict_to_str`）：

- `critic/kl`、`critic/kl_coeff`、`actor/pg_loss`、`actor/entropy_loss`、`actor/grad_norm`、
  `critic/rewards/{mean,max,min}`、`response_length/*`、`prompt_length/*`、
  `timing_s/*`、`timing_per_token_ms/*`、`Tokens/Sec/GPU`、`total_num_tokens`、`time_per_step`
- 验证（`test_freq` 触发）：`val/test_score/{avg_acc,<数据源>}`、`val_response_length/*`
- key 本身已带 `critic/ actor/ timing_s/ val/` 分组，直接保留原样（不再加 `train/` 前缀）。

## 用法

### 方式 A：手动挂到正在运行的训练

```bash
bash monitor/run_monitor.sh outputs/midtraining/logs/<log文件名> <run_name> nanotron
bash monitor/run_monitor.sh outputs/rl/long/logs/<log文件名> <run_name> verl
# 格式省略时默认 auto（两种行都能认）
```

幂等：同一 run_name 再跑一次会先杀掉旧 tailer 再启动。tailer 日志在
`outputs/tb/<run_name>/monitor.out`。

### 方式 B：启动训练时自动挂（MONITOR=1）

```bash
MONITOR=1 CONFIG=configs/smoke_3b_tiny.yaml bash midtraining/run/run_midtraining.sh
MONITOR=1 bash rl/scripts/run_grpo_3b_local.sh
```

训练脚本会后台启动 `log2tb.py`（nohup 式，随脚本退出自动被 trap 杀掉），
其余行为与不加 MONITOR 完全一致。

### 打开看板

```bash
bash monitor/serve.sh            # 默认 outputs/tb，端口 6006
bash monitor/serve.sh --root outputs/tb --port 6006
```

启动时会打印本机与局域网 URL。`outputs/tb` 下的每个子目录就是一个 run，
同一个页面里并排对比。

### 离线补跑（训练已结束）

```bash
/home/lawson/miniforge3/bin/python3 monitor/log2tb.py \
    --log <training.log> --tb-dir outputs/tb/<run_name> --format auto --once
```

## MFU / ppl / ETA 怎么算，以及坑

- **MFU**：`model_tflops_per_gpu / 312.0`。分母是 A100-PCIE-40GB 的 bf16 稠密算力
  312 TFLOPS（本机显卡）。原始 `model/hardware_tflops_per_gpu` 也一并记录。
- **ppl**：对 `lm_loss` 直接取 `exp`。注意 nanotron 日志里的 loss 是
  `human_format()` 保留 3 位有效数字后的值（如 `2.345`），ppl 有轻微舍入误差；
  且 loss 本身是所有日志 token 的平均，曲线抖动属正常。
- **ETA**：`剩余token = 总token - consumed_tokens`，`eta_hours = 剩余token / tokens_per_sec`。
  总 token 的取法（按优先级）：
  1. `--eta-total-tokens N` 直接指定；
  2. `--tokens-per-step N` × 日志里的总步数；
  3. `global_batch_size × --seq-len`（默认 8192）× 总步数 —— mid-training 脚本
     启动时会从 YAML 的 `tokens.sequence_length` 自动取，不用手填。
  日志中的 `consumed_tokens`、`tokens_per_sec` 都是 3 位有效数字（K/M/G 后缀已正确展开），
  ETA 是粗估，看个量级即可。
- **nanotron 数值格式**：K/M/G/T/P 后缀是 1000 进制（`51.4M = 51,400,000`），
  小于 1 的数可能是科学计数法（`3e-05`），解析器都已处理；只有 DP rank 0
  （`|DP=0|`）的行会被采纳，其余 DP rank 的行自动忽略。
- **verl 数值精度**：console 行里所有数值被格式化为 `:.3f`，很小的量（如 KL）
  会显示为 `0.000` —— 这是 verl 自身的打印精度，不是解析丢的；要更高精度请开 wandb。

## 文件

- `log2tb.py` — tailer：tail -F 式跟随（处理轮转/重建/截断），按 (step, key) 去重，
  每 10 个新 step flush 一次，SIGTERM/SIGINT 时 flush 后退出；每步打印一行终端摘要。
  TB 事件用纯 `tensorboard` 包写入（`EventFileWriter` + proto，不依赖 torch）。
- `serve.sh` — 启动 TensorBoard。
- `run_monitor.sh` — 一键挂日志（nohup 后台）。
- `tests/` — 样例日志 + `test_parse.py`，跑法：
  `/home/lawson/miniforge3/bin/python3 monitor/tests/test_parse.py`
