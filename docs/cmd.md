## 独立 Runtime Server

这是一个独立的推理服务，不改动 `lerobot-rollout`，也不要求机器人侧加载 policy。

server 负责加载 checkpoint、运行 policy 和保存动作队列。通讯协议只定义观测、动作和确认消息；机器人类型、控制周期、动作执行和安全逻辑都由外部 client 自己负责。

### Runtime 管理的 VLM/FM（两张 GPU）

所有服务参数直接通过 CLI 指定，不使用 YAML：

```bash
uv run --extra remote lerobot-runtime-server \
  --checkpoint=/data/checkpoints/my_policy \
  --mode=process \
  --vlm-device=cuda:0 \
  --fm-device=cuda:1 \
  --host=0.0.0.0 \
  --port=8000 \
  --runtime-queue-size=64 \
  --streaming-horizon=50 \
  --streaming-watermark=35 \
  --inference-mode=stream-head-mean \
  --tactile-cfg-lambda=1.5
```

`--checkpoint` 可以是本地 checkpoint 目录，也可以是 Hugging Face Hub repo id。模型结构、processor 和 normalization 从 checkpoint 加载。`--runtime-queue-size` 是 server 每次推理后最多缓存的动作数；它不代表机器人控制频率。

FM 的 timestep 注入由 checkpoint 中的 `policy.use_noise_embedding` 控制，默认 `true`；
Runtime 加载该配置并沿用训练时的设置。此开关与 observation delay 的调制、embedding 和 token 开关独立。
`policy.use_noise_adarms` 控制 Pi0.5 的逐 token 时间调制，默认 `false`，Runtime 同样从 checkpoint 加载。

当 checkpoint 是启用 `policy.streaming=true` 的 SmolVLA 时，Runtime 会使用分阶段推理：

1. 后台线程异步编码下一次 observation 的 VLM prefix，并保存 KV cache；
2. 当前 action queue 继续由 FM/action expert 消费；
3. queue 进入 watermark 后切换到最新的 prefix cache；
4. 每个 session 独立维护 prefix cache、执行步数、action queue 和 reset 生命周期。

`--mode=local` 在当前 Runtime 进程内完成 VLM/FM 两阶段；`--mode=process` 会启动独立的
VLM worker 和 FM worker；两边分别使用 `--vlm-device` 和 `--fm-device`。每次切换到新的 VLM
prefix 时，process 模式通过进程间管道向 FM worker 提交一次 KV cache，并保留在 FM GPU 上；
随后使用同一 prefix 的调用只发送 cache 标识、当前 observation 和 delay 等动态条件。
VLM 的旧 prepared batch 不再发送给 FM，FM 始终对当前 observation 做预处理。
刷新会替换 FM 中的旧 cache；reset 或任务切换会清除缓存及标识，下次调用重新提交。

触觉 history 由外部 RuntimeSession 统一采集：每个接受的新控制观测都追加一帧，
包括当前只消费已缓存动作、没有执行 VLM/FM 的步骤；未确认动作的重放不重复追加。
池容量为已启用分支所需历史长度的最大值，启动时用首帧补齐。VLM 编码和 FM 推理分别
携带截至对应观测的固定历史快照，独立截取各自长度。local/process 两种模式均传递同一观测流
中的历史，不再按 VLM 刷新或 FM 执行次数采样；reset、任务切换及断线后的新 session 清空 history。
历史原始值使用 checkpoint 中同名触觉 feature 的 normalization，避免漏归一化或重复归一化。
快照记录 Runtime 收帧的单调时间；这不替代传感器时间戳，也不自动重采样。
客户端上报观测的频率仍应与训练数据的采样频率对齐，Runtime 无法补回没有上报的传感器帧。

`--streaming-horizon` 默认取 `policy.chunk_size`，`--streaming-watermark` 默认在
`policy.streaming_chunk_size` 个 action 后刷新 prefix。没有启用 `policy.streaming` 时，Runtime
保持普通的单阶段 chunk 推理路径。

`--inference-mode` 支持：

- `diffusion`：普通单阶段 full-chunk 推理，绕过 VLM/FM streaming prefix 和 head 调度；
- `stream`：初始化完整 action horizon 后，按 `streaming_chunk_size` 建立阶梯式
  diffusion-forcing 窗口。每个执行 chunk 清理窗口头部、左移一个 streaming chunk、补入新噪声，并用
  各位置自己的 timestep 更新未来窗口。内部窗口保留一个待执行 chunk，但 FM 前向只接收
  移除该 chunk 后的完整未来 horizon；Runtime 每次返回一个执行 chunk。
- `stream-head`：只对每个 streaming chunk 的 head（位置 `0, C, 2C, ...`）执行向量场积分，
  其中 `C=streaming_chunk_size`；
- `stream-head-context-cfg`：对每个 chunk head 使用该 chunk 内后续 token 的向量场做 CFG 引导，
  再只更新各个 chunk head；
- `stream-head-mean`：沿 chunk 维度对相同相对 token 位置做均值。对于 head 位置，均值是
  `v_0, v_C, v_2C, ...`，不是单个 chunk 内 `v_0, v_1, ...` 的均值。
- `stream-head-tactile`：要求 `policy.use_tactile_token=true`，只在当前位置 `0` 使用 `CA0`
  的向量场替代 `A0`；其它 chunk head 仍使用对应的 `A` token。
- `stream-head-tactile-cfg`：要求 `policy.use_tactile_token=true`，当前位置 `0` 使用
  `v(A0) + tactile_cfg_lambda * (v(CA0) - v(A0))`；其它 chunk head 保持 `stream-head` 逻辑。
- `stream-head-mean-tactile-cfg`：要求 `policy.use_tactile_token=true`，当前位置 `0` 使用
  `v_mean(A0) + tactile_cfg_lambda * (v(CA0) - v(A0))`，其中 `v_mean(A0)` 包含当前
  `v(A0)`；其它 chunk head 保持 `stream-head-mean` 逻辑。

三个 tactile 模式只在启用 tactile token 且每次 FM 推理都提供 tactile observation 时可用。
临时消融实验参数 `--exp-random-tactile`（默认关闭）支持 local/process Runtime，要求 SmolVLA
开启 tactile token。开启后在预处理完成的触觉空间，用独立的 `N(0,1)` 随机值替代左右触觉输入，
每次新观测刷新，history 随随机帧维护，同次 denoising 复用；VLM prefix 不随机化。
独立 RNG 固定 seed=0，session reset 后重新开始，不消耗动作噪声的 RNG。
lambda CSV 的 `exp_random_tactile` 列标记该实验。启动时在原命令后追加 `--exp-random-tactile` 即可。
该实验检验对触觉输入的敏感性；随机值可能偏离真实触觉分布，不能单凭输出变化判断触觉是否有益。
`--tactile-cfg-lambda` 控制 `CA0` 相对 `A0` 的引导强度，默认值为 `1.5`。
该参数必须为有限的非负数，只对两个 tactile CFG 模式生效。`stream-head-tactile` 直接使用
`v(CA0)`，不受 lambda 影响。历史均值只保存原始 `A` 向量场，不包含 tactile CFG 修正。

当 checkpoint 设置 `policy.use_tactile_cfg_gate=true` 时，两个 tactile CFG 模式都改用
FM 最终最新一帧 tactile token hidden state 预测的逐样本 lambda：
`lambda = tactile_cfg_lambda_max * sigmoid(MLP(h_T))`，覆盖 `--tactile-cfg-lambda` 的固定值。
每次 FM forward 都重新预测；`stream-head-tactile` 仍直接使用 `CA0`，其它模式不应用 gate。
gate 默认初始 lambda 为 `0.5`、上限为 `2.0`。训练该 checkpoint 时只更新 gate，
冻结整个原模型，以 `MSE(v(A0) + lambda * (v(CA0)-v(A0)), target_0)` 监督；
mean 模式可以使用该 gate，但当前训练不模拟历史均值。无死区和额外 lambda 正则。

两个 tactile CFG 模式默认记录实际使用的 lambda（包括固定值和 gate 预测值）到仓库根目录
`log/tactile-cfg-lambda/${exp_name}/<UTC时间>_<PID>_<唯一标识>.csv`，不需要额外启动参数。
`exp_name` 优先从实际加载权重目录的 `resolved_experiment.yaml` / `experiment.yaml` 读取，
兼容 `config.json` 的 `exp_name` 和 `train_config.json` 的 `job_name`；缺失时使用 `unknown`。
每个控制步骤、每个 batch 样本记录一行，包含 UTC 时间、session、step、batch_index、推理模式、
lambda 来源（`learned` / `fixed`）、lambda、积分前 timestep 和 checkpoint 路径。
同一行还记录位置 0 两分支的差异：`ca0_a0_mse = mean((v(CA0)-v(A0))²)`、
`ca0_a0_rms = sqrt(ca0_a0_mse)`，以及 `a0_rms`、`ca0_rms` 和
`cfg_correction_rms = abs(lambda) * ca0_a0_rms`。均在归一化的 action 速度场空间计算，
只包含真实动作维度，不包含 padding。MSE 和 RMS 使用相同差异，后者方便与速度场幅度比较。
`ca0_a0_cosine_similarity` 记录 `dot(v(CA0),v(A0))/(norm(v(CA0))*norm(v(A0)))`，
范围为 `[-1,1]`：1 为同向，0 为正交，-1 为反向。同样排除 padding，在归一化的动作空间计算；
任一速度场 RMS 不超过 `1e-8` 时方向无定义，该列留空。
在线推理没有目标速度场，这些列衡量分支差异，不代表各分支的预测误差。
不同模型加载/进程使用独立文件，session reset 后 step 从零开始；训练 forward 和不使用 CFG 的模式不写日志。

有目标的 tactile-token 训练/离线评估 forward 还会输出 `tactile/a0_target_mse`、
`tactile/ca0_target_mse`、`tactile/ca0_a0_mse`、`tactile/ca0_a0_rms`、
`tactile/ca0_target_improvement`（A0 MSE 减去 CA0 MSE，正值表示 CA0 更好）和
`tactile/ca0_better_fraction`（CA0 的 MSE 严格更小的样本比例）。开启 gate 后另有
`tactile/fused_target_mse`。这些指标只比较位置 0，排除动作 padding 维度及位置 0 越过 episode
边界的样本，通过现有训练指标日志输出，不写推理 CSV，也不改变训练 loss。

head streaming 会把 action token 身份和窗口位置分开维护。窗口每前进一步，token 的历史
向量场跟着 token 移动；当同一个 token 到达 chunk head 时，才使用它在前面位置保存的字段
和当前字段做均值或 CFG。比如 `A0` 从 `v4` 滑到 `v0` 时，使用
`v4(A0), v3(A0), v2(A0), v1(A0), v0(A0)`。

对于 `streaming_chunk_size=C`，当前 head 的 `stream-head-mean` 向量场为：

```text
v_mean = (v0 + vC + v(2C) + ... + v((N-1)C)) / N
```

Runtime streaming 只通过 `streaming-horizon` 和 `streaming-watermark` 控制滑动窗口，避免
额外的 action 执行步数参数与 watermark 产生两套相互竞争的窗口语义。

### 通讯协议

协议实现位于 [`src/lerobot/runtime_protocol.py`](../src/lerobot/runtime_protocol.py)，使用长度前缀的 MessagePack 帧。每个控制周期的基本顺序是：

```text
client: OBSERVATION(sequence, features, task)
server: ACTION(action_id, action, queue_size)
client: CONTROL(op="ack", action_id)
```

`features` 是字符串到 NumPy 数组的映射，例如图像和本体状态。server 收到观测后，如果动作队列非空就取队列头；队列为空才调用 policy 生成新的动作块。server 返回动作后会保留 pending action，直到收到对应 `ack`；确认丢失时重发观测会得到同一个动作，不会跳过队列项。

reset 使用：

```text
client: CONTROL(op="reset")
server: ACK(op="reset")
```

协议层不包含 `robot_type`、`action_fps`、相机名称或模型选择逻辑。client 应使用与 checkpoint 训练约定一致的 feature 名称和数组 shape；具体映射属于 client 或机器人应用。


## 训练

### 数据图像的 RGB/BGR 约定

训练配置通过 `dataset.image_color_order` 声明**解码后数组的实际颜色通道顺序**，
默认 `rgb`。对于解码后红蓝仍然颠倒的历史数据，配置：

```yaml
dataset:
  image_color_order: bgr
```

也可通过 `--dataset.image_color_order=bgr` 覆盖。训练入口会在图像增强之前将 BGR
转为 RGB，支持普通和 streaming 数据集、DataLoader 和 GPU 增强；留出评估集执行相同的
颜色转换，不执行随机增强。深度不受影响。使用数据集自身的图像统计量时，对应通道的
统计量也会交换；ImageNet 的 RGB 统计量保持其原始顺序。此配置保存在训练配置中，
不加入推理预处理器；新训练模型应接收 RGB 输入。

不能根据像素或 MP4 的 `pix_fmt` 自动推断历史数据是否把 BGR 当 RGB 写入。
`bgr` 不表示“相机曾输出 BGR”：若编码时已经正确声明输入为 `bgr24`，再解码为
`rgb24`，输出就是 RGB，应配置 `rgb`。可用画面中已知的红色/蓝色物体检查解码结果。

PyAV 是 FFmpeg 库的 Python 接口，二者没有固定的 RGB/BGR 差异；顺序取决于输入、
输出像素格式。PyAV 的 `VideoFrame.from_ndarray(bgr, format="bgr24")` 对应正确声明
BGR 输入；`frame.to_ndarray(format="rgb24")` 会转换为 RGB。FFmpeg 命令行读取
裸帧时用输入侧 `-pixel_format bgr24` 声明，输出裸帧时用 `-pix_fmt rgb24` 选择 RGB。
常见 MP4 内部使用 YUV，内部像素格式不能反映编码前是否错误解释过红蓝通道。

### EMA 权重平均（YAML 配置）

`ema` 是训练配置的顶层字段，与 `policy`、`optimizer` 同级，不要放在 `policy` 下面。
[`configs/default.yaml`](../configs/default.yaml) 默认是 `ema: null`（关闭）；在实验 YAML
中添加下面任意一种配置即可覆盖默认值。以下数值均为对应类型的代码默认值。

渐进增大衰减系数（`warmup`）：

```yaml
ema:
  type: warmup
  update_after_step: 0
  inv_gamma: 1.0
  power: 0.75
  min_decay: 0.0
  max_decay: 0.9999
```

使用默认参数时，只写 `ema: {type: warmup}` 即可。`warmup` 的衰减系数随平均更新次数增大，
并限制在 `min_decay` 和 `max_decay` 之间。

固定衰减系数（`constant`）：

```yaml
ema:
  type: constant
  update_after_step: 0
  decay: 0.999
```

幂函数衰减（`power`，EDM2 风格）：

```yaml
ema:
  type: power
  update_after_step: 0
  sigma_rel: 0.1
```

`sigma_rel` 控制平均窗口相对训练长度的宽度，取值范围为 `(0, 1/sqrt(12)]`。
切换类型时，使用对应的参数，例如 `decay` 只用于 `constant`，`sigma_rel` 只用于 `power`。

三种类型都支持 `update_after_step`：按实际执行的 optimizer 更新次数计数，前 N 次更新仅复制
当前权重，第 N+1 次仍复制用于初始化，之后开始加权平均。设置梯度累积时，不按 micro-batch 计数。
EMA 只平均可训练参数，使用 float32 保存平均值；训练过程中的环境评估使用 EMA 权重。

要在某个实验中关闭继承来的 EMA，显式写：

```yaml
ema: null
```

### SmolVLA streaming、observation delay 和 tactile 实验

SmolVLA 的 policy 配置现在包含与 Pi05 实验对应的研究参数：

```yaml
policy:
  streaming: false
  streaming_chunk_size: 5
  streaming_attention_mode: bidirectional  # bidirectional | mask | causal | random_gate
  streaming_constant_weight: 0.2
  streaming_chunk_wise_weight: 0.8
  use_noise_embedding: true
  use_noise_adarms: false
  use_modality_count_bias: false  # FM logits subtract log(visible keys per modality).
  observation_delay_max_chunks: 0
  observation_delay_max_tokens: 0
  use_observation_delay_sincos: false
  use_observation_delay_embedding: false
  use_observation_delay_adarms: false
  use_observation_delay_adaln_zero: false
  observation_delay_zero_bypass: false
  use_observation_delay_token: false
  use_tactile: false
  use_tactile_adarms: false  # AdaLN-Zero: per-layer RMSNorm scale/shift + zero residual gates.
  use_tactile_token: false
  use_vlm_tactile_token: false  # Independent tactile MLP/TCN and projector in the VLM prefix.
  use_tactile_cross_attention: false  # CA0 and the tactile block read live FM tactile K/V.
  use_tactile_cfg_gate: false
  use_tactile_reconstruction: false
  tactile_reconstruction_hidden_dim: 256
  tactile_cfg_gate_hidden_dim: 256
  tactile_cfg_lambda_init: 0.5
  tactile_cfg_lambda_max: 2.0
  use_tactile_token_history: false  # True: TCN-fused token. False: direct history block.
  use_tactile_causal_attention: false  # Causal history attention requires TCN disabled.
  tactile_no_vlm_attention: false  # Tactile queries do not read VLM prefix K/V when true.
  tactile_token_history_length: 1  # Legacy fallback when a branch length is null.
  fm_tactile_token_history_length: null  # FM history length, including the current frame.
  vlm_tactile_token_history_length: null  # VLM history length at prefix refreshes.
  tactile_tcn_width: 256
```

数量偏置实验：设置 `policy.use_modality_count_bias: true`（默认 `false`）。仅对 FM query，
在已有 attention bias 之后、softmax 之前，对每个 key 减去 `log(n_visible)`。
模态分为 prefix（整个 VLM cache，含 state）、action（含 CA0）和辅助条件（tactile + delay 共组）。
归属由序列布局确定，数量按每次 attention 的布尔 mask 对每个样本、query 实时统计；
padding 和不可见 key 不计数，空模态不参与注意力。各 head 共用可见数量，匹配分数仍按 head 独立计算。
这等价于模态内 softmax 后，以各模态的平均指数分数归一化加权，不引入额外参数或 FM 前向。
VLM query 的 softmax 保持原样。self/cross 层均生效，但 cross 层只对实际提供的 K/V 分组：
当前 delay token 没有加入 cross K/V；开启触觉 cross 路径后，CA0 的 cross 归一化包含 prefix 和 tactile。
只有一个可见模态时，修正是 softmax 的常数平移，数学结果不变。

让 `CA0` 在 cross-attention 层也能读取当前触觉，训练配置设置：

```yaml
policy:
  use_tactile: true
  use_tactile_token: true
  use_tactile_cross_attention: true
  use_tactile_adarms: false
```

`use_tactile_cross_attention` 默认 `false`，要求 `attention_mode: cross_attn`。
开启后，FM cross-attention 的 K/V 为「缓存的 VLM prefix + 当前 FM tactile block 的独立 K/V 投影」。
每层投影整个 tactile block 的 hidden states，`CA0` 读取整个 block，触觉 query 按 block 内的双向/因果 mask 可见；
`A0、A1…`、delay query 仍只读 prefix。self-attention 的触觉隔离 mask 保持原规则，
普通动作也不能读取 `CA0`，防止触觉间接传播。
触觉只随 FM 请求刷新，不加入 VLM token 或复用的 prefix KV cache，无需重跑 VLM。
这条路径需要训练新增 K/V 投影（PEFT 下也会完整保存、训练），不增加 FM 前向次数。
`CA0-A0` 保留触觉/无触觉对照。要让普通动作完全不接收触觉，还需保持
`use_tactile_adarms: false`，因为该独立调制路径会向所有 FM token 注入触觉。

触觉重建探针训练可设置：

```yaml
policy:
  use_tactile: true
  use_tactile_token: true
  use_tactile_cfg_gate: false
  use_tactile_reconstruction: true
  tactile_reconstruction_hidden_dim: 256
```

从完整的 tactile-token checkpoint 初始化，冻结整个原模型（包括触觉编码器），
只训练新增的 `Linear → SiLU → Linear` 重建 MLP。输入为 `_encode_tactile()` 输出：
开启 TCN 时取历史 TCN 和 `tactile_token_out` 之后的融合特征；关闭 TCN 时取历史 block
中最新一帧的触觉 MLP 输出。目标为进入触觉编码器之前、已经归一化的当前帧
左右手 marker，按左手、右手顺序展平拼接。输出维度从两手输入特征形状推导，
UniVTAC 为 504 维，隐藏层宽度沿用 `tactile_reconstruction_hidden_dim`。
使用全部触觉维度上的 MSE，替代动作/CFG loss；编码器和目标均停止梯度。
该训练分支跳过 VLM/FM forward，直接检验触觉编码器及 TCN 的信息保留。
当前逻辑替换了此前的 `最终 h_T → TCN embedding` 重建，其 loss 数值不可直接比较。
该模式与 lambda gate、PEFT 互斥；重建头用于训练和离线评估，推理动作路径不使用它。

触觉逐层调制沿用配置名 `use_tactile_adarms`：同时设置 `use_tactile: true` 和
`use_tactile_adarms: true` 后，当前帧左右手 marker 经编码器和条件 MLP，传入 FM 每层
attention / MLP 的独立 AdaLN-Zero 投影。底层保留 RMSNorm，计算
`norm(h) * (1 + scale) + shift`，残差为 `h + gate * branch`；scale、shift、gate
的投影权重和偏置全零初始化，单独使用时 expert block 初始为恒等映射。
触觉条件按动作观测刷新，不写入 VLM prefix cache，也不再直接加到 action embedding。
与 noise AdaRMS 或 delay AdaLN-Zero 同时开启时，两路 residual gate 相加，避免零 gate
相乘阻断梯度；触觉不调制最终输出 norm。`use_tactile_token` 可独立开启。
旧版 `use_tactile_adarms` 的输入相加实现已替换，旧触觉 checkpoint 需要重新训练或微调新增投影。

`use_noise_embedding` 控制 FM 输入端的 timestep 注入，默认 `true`：sin/cos 时间编码与
action embedding 拼接后经过 MLP。设为 `false`（训练 CLI：`--policy.use_noise_embedding=false`）时，
时间编码分支置零，保留 MLP 和 checkpoint 参数形状；动作加噪、timestep 采样与去噪积分照常执行。
该开关独立于 observation delay 条件，保存到 checkpoint 后也用于推理。

要使用 Master_Research / Pi0.5 的噪声等级注入路径，设置：

```yaml
policy:
  use_noise_embedding: false
  use_noise_adarms: true
```

该组合用 `Linear(x_t)` 作为 action token；时间条件为
`SiLU(Linear(SiLU(Linear(sin/cos(t)))))`。FM 每层 attention 和 MLP 前使用不带静态权重的
RMSNorm（epsilon 为 `1e-6`），由条件直接线性生成 scale、shift、gate，计算
`RMS(h) * (1 + scale) + shift`，残差为 `h + gate * branch`。最终输出 norm 也由时间条件调制。
调制投影和 gate 零初始化，与参考实现一致；新参数需要训练。

stream 的 `[B, L]` timestep 逐 token 编码，保留 chunk-wise 的时间差异，不会压成 batch 级条件；
`CA0` 复制 `A0` 的时间条件，tactile/delay 条件 token 使用 clean time `t=0`。
如同时开启 delay 的 AdaRMS 或 AdaLN-Zero，delay 编码与时间条件相加后共用这组 scale/shift/gate，
沿用已有 delay 编码方式，不再叠加另一组残差 gate。触觉的输入 embedding 和 token 路径仍由原开关控制。
两个 noise 开关都为 `true` 时，会同时启用输入端拼接与逐层调制；两个都为 `false` 时，关闭显式时间条件。
旧 checkpoint 缺少 `use_noise_adarms` 时默认 `false`，保留原有结构。

`bidirectional` 和 `mask` 使用同一组 action chunk 双向可见规则；`causal` 在
`streaming_chunk_size` 边界建立新的 attention block；`random_gate` 训练时为每个样本和
chunk 采样 `[epsilon, e]` 的连续 gate，并把 `ln(gate)` 作为跨 chunk 的 additive attention
bias。`random_gate` 推理时使用 bidirectional attention。Pi05 的 `attention_gate` 和
`learnable_gate` 不在 SmolVLA baseline 中实现。

旧编码下（`use_observation_delay_sincos=false`），开启 `use_observation_delay_adarms` 后，
delay 经归一化和 MLP 编码，在 FM 每个 expert 层的
attention 和 MLP 前分别生成 scale/shift，执行 `RMSNorm(h) * (1 + scale) + shift`。
这是保留 RMSNorm 的 AdaLN 式调制，不再把该分支加到输入 action embedding 上。
调制投影零初始化；训练和 runtime 使用同一路径，runtime 每次根据当前 delay 调制，VLM prefix
cache 不包含 delay 条件。旧版 checkpoint 的 delay MLP 可加载，但新增的逐层调制参数需要训练，
无法等价复现旧版输入相加的行为。`use_observation_delay_embedding` 仍单独控制输入 embedding 相加。

`use_observation_delay_sincos=true` 将上述 delay 条件改为逐 action token 的相对观测时间：
采样的 `d` 是首个动作距观测的年龄，后续动作用 `d, d+1, ..., d+L-1`。
chunks 模式先用 `d * streaming_chunk_size` 换算成动作步；仅首个 delay 按配置上限截断，
后续 `d+i` 不归一化、不截断，采用标准 Transformer 频率 `10000^(-2k/H)` 的 sin/cos
编码，再经 `Linear(H,H) -> SiLU -> Linear(H,H)`。
embedding 路径将各自向量加到动作输入；AdaRMS/Zero 路径逐位置生成 scale/shift（及 gate），
保留原 RMSNorm；token 路径使用 `D0,...,D(L-1),[T,CA0],A0,...,A(L-1)`。
每个 `Di` 只能读取有效 prefix 和自身，`Ai` 读取自己的 `Di`，`CA0` 读取 `D0`。
触觉特殊 token 的隔离规则保留，`CA0` 与 `A0` 使用相同的 `d` 条件；同时开启多条 delay
注入路径时，各路径使用独立 MLP。输入 noise timestep 和 RoPE 保持原有机制。
该开关保存到 checkpoint，diffusion/stream/stream-head 都使用同一编码；即使 `d=0`，
各动作的位置条件仍是 `0,1,...,L-1`。默认 `false` 保留旧实验和 checkpoint；新路径须重新训练，
不能对旧 delay checkpoint 直接切换编码。delay 采样和历史帧选择规则保持原有实现。

`use_observation_delay_adaln_zero=true` 启用保留 RMSNorm 的 AdaRMSNorm-Zero 对照，与
`use_observation_delay_adarms` 互斥。沿用已有配置名；每个 expert 层的 attention/MLP 保留原有
RMSNorm（包括预训练权重、epsilon 和可训练状态），再由 `Linear(SiLU(delay_condition))`
生成 scale、shift、gate：`h = h + gate * Branch(RMSNorm(h) * (1 + scale) + shift)`。
投影全部零初始化，因此初始 expert block 为恒等映射，会关闭原预训练残差分支；这不是
保留预训练输出的 adapter，需要重新训练，不能直接复用 003 的逐层调制权重。
输入端 noise timestep、末尾 RMSNorm 和 delay 采样规则保持原有实现。配置保存进 checkpoint，
diffusion/stream/stream-head 推理自动使用相同结构，无需新增 runtime 参数。

`observation_delay_zero_bypass=true` 在独立的 delay Zero 路径上按每个样本首动作的 `d`
旁路调制（要求 `use_observation_delay_adaln_zero=true`、`use_noise_adarms=false`）：
`g=(d>0)`，`scale=g*scale_pred`、`shift=g*shift_pred`，残差 gate 为
`where(g, gate_pred, 1)`。因此 `d=0` 时整段 action 使用原有 RMSNorm 和普通残差相加，
而不是关闭 attention/MLP；`d>0` 保留原 Zero 行为。判定不使用 `d+i`，开启 sin/cos 时
也能旁路整个零延迟样本。此开关仅控制 Zero 层内调制，不屏蔽另行开启的 embedding/token
注入。训练和推理使用相同规则，配置随 checkpoint 保存；默认关闭以复现旧权重。
007 配置启用此开关，需要重新训练以比较效果；主干权重仍会更新，不保证输出等同于 001。
关闭 `use_observation_delay_sincos` 时，AdaRMS/Zero 恢复 `d/max_delay -> MLP`；只有
embedding/token 路径恢复可学习的 embedding table。

旧编码下开启 `use_observation_delay_token` 后，FM suffix 会加入一个 delay token。该 token 只能读取
VLM prefix KV cache 和自身，所有 action token 都可以读取它；delay token 不读取 action token，
因此不会改写 action 的噪声输入和 timestep 条件。它与 tactile token 使用独立的 attention mask。

当训练 batch 没有显式的 delay 字段时，policy 会按照配置的最大范围采样 delay；如果
batch 含有 observation history，policy 会选择对应的历史帧。tactile token history 在
训练时读取数据集历史，Runtime 推理时读取外部统一 history 的快照；直接 policy API 保留内部队列回退。

开启 `use_tactile_token` 后，触觉不会写入 VLM prefix cache，而是在每次新的 FM inference
调用开始时重新编码，并贯穿该次 denoising。触觉 MLP 为 `504 → 720 → 720`（UniVTAC 默认配置），
隐藏层和输出层均跟随 FM expert hidden size，不再包含 `tactile_token_proj`。
FM suffix 使用 `T0, ..., T(H-1), CA0, A0, A1, ...` 布局：`CA0` 复制 `A0` 的 noisy
action 输入和位置条件。self-attention 中，`CA0` 与 `A0` 读取相同的其他 action token
（遵循 streaming attention mask），各自可以读取自己，但互不可见；只有 `CA0` 动作分支可以读取触觉 block。
触觉 block 默认内部双向可见，只读取有效 prefix 和触觉 block，不读取 delay、CA0 或 action。
其他 action 可以按原动作 mask 读取 `A0`，不能读取 `CA0` 或触觉 block，因此不会间接接收触觉。
该隔离保证要求关闭独立的全 token 触觉调制 `use_tactile_adarms`。
`fm_tactile_token_history_length` 和 `vlm_tactile_token_history_length` 分别指定两侧包含当前帧的历史长度。
两者默认 `null`，各自回退到旧配置 `tactile_token_history_length`（默认 1），兼容已有实验与 checkpoint。
显式设置任意一侧的新字段只覆盖该侧，另一侧保持自己的配置或兼容回退值。
`use_tactile_token_history=true` 时，每帧先独立经过 MLP，再经过两层因果空洞 TCN
（dilation 1、2），最后一帧输出作为单个 tactile token（`H=1`）。
`use_tactile_token_history=false` 时跳过 TCN，每帧 MLP 输出按从旧到新的顺序直接组成触觉 block，
此时各侧 `H` 等于自己的历史长度。训练读取足以覆盖两侧历史及 VLM delay 的窗口，再独立截取；Runtime 按 session
维护可容纳两侧所需长度的共享池，启动时不足的历史用第一帧补齐，reset 后清空。
所有触觉 token 使用 clean noise time；CA0 保持与 A0 相同的 noise/delay 条件和 RoPE 位置。
action RoPE 从 A0 开始连续递增，CA0 复用 A0 位置，不额外占用一个位置。
例如忽略 prefix/delay 的位置偏移，3 帧触觉的 `T0,T1,T2,CA0,A0,A1,A2` 位置为
`0,1,2,3,3,4,5`，训练和缓存推理一致。

若要使用触觉历史内部的因果注意力：

```yaml
policy:
  use_tactile: true
  use_tactile_token: true
  use_tactile_token_history: false
  fm_tactile_token_history_length: 10
  use_tactile_causal_attention: true
```

每个 `Ti` 只读取自己及更早的 `T0..Ti`，仍可读取有效 prefix；CA0 继续读取整个触觉 block，
普通 action 和 delay 不读取触觉。self-attention 与开启触觉 cross-attention 时的 live K/V
使用相同的因果规则。该开关默认 `false`，与 TCN 模式互斥，保存至 checkpoint 后推理自动使用。

要让触觉 token 不读取 VLM 的图像、语言及 state prefix，设置：

```yaml
policy:
  tactile_no_vlm_attention: true
```

要求 `use_tactile_token=true`，可与 TCN、直接拼接的双向或因果触觉 block 配合。
只屏蔽触觉 query 到 VLM prefix K/V 的 attention，CA0 和普通 action 仍按原规则读取 prefix，
CA0 仍读取整个触觉 block。noise time、delay 调制和 RoPE 不变。
训练与缓存推理的 self/cross-attention 均应用该限制；没有触觉 live K/V 的 cross 层中，
触觉 query 无可见 key 时 attention 输出为零。开关默认 `false`，保存到 checkpoint 后推理自动使用。

旧版 `960 → 960` 触觉编码器及额外投影与该结构不兼容，需要重新训练触觉参数。

VLM 侧触觉模式可单独开启，也可与 FM 触觉同时开启：

```yaml
policy:
  use_tactile: true
  use_vlm_tactile_token: true
  use_tactile_token: false  # FM 分支开关；设为 true 可同时启用两套独立编码器。
  use_tactile_token_history: false
  vlm_tactile_token_history_length: 7
  fm_tactile_token_history_length: 3
```

VLM prefix 顺序为 `[图像, 语言, state, VLM触觉block]`。触觉与图像/语言双向可见，
触觉不读取 state；图像/语言继续不读取 state。state 保持原来的 prefix 可见关系，
不直接读取后追加的触觉。有效帧和 padding mask 同时生效。

两分支独立控制历史长度，共享 TCN 开关与宽度、触觉内部因果开关和 `tactile_no_vlm_attention`，
但 MLP/TCN 的参数完全独立。默认均为 `504 → 720 → 720`；TCN 输出也为 720 维，
VLM 侧额外经过 `vlm_tactile_projector: 720 → VLM hidden size`（默认 960）。
关闭 TCN 时两边各有对应历史长度的 token，开启时各自使用配置的历史帧并融合为一个。VLM 侧无 FM noise/delay AdaLN，
使用普通 VLM RMSNorm；其 RoPE 位置在 state 后连续递增。

`use_tactile_causal_attention` 只约束触觉 block 内部的直接边；当图像/语言与触觉双向交互时，
后层触觉仍可能经图像/语言间接读取较新的触觉信息。若同时开启 `tactile_no_vlm_attention`，
触觉不读取图像/语言，但图像/语言仍读取触觉。

VLM 触觉及其对图像/语言的影响写入 KV cache，随 prefix 刷新；FM 使用当前观测的触觉。
Runtime 统一维护原始触觉历史，再向 VLM/FM 提供对应观测时刻的快照，各侧独立截取自己的长度。
VLM 低频刷新只影响缓存的新鲜程度，不改变历史帧的采样间隔。训练时 VLM 触觉历史与采样
delay 对应的图像/state 对齐，FM 保留最新历史。不经过 Runtime 的直接 policy API 仍保留内部
历史队列作为兼容路径；Runtime 提供历史窗口时不再更新内部队列。
开启 VLM 触觉后普通 A0 也可通过 prefix 获取触觉，因此 CA0/A0 对比仅隔离额外的 FM 触觉路径。
该模式默认关闭，不新增 Runtime CLI；按 checkpoint 配置自动加载，新增 VLM 编码器须训练。
