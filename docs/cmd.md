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
`--tactile-cfg-lambda` 控制 `CA0` 相对 `A0` 的引导强度，默认值为 `1.5`。
该参数必须为有限的非负数，只对两个 tactile CFG 模式生效。`stream-head-tactile` 直接使用
`v(CA0)`，不受 lambda 影响。历史均值只保存原始 `A` 向量场，不包含 tactile CFG 修正。

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
  observation_delay_max_chunks: 0
  observation_delay_max_tokens: 0
  use_observation_delay_sincos: false
  use_observation_delay_embedding: false
  use_observation_delay_adarms: false
  use_observation_delay_adaln_zero: false
  observation_delay_zero_bypass: false
  use_observation_delay_token: false
  use_tactile: false
  use_tactile_adarms: false
  use_tactile_token: false
  use_tactile_token_history: false
  tactile_token_history_length: 1
  tactile_tcn_width: 256
```

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
训练时读取历史特征，在推理时由 policy 按 session 内的最近帧维护。

开启 `use_tactile_token` 后，触觉不会写入 VLM prefix cache，而是在每次新的 FM inference
调用开始时重新编码，并贯穿该次 denoising。FM suffix 使用 `T, CA0, A0, A1, ...` 布局：`CA0` 复制 `A0` 的 noisy
action 输入和位置条件，只有 `CA0` 可以读取 `T`，后续动作不能通过 `A0` 间接读取触觉。
开启 history 后，每个历史帧先独立编码，再经过两层因果空洞 TCN（dilation 1、2），最后一帧
输出作为 tactile token。
