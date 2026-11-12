## 独立 Runtime Server

这是一个独立的推理服务，不改动 `lerobot-rollout`，也不要求机器人侧加载 policy。

server 负责加载 checkpoint、运行 policy 和保存动作队列。通讯协议只定义观测、动作和确认消息；机器人类型、控制周期、动作执行和安全逻辑都由外部 client 自己负责。

### 启动 server

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
  --inference-mode=stream-head-mean
```

`--checkpoint` 可以是本地 checkpoint 目录，也可以是 Hugging Face Hub repo id。模型结构、processor 和 normalization 从 checkpoint 加载。`--runtime-queue-size` 是 server 每次推理后最多缓存的动作数；它不代表机器人控制频率。

当 checkpoint 是启用 `policy.streaming=true` 的 SmolVLA 时，Runtime 会使用分阶段推理：

1. 后台线程异步编码下一次 observation 的 VLM prefix，并保存 KV cache；
2. 当前 action queue 继续由 FM/action expert 消费；
3. queue 进入 watermark 后切换到最新的 prefix cache；
4. 每个 session 独立维护 prefix cache、执行步数、action queue 和 reset 生命周期。

`--mode=local` 在当前 Runtime 进程内完成 VLM/FM 两阶段；`--mode=process` 会启动独立的
VLM worker 和 FM worker。process 模式通过进程间管道传输 prepared batch 和 Torch KV-cache
artifact；两边分别使用 `--vlm-device` 和 `--fm-device`。

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

### SmolVLA streaming、observation delay 和 tactile 实验

SmolVLA 的 policy 配置现在包含与 Pi05 实验对应的研究参数：

```yaml
policy:
  streaming: false
  streaming_chunk_size: 5
  streaming_attention_mode: bidirectional  # bidirectional | mask | causal | random_gate
  streaming_constant_weight: 0.2
  streaming_chunk_wise_weight: 0.8
  observation_delay_max_chunks: 0
  observation_delay_max_tokens: 0
  use_observation_delay_embedding: false
  use_observation_delay_adarms: false
  use_tactile: false
  use_tactile_adarms: false
  use_tactile_token: false
  use_tactile_token_history: false
  tactile_token_history_length: 1
  tactile_tcn_width: 256
```

`bidirectional` 和 `mask` 使用同一组 action chunk 双向可见规则；`causal` 在
`streaming_chunk_size` 边界建立新的 attention block；`random_gate` 训练时为每个样本和
chunk 采样 `[epsilon, e]` 的连续 gate，并把 `ln(gate)` 作为跨 chunk 的 additive attention
bias。`random_gate` 推理时使用 bidirectional attention。Pi05 的 `attention_gate` 和
`learnable_gate` 不在 SmolVLA baseline 中实现。

当训练 batch 没有显式的 delay 字段时，policy 会按照配置的最大范围采样 delay；如果
batch 含有 observation history，policy 会选择对应的历史帧。tactile token history 在
训练时读取历史特征，在推理时由 policy 按 session 内的最近帧维护。

开启 `use_tactile_token` 后，触觉不会写入 VLM prefix cache，而是在每次新的 FM inference
调用开始时重新编码，并贯穿该次 denoising。FM suffix 使用 `T, CA0, A0, A1, ...` 布局：`CA0` 复制 `A0` 的 noisy
action 输入和位置条件，只有 `CA0` 可以读取 `T`，后续动作不能通过 `A0` 间接读取触觉。
开启 history 后，每个历史帧先独立编码，再经过两层因果空洞 TCN（dilation 1、2），最后一帧
输出作为 tactile token。
