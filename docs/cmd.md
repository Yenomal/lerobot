## 独立 Runtime Server

这是一个独立的推理服务，不改动 `lerobot-rollout`，也不要求机器人侧加载 policy。

server 负责加载 checkpoint、运行 policy 和保存动作队列。通讯协议只定义观测、动作和确认消息；机器人类型、控制周期、动作执行和安全逻辑都由外部 client 自己负责。

### 启动 server

所有服务参数直接通过 CLI 指定，不使用 YAML：

```bash
uv run --extra remote lerobot-runtime-server \
  --checkpoint=/data/checkpoints/my_policy \
  --device=cuda:0 \
  --host=0.0.0.0 \
  --port=8000 \
  --runtime-queue-size=64
```

`--checkpoint` 可以是本地 checkpoint 目录，也可以是 Hugging Face Hub repo id。模型结构、processor 和 normalization 从 checkpoint 加载。`--runtime-queue-size` 是 server 每次推理后最多缓存的动作数；它不代表机器人控制频率。

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
