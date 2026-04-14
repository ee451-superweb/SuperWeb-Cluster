# Networking Experiments

`superweb-cluster` 的总入口仍然是项目根目录下的 `bootstrap.py`。  
这个目录不是主运行时入口，而是 Sprint 1 阶段保留下来的独立网络实验区。

目前项目已经完成 Sprint 1，准备进入 Sprint 2。`experiments/networking/` 继续承担：

1. mDNS 收发端，用来验证局域网或校园 Wi-Fi 上的 `224.0.0.251:5353` 是否能通。
2. TCP 吞吐测试端，用来粗测两台机器之间应用层 TCP 发送/接收速度。

如果你要看当前计算基准相关的目录，请分别看：

- `../../compute_node/input_matrix/README.md`
- `../../compute_node/performance_metrics/README.md`

`experiments/networking/` 只保留网络连通性和吞吐实验，不负责共享数据集生成或计算性能评分。

## 文件

- `receiver.py`: mDNS 接收端。加入组播组，收到目标服务的 PTR 查询后，回一个包含 `PTR + SRV + TXT + A` 的单播响应。
- `sender.py`: mDNS 发送端。从临时 UDP 端口发一个 PTR 查询，然后等待响应。
- `mdns_minimal.py`: mDNS 两端共用的最小 DNS/mDNS 编解码和 socket 帮助函数。
- `tcp_receiver.py`: TCP 吞吐测试接收端。监听一个 TCP 端口，接收测试流量并回传统计。
- `tcp_sender.py`: TCP 吞吐测试发送端。连接到 `tcp_receiver.py`，支持单流或多流并发发送，并打印两端统计。
- `tcp_speed.py`: TCP 两端共用的控制消息和速率格式化帮助函数。
- `zmq_receiver.py`: ZeroMQ 吞吐测试接收端，依赖 `pyzmq`。
- `zmq_sender.py`: ZeroMQ 吞吐测试发送端，依赖 `pyzmq`。

默认 mDNS 服务名是 `_superweb-cluster._tcp.local.`，和主项目 discovery 的服务类型一致。

## mDNS 用法

在接收端机器上：

```bash
python3 experiments/networking/receiver.py --name dorm-node --port 52020 --once
```

在发送端机器上：

```bash
python3 experiments/networking/sender.py
```

如果你想显式指定 Wi-Fi 的本地 IPv4：

```bash
python3 experiments/networking/receiver.py --interface-ip 10.23.44.193 --host-ip 10.23.44.193 --once
python3 experiments/networking/sender.py --interface-ip 10.23.135.240
```

### mDNS 结果判断

- `sender.py` 打印出 `service endpoint x.x.x.x:port`，说明 mDNS 查询和回包都成功了。
- `receiver.py` 一直没看到 `received matching query`，通常表示组播查询没到达这台机器。
- `receiver.py` 看到了查询，但 `sender.py` 没收到回包，通常表示单播返回路径或校园网终端互访被拦了。

## TCP 吞吐测试用法

在接收端机器上先启动：

```bash
python3 experiments/networking/tcp_receiver.py --bind 0.0.0.0 --port 52021 --once
```

在发送端机器上连接它：

```bash
python3 experiments/networking/tcp_sender.py 192.168.1.23 --port 52021 --duration 10
```

如果你想模拟 `iperf3 -P 4` 这种多流并发，可以这样：

```bash
python3 experiments/networking/tcp_sender.py 192.168.1.23 --port 52021 --duration 15 --streams 4 --chunk-size 1MiB --send-buffer 4MiB
```

如果你想固定测试总数据量，也可以这样：

```bash
python3 experiments/networking/tcp_sender.py 192.168.1.23 --port 52021 --bytes 1GiB
```

一些常用参数：

- `--chunk-size 1MiB`: 调大发送块大小，减少 Python 循环开销。
- `--send-buffer 4MiB`: 调整发送端 `SO_SNDBUF`。
- `--recv-buffer 4MiB`: 调整接收端 `SO_RCVBUF`。
- `--streams 4`: 同时开 4 条 TCP 流，适合排查“单流吃不满链路”的情况。
- `--progress-interval 0.5`: 更频繁打印进度。

### TCP 结果判断

- `sender.py` 和 `receiver.py` 的 mDNS 输出只回答“能不能发现”，不回答“TCP 能跑多快”。
- `tcp_sender.py` 会打印发送侧统计和接收侧统计；接收侧统计更接近真实链路吞吐。
- 如果 `--streams 1` 只能跑到几百 Mbps，而 `--streams 4` 明显更高，说明瓶颈更像是“单条 TCP 流吃不满链路”，这时应用层并发传输是可行方向。
- 如果想测反方向速度，把两台机器角色对调再跑一遍，因为很多 Wi-Fi/路由场景上下行并不完全对称。
- 这套脚本测的是 Python 应用层 TCP 吞吐，不是网卡理论线速；如果你想做更标准的基准，`iperf3` 会更专业。

## ZeroMQ 对比测试

优先复用项目根目录下的 `.venv`。如果你已经跑过 `bootstrap.py`，这个环境通常已经准备好了。

如果你只是单独运行这里的 ZeroMQ 对比测试，也可以手动准备：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install pyzmq
```

在接收端机器上先启动：

```powershell
.\.venv\Scripts\python experiments\networking\zmq_receiver.py --bind 0.0.0.0 --port 52041 --rcvbuf 4MiB
```

在发送端机器上连接它：

```powershell
.\.venv\Scripts\python experiments\networking\zmq_sender.py 192.168.1.23 --port 52041 --duration 15 --chunk-size 1MiB --sndbuf 4MiB
```

说明：

- 这里用的是 ZeroMQ `PAIR`，只适合一对一 benchmark，优点是收发协议很小，便于先看“换成消息库以后”吞吐有没有变化。
- ZeroMQ 发送侧可能因为内部队列而看起来特别快，所以对比时优先看 `zmq_receiver.py` 打出来的吞吐。
- 如果 ZeroMQ 明显比当前 TCP 版更好，说明你的负载可能更适合“消息分帧 + 库内缓冲”这种风格。
- 如果结果差不多甚至更低，也很正常，因为 ZeroMQ 主要解决的是消息模式、队列和连接管理，不保证一定比原始 TCP 更快。
