# standalone_model

这个目录放了一套最小、自包含的 mDNS 收发端，用来单独验证局域网或校园 Wi-Fi 上的 `224.0.0.251:5353` 是否能通。

文件：

- `receiver.py`: 加入 mDNS 组播组，收到目标服务的 PTR 查询后，回一个包含 `PTR + SRV + TXT + A` 的单播响应。
- `sender.py`: 从临时 UDP 端口发一个 PTR 查询，然后等待响应。
- `mdns_minimal.py`: 两端共用的最小 DNS/mDNS 编解码和 socket 帮助函数。

默认服务名是 `_homecluster-hs._tcp.local.`，和主项目 discovery 的服务类型一致。

## 用法

在接收端机器上：

```bash
python3 standalone_model/receiver.py --name dorm-node --port 52020 --once
```

在发送端机器上：

```bash
python3 standalone_model/sender.py
```

如果你想显式指定 Wi-Fi 的本地 IPv4：

```bash
python3 standalone_model/receiver.py --interface-ip 10.23.44.193 --host-ip 10.23.44.193 --once
python3 standalone_model/sender.py --interface-ip 10.23.135.240
```

## 结果判断

- `sender.py` 打印出 `service endpoint x.x.x.x:port`，说明 mDNS 查询和回包都成功了。
- `receiver.py` 一直没看到 `received matching query`，通常表示组播查询没到达这台机器。
- `receiver.py` 看到了查询，但 `sender.py` 没收到回包，通常表示单播返回路径或校园网终端互访被拦了。

这个模型故意做得很小，便于你在学校 Wi-Fi 上直接区分“组播发不出/收不到”和“回包返不回来”。
