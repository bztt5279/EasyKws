# 特征预处理

[文档首页](../README.md) · [操作教程](../tutorial.md)

当前模块提供 clean 音频及可选离线增强到原始 fbank 特征的流程，以及仅用训练集拟合的 CMVN 统计。默认仍为完整 clean 音频；增强配置与资源说明见[离线增强](augmentation.md)。产物可交给[训练入口](training.md)；运行预处理本身不产生模型，也不验证唤醒效果。

## 安装与命令

在仓库根目录、独立虚拟环境中安装：

```powershell
python -m pip install -e '.[preprocess,dev]'
python -m pytest
```

项目声明 Python >=3.10；当前实测环境仅为 Windows Python 3.12。声明范围不等于逐版本验证，Python 3.14 不在当前测试覆盖内，预处理可选依赖的 wheel 可用性也需另行确认。

先用 prep 生成 prepared bundle，再执行预处理。以下路径均为本地示例，输出目录必须尚不存在，且位于 prepared 和音频根目录之外：

```powershell
easykws-preprocess plan --prepared ../easykws-demo-prepared --audio-root ../easykws-demo-recordings/audio --config configs/preprocess.example.json
easykws-preprocess run --prepared ../easykws-demo-prepared --audio-root ../easykws-demo-recordings/audio --config configs/preprocess.example.json --output ../easykws-demo-features
```

也可使用 `python -m easykws.preprocess` 代替 `easykws-preprocess`。两个子命令均要求 `--prepared`，v1 提供 `--audio-root`，v2 提供 `--path-map`，接受可选的 `--config`；`run` 还要求 `--output`。`plan` 验证输入并输出 JSON 摘要，不写特征；它仍会读取、解码全部纳入的音频，启用增强时还会逐条执行波形操作以核对长度，并需要预处理依赖。`run` 输出 JSON 结果，失败返回非零状态。

## 输入身份与格式策略

输入需包含合法的 `READY.json`，且不能有 `FAILED.json`。程序核对其中列出的 `train.jsonl`、`val.jsonl`、`test.jsonl`、`excluded.jsonl`、`config.json` 的 SHA256，检查完整清单、配置身份、行字段、纳入与排除规则、样本关系及划分统计，并按 prep 配置重建划分进行核对。对纳入样本，还会检查 WAV 文件字节哈希、PCM 哈希和音频元数据。音频路径必须位于指定根目录内；排除样本也需路径安全，但不要求其音频存在。

这些检查能发现输入相对记录发生变化，不能证明记录最初来自可信来源、已获使用许可或人工标注正确。`READY.json` 中的 `purpose` 按纳入样本的 `origin` 得到：包含合成样本时为 `synthetic-test`，否则为 `recorded-data`。`recorded-data` 只描述输入类型，不能据此推断模型用途、发布许可或已达到 release 条件。

格式转换必须显式选择，示例配置默认是：

```json
{
  "schema": "easykws-preprocess-config/v1",
  "channel_policy": "reject",
  "resample": false
}
```

默认拒绝多通道与非 16 kHz 输入。确有需要时，将 `channel_policy` 改为 `mean`，以 float32 通道平均转单声道；将 `resample` 设为 `true`，以多相重采样转到 16 kHz。未知字段、无效策略或非布尔的 `resample` 都会报错。程序拒绝空音频、非有限值、触及满幅阈值、处理后全零和不足 400 点的音频；不自动补零，也不做峰值归一化。满幅拒绝是质量复核门槛，不能代替完整的音频质量检测。

## 数值契约与产物

前端调用 `torchaudio.compliance.kaldi.fbank`，固定 16 kHz、40 个 mel bin、25 ms 帧长、10 ms 帧移。输入浮点波形乘以 `32768.0` 后送入 fbank；使用 Povey 窗、预加重 0.97、去直流、`snip_edges=true`、`dither=0`，其余显式参数记录在 `frontend.py` 和产物元数据中。有效长度为 N 时，帧数为 `1 + (N - 400) // 160`。

每条样本保存为未做 CMVN 的 float32 `[frames, 40]` 数组，路径为 `features/SHA256(sample_id 的 UTF-8 字节).npy`。哈希文件名用于避免大小写或平台保留名称冲突，样本身份仍由索引保存。写入后以 `allow_pickle=False` 回读，检查数组类型、形状和数值一致性。

`cmvn.json` 只累计实际写出的 train 特征帧，包括 clean、扩充样本和启用的增强变体，以 float64 计算均值、方差与逆标准差，方差下限为 `1e-10`；val/test 不参与拟合。缓存的 `.npy` 保持未归一化，当前训练模型在内部应用一次 CMVN；使用者自行调用特征时应避免重复归一化，也不能用验证集、测试集重新拟合统计。

输出还包含各 split 的 JSONL、`excluded.jsonl`、本次 `config.json`、`prep_config.json`、`source_ready.json` 和 `plan.json`。索引保留样本字段及转换信息，并加入特征路径、哈希、帧数和维度。元数据记录依赖版本、平台、prep/preprocess 源码身份哈希、输入 READY 身份、前端参数和统计汇总。它们用于追溯，不能替代可信来源证明。

`run` 先完成输入规划，再创建输出目录。写入后核对产物哈希，并复核 prep 输入与磁盘上的源码身份未变，最后通过临时文件替换发布 `READY.json`。源码哈希用于发现处理过程中发生的文件修改，不能证明运行进程未被篡改。创建目录后的执行失败会尽力写 `FAILED.json`，可能留下部分文件；规划阶段失败则通常没有输出目录。缺少 READY、存在 FAILED 或内容未经核对时，都不应把目录当作完成产物；重试需使用新目录。

## 验证范围

小型合成信号测试用于检查格式策略、输入篡改拒绝、划分身份保留、train-only CMVN、产物回读及失败行为。与留存前端做合成输入数值对照，只能支持指定输入、参数与依赖环境下的数值兼容性；不能证明真实录音上的识别效果、误唤醒率、泛化、实时性能或硬件部署。当前适配代码为新编写实现，不能由数值接近反推历史上游版本。

## v2 兼容说明

v1 prepared bundle 继续受支持。v2 的输入、路径映射及产物规则见 [v2 数据契约](data_contract_v2.md)；本机 `path-map` 不写入产物。
