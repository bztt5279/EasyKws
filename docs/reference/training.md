# 有限步训练与轮末恢复

[文档首页](../README.md) · [操作教程](../tutorial.md)

训练消费 `easykws-preprocess run` 生成的特征目录，先核验 READY、特征文件、形状、有效长度、CMVN 和用途信息。输入缓存保持只读；默认读取完整有效帧，不裁短、不因为样本长而静默排除。标签继续使用 `0` 表示目标词、`-1` 表示负类。

当前实现范围为 CPU、FP32、`num_workers=0`。模型、训练步骤与验证损失可运行；训练后通过[流式校准](calibration.md)选择候选和阈值，再按[导出与部署说明](export.md)打包和加载。小型合成信号训练仅验证更新和恢复，不代表真实语音效果。

## 命令行入口

```sh
python -m pip install -e ".[training]"
easykws-train run --features ../demo-augmented-features --config configs/training.smoke.example.json --output ../demo-training
```

也可用 `python -m easykws.training` 代替 `easykws-train`。示例使用缩小模型、两个 epoch、每轮 10 次有放回抽样、batch_size=2，共 10 次优化更新；只适合验证接口。示例组名为 target/background，需要与输入索引的 group 相符，实际组必须显式提供权重。

验证轮末恢复时，从新的输出目录运行：

```sh
easykws-train run --features ../demo-augmented-features --config configs/training.smoke.example.json --output ../demo-resumed --stop-after-epoch 1
easykws-train resume --run ../demo-resumed --features ../demo-augmented-features
```

`--stop-after-epoch` 仅控制停止时点，不改总训练轮数或学习率配方。resume 默认使用运行保存的配置，也可显式传 `--config` 核对；输入目录可以搬迁，但内容身份必须一致。源码更新后不自动认可旧运行可精确恢复。

## 模型与特征增强

模型沿用 WeKWS 派生 DS-TCN 的结构及状态名称，输入为 40 维原始 fbank，CMVN 在模型内应用一次，之后是线性投影、因果深度可分离卷积和单关键词 sigmoid。来源边界见[模型来源说明](MODEL_PROVENANCE.md)。小型测试可以显式缩小 hidden_dim 和层数；这不是对历史模型效果的复现。

SpecAugment 在 train 读取缓存后复制特征，再做时间与频率遮挡，随后进入模型内 CMVN；val/test 不做遮挡，磁盘缓存不变，也不按每轮遮挡重新拟合 CMVN。它与预处理阶段的离线波形增强分别发生在不同阶段；训练时目标音量增强仍未实现。

损失及样本最大分只使用有效帧，不把 padding 纳入池化。保留历史网络中的 BatchNorm，因此不把“池化掩码正确”扩大解释为训练模式下任意 padding 都不影响网络统计。

## 一个 epoch 是多少数据

采样组直接取索引里的 `group`，权重由配置提供。默认模式为 `balance_unit=sample`、`replacement=true`：非空组 g 的总权重 W_g 平分给组内 n_g 条样本。空组不进入概率分母，零权重组不被抽取。负数、非有限权重或总有效权重为零会报错。

默认抽样次数为 `floor(N * sample_ratio)`，sample_ratio 默认为 1；也可显式给定 `draws_per_epoch`，两者不能同时指定。保留最后不足 batch_size 的 batch。一个 epoch 表示这些有放回抽样，并不保证遍历所有数据。

同组某条父录音增加 VC 或增强变体后，父录音的总曝光概率可能提高；这不是父级均衡。比较不同扩充规模时可显式固定 draws_per_epoch。报告区分理论概率、期望次数和实际抽样序列，记录样本、组、谱系根的曝光与实际读入帧数；短轮次的实际比例不必等于理论比例。

## 保存和恢复

新运行拒绝覆盖已有非空目录。训练保存初始状态及每个完整 epoch 的候选 checkpoint；只承诺从完整轮末恢复，轮中失败后重做该轮。没有完整轮时从已记录的初始状态重做，不把部分轮次当作完成。

保存内容包括模型、优化器、调度器、采样生成器、Python/NumPy/torch 随机状态、完成轮数与全局步数。普通加载始终使用 `weights_only=True`，没有不受限加载回退。恢复前核对配置、数据、CMVN、前端、模型、源码及执行环境身份；换配方应新建运行，不能伪装成连续恢复。

运行状态区分 running、completed、failed、interrupted。失败尝试留档；残留 running 不表示成功。候选 checkpoint 保留，`best_val_loss` 仅描述验证损失最优的候选。校准读取训练产物但不回写训练状态；训练 `status.json` 的 `selected_checkpoint` 保持未选定，实际选择另存于校准输出的 `selected_checkpoint.json`。测试集不能参与候选或阈值选择。

恢复要求状态文件包含有效的 `last_checkpoint_sha256`，不得通过缺失哈希跳过核验。日志创建失败也进入失败处理：失败原因写入独立的 `status.json`，不依赖日志能够创建或写入；若状态文件本身也不可写，则无法保证落盘记录。

验证损失按样本数加权，包含最后不足一批的样本；相等时保留较早 epoch。`status.json`、每次尝试的 `attempt_*.jsonl` 和完整轮 checkpoint 记录步数、实际曝光与恢复状态。测试特征只做完整性核验，不计算测试指标。数据或配置预检失败时可能尚未建立运行；这不属于可恢复的已完成保存点。

## 旧 checkpoint 的显式迁移

`easykws-train migrate-checkpoint` 仅面向使用者确认可信的旧输入，必须显式确认信任。该命令的不受限 pickle 加载可能执行输入中的代码；普通训练或加载失败时不会自动调用它。只对自己的可信文件使用，文件 SHA256 相同只能证明内容一致，不能证明来源可信。

```sh
easykws-train migrate-checkpoint --input ../trusted-legacy.pt --output ../converted.pt --trust-legacy-input
```

迁移格式不等于补齐旧训练的身份和随机状态，也不自动授予恢复新运行或发布模型的资格。未提供必要运行契约的旧文件不能作为连续训练恢复证据。
