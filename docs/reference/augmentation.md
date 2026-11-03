# 离线数据增强

[文档首页](../README.md) · [操作教程](../tutorial.md)

增强在 fbank 提取前作用于浮点波形，随后缓存原始 float32 特征。默认关闭增强、保留完整音频。启用后仅对选中的 train 样本生成变体，val/test 保持 clean。这里的 gain 是离线幅度乘法；目标音量的训练时在线波形增强尚未实现。

## 算法与历史行为

| 操作 | 数值语义 |
| --- | --- |
| clean | 不做增强，也不暗中归一化峰值。 |
| gain | 波形乘以配置的幅度系数；不是目标 dBFS 归一化。 |
| speed | 多相重采样，同时改变时长和音高；不是保音高变速。 |
| noise | 按当前波形 RMS 和指定 SNR 混合，短噪声重复后取段，记录选段和重复次数。 |
| RIR | 选择显式通道，重采样后定位绝对值主峰；完整卷积去掉主峰前传播延迟，保留尾音；用前原长度片段的 RMS 恢复输入音量。 |
| filter | 四阶 Butterworth 带通，使用二阶节与 `sosfilt`，不是零相位滤波。 |
| peak | 对有操作的变体统一做峰值保护，记录保护前峰值和实际安全增益。 |
| 历史 crop | 仅在显式兼容配置启用，记录保留区间、时长与丢弃量；不能借多跑 epoch 恢复已经丢弃的内容。 |

历史参数范围为 gain `[0.3,1.5]`、speed `[0.8,1.2]`、noise SNR `[5,20]` dB、滤波低截止 `[50,300]` Hz / 高截止 `[4000,7600]` Hz，增强峰值上限 `0.98`。这些是兼容参数，不是已验证适合所有唤醒词的推荐训练配方。历史每条 clean + 4 regular + 3 challenge 的数量也不是新项目的默认数量。

regular 每个变体随机选择一个操作；challenge 选择两个不同操作，按选中的顺序执行。随机性绑定全局种子、稳定样本 ID 与变体编号。修改清单遍历顺序不会改变同一 ID 的操作；新 ID 不承诺重放旧工程 UID 的历史随机序列。

## 配置与运行

在现有 preprocess 配置中添加 `augmentation`。下面是无需额外资源的最小演示，每个匹配的训练样本保留一份 clean，另产生一份单操作、一份双操作变体：

```json
{
  "schema": "easykws-preprocess-config/v1",
  "channel_policy": "reject",
  "resample": false,
  "augmentation": {
    "schema": "easykws-offline-augmentation/v1",
    "seed": 666,
    "selectors": {
      "groups": ["target", "background"],
      "labels": [0, -1],
      "origins": ["test-fixture"]
    },
    "variants": {"regular": 1, "challenge": 1},
    "kinds": ["gain", "speed", "filter"]
  }
}
```

此处 `test-fixture` 对应 v1 示例显式升级后的 v2 测试音调。真实录音使用清单中的 `recorded`，已导入 VC 为 `vc`。groups、labels、origins 三个选择条件同时满足才增强，不能照抄示例后假定已匹配自己的样本。变体数必须显式提供；省略整个 `augmentation` 时仍执行原 clean 流程。

仓库中的 [preprocess.augmentation.example.json](../../configs/preprocess.augmentation.example.json) 提供了上述配置。v1 清单仍使用 `synthetic` 来源名；可先按 [v2 数据契约](data_contract_v2.md) 显式升级再使用这个 v2 示例。

保存为仓库外的配置文件，再运行现有命令：

```sh
easykws-preprocess plan --prepared ../demo-prepared-v2 --path-map ../paths.json --config ../augmentation.json
easykws-preprocess run --prepared ../demo-prepared-v2 --path-map ../paths.json --config ../augmentation.json --output ../demo-augmented-features
```

`plan` 实际读取波形并逐条执行增强来核对长度和操作信息，不写特征；`run` 重新按同一操作计划执行并核对。两者都不是只数文件的快速估算。

使用噪声或 RIR 时，在 `kinds` 中加入相应名称，并声明资源池，例如 `"noise":{"resource_ids":["noise_a"],"snr_db":[5,20]}`、`"rir":{"resource_ids":["room_a"],"channel":0}`。资源 ID 必须存在于 v2 清单中；通道从 0 开始。不会扫描目录自动猜测资源。

历史短裁剪需显式设置 `crop`，例如 `{"mode":"replace-clean","selectors":{"groups":["background"],"labels":[-1],"origins":["public"]},"seconds":[2,3],"only_if_longer_than":3}`。它替换匹配 train 样本的 clean，并把同一裁剪作为其所有变体的第一步；若只需要历史单份短裁剪，将该次 regular/challenge 数量都设为 0。裁剪坐标位于前端转换后的 16 kHz 波形，原录音采样率和点数另由源记录保存。默认 `crop:null`，不裁短。

## 资源与产物

噪声和 RIR 使用 v2 清单中的独立资源条目，记录资源 ID、来源、版本、许可说明、相对路径、哈希与划分。配置引用逻辑 ID，本机根目录仍通过 `--path-map` 提供，不写入产物。原始录音、噪声和 RIR 均由使用者自行准备；仓库不包含历史资源音频。

已知 val/test 内容、关联说话人或谱系不能作为训练增强资源。文件哈希及完全相同 PCM 的检查不能识别所有重编码、变速或同源音频，资源来源仍需如实填写。测试资源的 `synthetic-test` 用途会向产物传播；该标记用于工具链内防止静默丢失，不是来源认证。

已排除的 val/test 样本和测试夹具也参与资源检查：若原 WAV 可读且与声明哈希一致，提取其 PCM 身份，拒绝将相同 PCM 的 val/test 音频重新封装后用作资源；相同 PCM 的测试夹具继续传播测试用途。可读文件的身份不符会报错。排除项不要求全部存在：`augmentation.excluded_identity_checks` 逐条记录 `verified-pcm`、`unavailable`（缺失或无法读取 PCM）或 `unverified-identity`（没有声明哈希）。后两类没有完成 PCM 匹配，不能将整个运行解释为已排除所有同源泄漏。

每个变体保留原样本身份、操作链、实际参数、资源身份、波形长度、帧数及特征哈希。CMVN 只累计本次实际写出的 train 特征，包括 clean、已导入扩充样本和增强变体；缓存不应用 CMVN。val/test 不参与统计。

默认从波形到特征保留全长有效内容；fbank 的 25 ms / 10 ms 帧边界仍可能舍去不足一帧的末尾点，这与短裁剪不同。RIR 尾音和变速后的实际长度参与帧数统计。数值对照比较同一阶段的浮点波形与特征，不把另存 PCM16 WAV 后的量化误差误认为算法不一致。

不提供增强缓存复用或断点续跑。输出目录必须是新目录；执行失败的部分产物没有可消费的 READY。小型合成信号验证只能证明数值实现和流程契约，不能证明增强提升了唤醒效果。
