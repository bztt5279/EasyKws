# EasyKws 操作教程：从录音到部署

[文档首页](README.md) · 采集前读[录制指南](recording_guide.md) · 方法解释见[技术报告](technical_report.md)

本教程带你把「自己选词，和几位朋友录制」变成一套能运行的唤醒词方案。先用合成演示确认软件能运行，再替换为自己组织的录音，并用公开数据补充普通语音和背景条件。

所有命令从仓库根目录执行，假设已经按[安装说明](install.md)激活环境并安装 `preprocess` 依赖；用麦克风时另需 `listen`。

命令中的 `../my-kws-data`、`../my-kws-runs` 都是仓库外的示例目录，可按本机情况调整。每次运行使用新输出目录，不覆盖旧结果。README 和教程中的配置是起点，不是已经验证过识别效果的训练配方。

## 1. 先跑一遍合成演示

```sh
easykws-smoke --output ../easykws-smoke-run
```

该命令生成全部输入，无需下载语音或接麦克风。完成后打开输出目录中的 `summary.json`：

| 阶段 | 查看什么 |
| --- | --- |
| `generate`、`prepare` | 生成录音和准备输入的数量 |
| `expand` | 模拟 VC、公开负样本导入及审核结果 |
| `preprocess` | 五类增强、特征帧数与 train-only CMVN |
| `train` | 短训练是否结束，是否保存候选模型 |
| `calibrate`、`export` | 是否选择候选、生成受限测试部署包 |
| `replay` | 新进程中各会话的帧、事件和时钟核对 |

最后检查 `source_inputs_unchanged` 与 `conclusion`。演示产物带 `synthetic-test` 标记；不能改标记后当作真实唤醒模型使用。完整演示是一项接口检查，不提供唤醒率或现场误触发率。

## 2. 准备自己的录音与清单

按[录制指南](recording_guide.md)自录目标词正样本和近似、近音或不完整对照词，完成试录、审核与切分。普通讲话、音乐、环境噪声和 RIR 优先使用合适的公开数据资源，不需要参与者另录。起步可以使用较简单的 v1 JSONL：一行对应一个 WAV，路径相对音频根目录。

```jsonl
{"sample_id":"speaker_001_target_001","speaker_id":"speaker_001","source_recording_id":"speaker_001_session_01","audio_path":"speaker_001/target_001.wav","label":0,"group":"target","expected_phrase":"你好小A","review_status":"pass","origin":"recorded"}
{"sample_id":"speaker_001_negative_001","speaker_id":"speaker_001","source_recording_id":"speaker_001_session_02","audio_path":"speaker_001/negative_001.wav","label":-1,"group":"background","expected_phrase":"你好小B","review_status":"pass","origin":"recorded"}
```

上面只有一个人的两条示例，不能单独完成 train/val/test 划分。实际需提供足够的独立单元和正负样本；当前接口至少需要 3 个独立单元，并且默认要求每个划分含正负类。只有一个人时，不能通过把不同场次伪装成不同说话人来运行完整默认管线；实际说话人泛化也不能由这个最低要求保证。

同一人的编号保持一致；同一原始录音切出的样本使用相同 `source_recording_id`。审核未通过的样本保留在清单中，填相应状态，不靠删除关联记录解除同源关系。字段细节见[清单参考](reference/recording_manifest.md)。

将清单保存为 `../my-kws-data/recordings.jsonl`，音频放入 `../my-kws-data/audio/`。复制 `configs/prepare.example.json` 为自己的准备配置。默认按独立单元 60%/20%/20% 划分；若已确定人员分配，用 `speaker_splits` 显式列出全部实际说话人的 train/val/test 归属。

## 3. 预览划分，再生成索引

```sh
easykws-prep plan --manifest ../my-kws-data/recordings.jsonl --audio-root ../my-kws-data/audio --config configs/prepare.example.json
easykws-prep run --manifest ../my-kws-data/recordings.jsonl --audio-root ../my-kws-data/audio --config configs/prepare.example.json --output ../my-kws-runs/prepared-01
```

若使用自己的配置，替换两个命令的 `--config`。`plan` 会读取并核对录音，但不写产物；确认人员归属、排除原因和类别覆盖后再执行 `run`。

**停下来检查：** 打开 `train.jsonl`、`val.jsonl`、`test.jsonl`、`excluded.jsonl`，核对人和原始录音的归属。最后存在 `READY.json` 表示本步骤完成；它只证明产物符合程序检查，不代表录音质量已经由人审核。

## 4. 需要扩充时，先升级多来源清单

只用自录语音可以跳到第 5 节。需要外部 VC、公开负样本或噪声/RIR 资源时，使用 v2 清单；不要手工修改已经发布的 prepared 索引。

可以把第 2 节的 v1 清单显式升级到新目录，使用自己的准备配置时同时提供 `--config`：

```sh
easykws-prep upgrade-v1 --manifest ../my-kws-data/recordings.jsonl --audio-root ../my-kws-data/audio --output ../my-kws-runs/upgraded-01
```

按升级输出核对 dataset 来源、版本、许可说明及固定划分。为本例创建 `../my-kws-data/paths.local.json`，内容为 `{"recordings":"../my-kws-data/audio"}`；相对根目录按当前工作目录解析。然后生成 v2 prepared：

```sh
easykws-prep plan-v2 --manifest ../my-kws-runs/upgraded-01/manifest.json --config ../my-kws-runs/upgraded-01/config.json --path-map ../my-kws-data/paths.local.json
easykws-prep run-v2 --manifest ../my-kws-runs/upgraded-01/manifest.json --config ../my-kws-runs/upgraded-01/config.json --path-map ../my-kws-data/paths.local.json --output ../my-kws-runs/base-v2
```

v1 升级不会自动确定外部新来源；新增数据集时按[显式升级与路径映射](reference/data_contract_v2.md)补齐声明。外部根目录映射留在仓库外。已有 v1 索引与不同配置的升级输入可能得到不同划分，采用新集合前应重新核对，不能混用两个版本的评估记录。

VC 需要先在外部生成和听审，再通过扩充入口导入：

```sh
easykws-prep expand-plan --prepared ../my-kws-runs/base-v2 --candidates ../my-kws-data/candidates.json --path-map ../my-kws-data/paths.local.json --qc ../my-kws-data/qc.json
easykws-prep expand-run --prepared ../my-kws-runs/base-v2 --candidates ../my-kws-data/candidates.json --path-map ../my-kws-data/paths.local.json --qc ../my-kws-data/qc.json --output ../my-kws-runs/expanded-01
```

这些命令以**已按 v2 参考生成的 base-v2 及候选文件**为前提，并不自动生成 VC 音频。候选清单与 QC 的完整示例见[扩充导入参考](reference/data_expansion.md)。检查输出 `report.json` 的 accepted、flagged、failed、pending、duplicate、limited，实际可交接的 prepared 位于 `expanded-01/prepared/`。

普通语音可参考 [Mobvoi](https://www.openslr.org/87/)，语音、音乐和噪声可参考 [MUSAN](https://www.openslr.org/17/)，脉冲响应与噪声可参考 [RIRS_NOISES](https://www.openslr.org/28/)。先核对许可与用途。仓库仅提供 Mobvoi 的[本机元数据适配器](reference/public_negatives.md)，其他资源需自行下载、组织和声明，不会自动下载。导入前先明确每个类别相对于自己目标词的含义。不要直接把所有公开关键词映射为负类。VC 参考音色、噪声和 RIR 也不能取自评估数据。

## 5. 生成特征与训练集 CMVN

建议先用 clean 流程完成一次小规模检查：

```sh
easykws-preprocess plan --prepared ../my-kws-runs/prepared-01 --audio-root ../my-kws-data/audio --config configs/preprocess.example.json
easykws-preprocess run --prepared ../my-kws-runs/prepared-01 --audio-root ../my-kws-data/audio --config configs/preprocess.example.json --output ../my-kws-runs/features-01
```

消费 v2 prepared 时，将 `--audio-root` 换成 `--path-map ../my-kws-data/paths.local.json`。需要离线增强时，参考 `configs/preprocess.augmentation.example.json`，按自己实际的 group、label、origin 及资源 ID 配置，不要照抄测试夹具的选择条件。具体见[增强参考](reference/augmentation.md)。

**停下来检查：** `plan.json` 中的源样本数、输出变体数、每侧帧数应符合你的计划。特征为未做 CMVN 的 `[帧数, 40]` float32 数组；`cmvn.json` 的 `fit_split` 应为 `train`，统计帧数应对应实际输出的训练特征。默认保留全长，val/test 不生成训练增强变体。

格式不同会明确报错。若原始 WAV 不是 16 kHz 单声道，在自己的预处理配置中显式启用重采样或通道平均，具体见[前端参考](reference/preprocessing.md)。`plan` 也会逐条读取音频，大数据集需要时间，不能把它当成只读文件名的快速统计。

## 6. 先短训练，再确定正式配方

先用小模型检查自己的数据能否进入训练：

```sh
easykws-train run --features ../my-kws-runs/features-01 --config configs/training.smoke.example.json --output ../my-kws-runs/training-check-01
```

示例只适用于 `target`、`background` 两组，模型 hidden_dim=8、1 层，2 个短 epoch，共 10 次更新。它检查流程，不是正式训练配置。若清单组名不同，先在自己的配置里改 `sampler.group_weights`。

正式训练前，将配置复制到仓库外，逐项决定模型规模、训练轮数、抽样次数、组权重和增强强度。当前模型默认结构是 hidden_dim=256、4 层、kernel_size=8；这描述实现默认值，不是适合任意数据的最优配置。可将这些值填入自己的 `model` 节。`draws_per_epoch` 与 `sample_ratio` 只能选一个：想按数据规模抽样时删除前者、显式设置后者。较长录音和较多抽样会增加内存与耗时，先小规模运行并记录成本。

**停下来检查：** `status.json` 应记录已完成轮数、候选 checkpoint、预期和实际曝光；`attempt_*.jsonl` 有抽样与训练记录。仅 loss 下降或某轮训练准确率高，不表示实际能唤醒。最终候选需要后续校准选择。

轮末恢复的使用方式如下，新运行目录不能覆盖已有内容：

```sh
easykws-train run --features ../my-kws-runs/features-01 --config configs/training.smoke.example.json --output ../my-kws-runs/resume-check-01 --stop-after-epoch 1
easykws-train resume --run ../my-kws-runs/resume-check-01 --features ../my-kws-runs/features-01
```

中断的一轮从上一完整轮末重做；改数据、代码、环境或配方后不自动视为同一次可恢复运行。详细说明见[训练参考](reference/training.md)。

## 7. 用独立连续录音选择模型与阈值

以下步骤需要在真实使用条件下另行准备连续校准录音，是独立于参与者训练采集的任务；训练切片和公开素材都不能代替。按[录制指南第 7 节](recording_guide.md#7-连续校准与最终测试要另行准备)安排验证人员和场次。

先查看整句评分可用：

```sh
easykws-train evaluate --run ../my-kws-runs/training-check-01 --features ../my-kws-runs/features-01 --output ../my-kws-runs/evaluation-01 --thresholds 0.1 0.5 0.9
```

这个结果不决定部署阈值，也不是连续 FAH。真正校准需要 `sessions.json`：在[会话模板](reference/calibration.md#会话清单)中填写真实路径、说话人、样本数、文件 SHA256、正例窗口和负区间。可用下面的脚本读取文件身份，结果再填回模板；区间必须由人工审核，不能由脚本猜测。

```python
from pathlib import Path
import hashlib
import wave

path = Path("../my-kws-data/calibration/session-01.wav")
with wave.open(str(path), "rb") as wav:
    print("rate / channels:", wav.getframerate(), wav.getnchannels())
    print("duration_samples:", wav.getnframes())
print("sha256:", hashlib.sha256(path.read_bytes()).hexdigest())
```

再复制 `configs/calibration.example.json` 为自己的配置，填入本次运行实际存在的候选文件名、阈值网格与选择约束。示例的 0.1/0.5/0.9 网格和约束只展示格式，不能保证找到合适阈值。

为校准、导出和后续加载保持相同的执行环境。线程设置在启动各命令前统一：

```powershell
# PowerShell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
```

```sh
# bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

在所用终端设置一次，再执行：

```sh
easykws-train calibrate --run ../my-kws-runs/training-check-01 --features ../my-kws-runs/features-01 --manifest ../my-kws-data/calibration/sessions.json --config ../my-kws-data/calibration.local.json --output ../my-kws-runs/calibration-01
```

**停下来检查：** 报告是否有可行候选、实际观测的正例数与负时长是多少。`selected_checkpoint.json` 记录最终选择；无可行候选时不会兜底选一个阈值。先检查数据、标注、候选和约束，保留失败结果，不手改选择记录。

## 8. 导出，再在独立入口加载

```sh
easykws-train export --run ../my-kws-runs/training-check-01 --features ../my-kws-runs/features-01 --calibration ../my-kws-runs/calibration-01 --manifest ../my-kws-data/calibration/sessions.json --output ../my-kws-runs/package-01
```

导出时仍需能读取原校准录音，用于重新核验与回放。成功后包内有 model、CMVN、threshold、calibration、manifest 和 READY 六个文件。包生成后，加载不再依赖原训练目录和校准音频。

先核验包，不打开麦克风：

```sh
easykws-listen --model ../my-kws-runs/package-01 --check
```

若它是合成演示或诊断子集产物，默认拒绝加载。仅做接口检查时可显式加 `--allow-test-model`，它不会绕过模型内容或执行环境校验。

文件流式推理示例，保存为本机脚本后在相应工作目录执行：

```python
import soundfile as sf
from easykws.runtime import load_model

audio, rate = sf.read("../my-kws-data/calibration/session-01.wav", dtype="float32")
assert rate == 16000 and audio.ndim == 1
detector = load_model("../my-kws-runs/package-01", apply_execution_profile=True)
for start in range(0, len(audio), 512):
    result = detector.push(audio[start:start + 512])
    for event in result["events"]:
        print(event)
detector.flush()
```

上面的音频用于确认调用方式。最终效果测试应换成留出的新数据，制定独立协议。模型、前端、触发规则或执行环境改变时，需要重新校准或提供兼容性验证，不能直接沿用旧阈值。

## 9. 最后才测试真实麦克风

按[安装说明](install.md)安装 `listen` 依赖，再选择实际输入设备：

```sh
easykws-listen --list-devices
easykws-listen --model ../my-kws-runs/package-01 --device 0
```

设备号以本次列表为准，不能照抄 `0`。当前设备必须能直接以 16 kHz 单声道打开；不支持时会报错，不自动重采样。Ctrl+C 结束，默认不落盘音频。队列溢出或断流意味着连续性丢失，程序会停止，本次不能当作完整连续测试。

麦克风代码已做模拟设备测试，真实硬件仍待验证。一次能打开设备、一次触发成功、长期误触发达标是三个不同结论。流式 API 和设备行为细节见[运行参考](reference/export.md)。

## 10. 出错时从哪里查

| 现象 | 先检查 |
| --- | --- |
| 输出目录已存在 | 用新目录，不删除失败证据后假装首次成功 |
| 划分缺正例或负例 | 核对人员、组覆盖和显式划分；不要伪造说话人 |
| WAV 格式或满幅错误 | 回看原始录音和前端设置，不只改扩展名或缩放失真文件 |
| 没有可行校准候选 | 先核对窗口与负时长，再分析模型和约束 |
| 执行环境不匹配 | 核对线程、精度、依赖、平台与源码；按[执行环境参考](reference/export.md#执行环境)处理 |
| 测试模型被拒绝 | 检查来源标记；接口演示显式允许测试用途，正式效果另行训练评估 |

理解异常背后的设计原因可继续读[技术报告](technical_report.md)，精确字段在[参考索引](README.md#需要时再查)中查找。
