# EasyKws

**找几位朋友录些语音，训练属于自己的唤醒词。**

EasyKws 希望让每个人都能定制属于自己的唤醒词。你来选一个喜欢的词，和几位朋友录下目标词及容易混淆的对照词，再利用公开数据补充普通语音与背景声音，按教程完成训练和验证。

项目基于 PyTorch，将数据准备、增强、模型训练、流式校准、导出和推理串联起来。它以「你好小A」（A 读 /ēi/）为示例，面向想为自己的设备或应用定制唤醒词的个人和小团队。

项目提供代码和配置，不附带真人录音或预训练唤醒模型。更换唤醒词需要采集、标注和重新训练。

## 从哪里开始

| 你的目标 | 阅读入口 |
| --- | --- |
| 先知道整个项目怎么用 | [操作教程：从录音到部署](docs/tutorial.md) |
| 组织人录制自己的数据 | [数据集录制与标注指南](docs/recording_guide.md) |
| 理解方法、设计取舍与工程经验 | [技术报告](docs/technical_report.md) |
| 看字段、参数和命令细节 | [文档首页与参考索引](docs/README.md) |
| 核对上游、数据集和许可证 | [来源与许可证](docs/DATA_PROVENANCE.md) |

## 准备什么数据

采集重点是自己要使用的**目标唤醒词**，以及容易混淆的近音词、相似对照负词。普通语音、音乐、环境噪声和房间脉冲响应（RIR）可从适用的公开语料准备，不需要参与者把这些素材全部自录。公开素材不能代替另行安排的真实连续音频校准和现场验证。

本项目使用过的公开语音来源包括 **[MobvoiHotwords（SLR87，小问唤醒词数据集）](https://openslr.org/87/)**。官方页面分别提供音频和标注元数据；接入时需准备两部分，并明确哪些类别相对自己的目标词属于负类、哪些应排除。具体步骤见[公开负样本参考](docs/reference/public_negatives.md)。

起步可以邀请几位朋友一起录制：一部分人的录音用于训练，另外留出人员验证和测试，看看模型是否能识别没有参与训练的声音。具体怎样分组、每组录什么，见[录制指南](docs/recording_guide.md)。

## 系统怎样工作

```mermaid
flowchart LR
    A[自录目标词和相似负词 / 公开语料] --> B[清单 / 审核 / 隔离划分]
    V[外部生成的 VC 产物] --> B
    B --> C[训练集波形增强]
    C --> D[fbank / 训练集 CMVN]
    D --> E[DS-TCN 训练]
    E --> F[独立连续会话校准]
    F --> G[模型与触发配置导出]
    G --> H[流式推理 / 唤醒事件]
```

模型输出每一帧的目标词分数，触发器再根据连续确认、去重和冷却规则产生事件。训练后的模型与阈值需要共同校准，不能只拿训练日志里的准确率判断是否适合连续监听。

## 项目特点

- **把自定义唤醒词带到个人项目中**：自己选词，把自录工作集中在目标词和相似对照词，其他训练素材利用公开资源补充。
- **流程连贯**：录音清单、离线增强、训练、校准和部署包使用明确的输入输出，方便分阶段检查。
- **数据关系可追溯**：记录说话人、原始录音、派生样本和增强资源，检查划分冲突及已知重复内容。
- **校准贴近实际触发规则**：文件回放与运行时共用流式评分和触发器，区分片段分类与连续误触发。
- **便于复查实验**：保留候选模型、抽样曝光、轮末恢复状态与导出身份，减少模型、前端和阈值错配。
- **已有微控制器端侧实践**：既有 DS-TCN 方案已在 ESP32-S3 上实现板端特征、FP32 推理和事件触发，并完成固定音频对照与短时真实采音调度验证。

这些是实现和工程设计上的特点；项目当前没有公开的跨项目识别性能比较。

## ESP32-S3 端侧部署实践

这条方案已有独立的 ESP32-S3 部署工程：INMP441 麦克风通过 I2S 采音，在芯片上完成 fbank、DS-TCN 推理与唤醒事件判定，推理链路无需将音频送回电脑。采用 BatchNorm 折叠后的 FP32 权重与连续缓存，每 8 帧做一次模型推理，再逐帧执行触发规则。

既有记录包含固定 PCM 与 PC 参考的数值对照，以及真实麦克风的短时连续运行。现存一份 20 秒日志记录特征帧全部进入推理，无队列丢帧、I2S 读取错误或超时，模型推理 p99 约 50 ms/8 帧，支持该配置下的端侧实时计算可行性。

这项实践验证了微控制器上的本地计算路线；功耗、续航、远场与长期误触发仍需专门测量。板端工程尚未迁入当前主仓，当前 `export` 也不直接生成固件。实现思路、记录范围与局限见[技术报告](docs/technical_report.md#81-esp32-s3-端侧部署实践)。

## 快速运行合成演示

以下命令在仓库根目录执行。建议使用 Python 3.12 和独立虚拟环境，安装步骤见[安装说明](docs/install.md)。先安装 CPU 版配对依赖，再安装项目：


```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchaudio==2.6.0
python -m pip install -c configs/cpu-constraints.txt ".[preprocess]"
easykws-smoke --output ../easykws-smoke-run
```

输出目录必须是新的目录。程序会生成音调，完成数据准备、模拟扩充导入、五类增强、短训练、校准、导出和全新进程回放，并写出 `summary.json`。这是检查软件流程的演示，生成的模型没有可据此声称的语音识别能力。

准备真实训练时，从[录制指南](docs/recording_guide.md)开始，再按[操作教程](docs/tutorial.md)逐步执行。VC 引擎需要在外部运行，本仓库只导入其产物。

## 代码怎么读

所有主程序位于 `src/easykws/`：

| 模块 | 负责什么 |
| --- | --- |
| `prep/` | 清单、审核、划分，以及外部数据导入 |
| `preprocess/` | 波形增强、fbank 和 CMVN 统计 |
| `training/` | 模型、损失、采样、训练与恢复 |
| `streaming.py`、`trigger.py` | 音频缓存、连续模型评分与事件判定 |
| `calibration/`、`export/` | 候选与阈值选择，生成部署包 |
| `runtime/` | 部署包加载、流式 API 和麦克风入口 |

想理解算法，建议从 [model.py](src/easykws/training/model.py)、[loss.py](src/easykws/training/loss.py)、[streaming.py](src/easykws/streaming.py)、[trigger.py](src/easykws/trigger.py) 开始，配合[技术报告中的代码地图](docs/technical_report.md#10-代码阅读路线)。`tests/` 和 `tools/` 主要服务开发与发布检查，可稍后阅读。

## 上游项目与致谢

- [WeKWS](https://github.com/wenet-e2e/wekws)：模型及部分训练实现的历史来源。提取时未记录上游 commit/tag；具体继承与修改见[模型来源记录](docs/reference/MODEL_PROVENANCE.md)。
- [Seed-VC](https://github.com/Plachtaa/seed-vc)：历史上用于说话人转换式数据扩充；当前仓库不包含或调用该引擎。
- [PyTorch](https://pytorch.org/) 与 [torchaudio](https://github.com/pytorch/audio)：训练、推理与 fbank 特征实现；其他依赖见 [NOTICES](NOTICES.md)。

公开数据来源、历史工具与当前接口的关系见[来源与许可证](docs/DATA_PROVENANCE.md)。参考项目不等于当前依赖，历史使用也不等于此次重新运行过。

## 当前范围

当前主仓工具链的本地验证覆盖 Windows / Python 3.12 / CPU FP32，torch 与 torchaudio 配对版本为 2.6.0。Linux、Python 3.10/3.11 的 CI 矩阵已配置，远程运行结果待验证。主仓 `easykws-listen` 已做模拟设备测试，真实设备行为尚待验证，设备须直接支持 16 kHz 单声道；上文 ESP32-S3 实机记录属于独立板端工程。

首版不复现历史最终模型的完整训练配方，也不承诺历史现场效果。训练时在线目标音量增强尚未纳入，离线 gain 不能替代它；量化、ONNX 和自动板端导出尚未接入当前工具链。

本项目采用 [Apache-2.0](LICENSE)。录音、权重与生成数据由使用者自行准备；敏感信息的发布前检查依赖私有流程，公开卫生检查不是完整保证。参与开发见[开发说明](docs/DEVELOPMENT.md)。

## English summary

EasyKws helps individuals and small teams build their own wake words: choose a phrase, record it and similar-sounding phrases with a few friends, then combine those recordings with public audio resources for training and validation. The PyTorch toolkit covers data preparation, augmentation, training, streaming calibration, export and inference. No pretrained wake-word model is provided. Start with the [documentation map](docs/README.md); the [technical report](docs/technical_report.md) explains the approach and current limitations.
