# 模型导出、部署包与流式推理

[文档首页](../README.md) · [操作教程](../tutorial.md)

训练与流式校准完成后，`easykws-train export` 将已验证的校准选择打包为自包含部署包；`easykws.runtime.load_model` 只依赖包目录与已安装软件即可加载推理；`easykws-listen` 提供显式启动的 16 kHz 单声道麦克风入口。导出生成的是用户自己的本地产物：本仓库不提供预训练唤醒模型，也不把任何合成冒烟结果表述为识别效果或现场误唤醒率。

## 导出命令

```sh
easykws-train export --run ../demo-training --features ../demo-features \
  --calibration ../demo-calibration --manifest ../calibration/sessions.json \
  [--path-root ../calibration/audio] --output ../demo-package
```

checkpoint 与阈值完全以校准目录中的选择记录为准，不提供绕过选择的替换参数。`--manifest` 与 `--path-root` 用于重新核对实际校准输入：导出会重新读取会话清单、逐个核验音频哈希，并与绑定记录中的清单/输入/协议一致比对，不一致即失败。输出必须是新目录，不得与训练运行、特征包、校准目录或音频根目录重叠；所有输入保持只读。

导出在写入任何文件之前完成全部校验：训练运行与候选 SHA、特征与 CMVN 身份、校准 READY 与无失败标记、严格 JSON（拒绝重复键、NaN、未知版本）、选择记录与报告语义一致、且该选择确实是既定选择规则的确定性胜者；随后对实际模型执行 `verify_binding`（权重内容摘要、结构、CMVN、完整前端、阈值与完整触发规则、校准协议与输入、执行配置、实现身份）。通过后写入受限 `model.pt` 并重新加载，再以重载模型完整回放同一批已核验校准会话：同 CPU/FP32 实现要求与保存的逐帧分数、事件完全一致。全部成功后最后写入 `READY.json`；任何一步失败只留下 `FAILED.json` 与部分文件，不会被 runtime 当作完整包，输入文件哈希保持不变。

## 部署包内容

| 文件 | 内容 |
| --- | --- |
| `model.pt` | 受限加载的 state_dict 与基础元数据；不含 optimizer、scheduler、RNG |
| `cmvn.json` | 训练特征包的 CMVN 统计与身份，与模型内缓冲区逐值一致 |
| `threshold.json` | 阈值、比较符、完整触发规则、校准来源与经哈希绑定的校准记录引用 |
| `calibration.json` | 完整校准绑定、来源身份与导出回放证据 |
| `manifest.json` | 版本化 schema、结构名与参数配置、实算参数量、各 payload SHA、内容摘要、完整前端、执行配置、用途与数据标记 |
| `READY.json` | 最后写入的完整性清单，覆盖上述所有文件 |

包内不保存音频、特征缓存、逐帧历史或机器绝对路径。`--allow-test-model` 之类的用途升级入口不存在于导出；包的用途由训练与校准来源中更受限者决定，测试（`synthetic-test`）与诊断子集（`diagnostic-subset`）标记持续传播，受限来源不能通过校准或导出洗成普通用途。

## 校准重用限制

阈值只能与其校准时的实际模型内容、CMVN、前端、触发规则、校准协议和执行配置一起使用。更换同结构不同权重的模型、CMVN、前端或执行配置都必须重新校准；重新计算包内文件哈希不能让 A 模型搭配 B 的校准通过。正常重存模型允许容器文件 SHA 变化，推理内容摘要必须相同。

## 独立加载与流式 API

加载不依赖原训练目录、特征包、校准音频或任何绝对路径；可在搬迁包目录、更换工作目录或全新进程中运行：

```python
from easykws.runtime import load_model

model = load_model("../demo-package")            # 默认拒绝测试/诊断用途包
model = load_model("../demo-package", allow_test_model=True)  # 显式豁免用途限制
result = model.push(pcm_block)                   # 1D float32，16 kHz 单声道，[-1,1]
result["frames"]                                 # 本次新增帧 {frame_index, t_audio_sec, score}
result["events"]                                 # 本次新增事件（含 session_id）
model.flush()                                    # 关闭会话；push 需先 reset()
model.reset()                                    # 新会话：新 session_id，时钟与状态清零
```

加载时依次核验包完整性、schema、用途标记，再受限加载模型并实算内容摘要，逐项核对 CMVN、结构、参数量、前端、触发规则、绑定与当前执行配置。包内 `cmvn.json` 的 mean/istd/var/frame_num/fit_split 与校准绑定记录逐字段比对，加载后模型 CMVN 缓冲区再与包内统计逐值对照；仅重算文件哈希的 CMVN 篡改会被拒绝。测试或诊断用途（`synthetic-test`、`diagnostic-subset`）的包默认拒绝；`allow_test_model` 只豁免用途限制，不豁免任何权重、阈值、前端、执行环境或哈希校验，且输出始终显示其测试/诊断属性。

流式语义与校准共享同一实现：25 ms 窗、10 ms 帧移、帧时间为结束时间；数字静音正常推进时钟与状态；空块不推进；flush 不补零；事件由共享 `FreshFrameTrigger` 产生。新会话有新的 `session_id`，不伪装成未中断的连续流。输入校验失败抛出 `ValueError` 且不改变状态；每次 push 前重新检查全部已绑定的可变执行设置（训练模式含全部子模块、autocast、线程数、确定性及 warn-only、MKLDNN、默认 dtype、TF32、matmul 精度），任何偏离立即拒绝且不消费输入，不再称为已批准配置下的推理。

## 执行环境

首版仅支持 CPU/FP32；包内记录校准时实际生效的执行配置（设备、精度、autocast、TF32、线程数、确定性、依赖版本、平台）。加载时逐项比对，任何不一致都会报错并列出差异项，不静默改写。线程数等可复现设置建议在启动 Python 前固定：

```sh
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
```

也可显式调用 `load_model(package, apply_execution_profile=True)` 应用包内批准的 CPU 设置后再核验实际生效值；版本、平台等不可设置项仍必须一致。不支持 CUDA、TF32 实际算子、autocast 或未验证版本的免校准切换，也没有绕过开关。

## 麦克风入口（easykws-listen）

```sh
python -m pip install -e ".[listen]"   # 可选依赖；模型检查不需要它
easykws-listen --list-devices            # 只列出输入设备，不开始录音
easykws-listen --model ../demo-package --check          # 只加载核验，不打开设备
easykws-listen --model ../demo-package --device 0       # 显式选择设备后才开始监听
easykws-listen --model ../demo-package --device "麦克风名" --allow-test-model
```

`--device` 接受设备索引（整数或纯数字字符串，按 PortAudio 索引解析）或精确的完整设备名；名称恰为纯数字的设备无法按名称选择，请使用其索引。无参数显示帮助，不默认打开设备；模式互斥，错误退出码非零。设备须能以 16 kHz 单声道 float32 实际打开；暂不支持原生采样率转换，不偷偷换采样率、平均双声道或启用 AGC/降噪。音频回调只做复制入队（有界队列），模型消费在回调外。队列溢出或设备断流会停止监听、释放设备并以非零状态退出，明确标明连续性已丢失，不补零伪装完整观测。Ctrl+C 正常退出；默认不保存音频，只输出事件与会话状态。缺 sounddevice 时给出安装提示，`--check` 与文件回放不受影响。

麦克风入口的自动验收全部基于模拟设备（成功、无设备、权限失败、格式不支持、回调/消费异常、队列溢出、断流与退出清理）。实机麦克风行为未验证：此入口在本版本明确处于"设备实测待完成"状态，模拟通过不代表硬件兼容性。

## 已知边界

以下范围针对当前主仓 Python 导出与运行入口。既有方案另有独立的 [ESP32-S3 端侧实践](../technical_report.md#81-esp32-s3-端侧部署实践)，尚未迁入本模块，不能由当前 `export` 命令直接生成固件。

- 当前流式推理入口仅 16 kHz 单声道 FP32；实时重采样、量化、ONNX、板端移植均不在范围内。离线预处理可显式重采样，不能据此推断实时设备兼容性。`listen` 可选依赖包含 numpy/scipy/soundfile/sounddevice/torch/torchaudio。Windows 干净环境的 wheel 安装与合成链路已验证，真实麦克风及跨平台安装仍待验证。
- 校准绑定绑定当前实现源码摘要：runtime 相关代码更新后，旧包需在旧代码下使用或重新校准导出。
- 用途标记防止工具链内静默丢失，不是对手工伪造整个包的防伪或可信来源认证。
- 包中阈值来自短时合成或用户自备回放的校准，不代表长期现场误触发率。
