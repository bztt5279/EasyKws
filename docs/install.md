# 安装、支持范围与复现

[文档首页](README.md) · 安装后继续读[操作教程](tutorial.md)。本页只负责环境与依赖，方法解释见[技术报告](technical_report.md)。

本页说明从源码分发或已构建 wheel 安装 EasyKws、已核实的依赖组合、以及端到端复现入口。安装包与 Python 导入名均为 `easykws`，命令使用 `easykws-*`。所有示例输出都指向仓库外的新目录。

## 支持范围（首版边界）

| 维度 | 已实测 | 计划矩阵内待验证 |
| --- | --- | --- |
| 操作系统 | Windows（本地开发环境） | Linux（CI 配置已提供，远程执行待授权） |
| Python | 3.12 | 3.10、3.11 |
| 执行 | CPU / FP32，torch 2.6.0 + torchaudio 2.6.0 配对 | 其他版本不承诺 |

`requires-python >=3.10` 是安装下限声明；在支持矩阵的每个目标组合实际通过前，本页只把上表"已实测"范围称为已验证。CI（`.github/workflows/ci.yml`）配置了 Windows/Linux × Python 3.10/3.11/3.12 的 CPU 测试与打包 job；远程 workflow 结果在获得推送授权前不存在，不能当作已通过。

CPU 推理不等于使用了 CPU-only 发行包：请用 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 记录实际 torch 构建与设备。本项目不使用 CUDA/TF32/autocast，但旧的 `+cu126` wheel 也能在 CPU 上运行。

## 从源码分发安装

Windows（PowerShell）：

```powershell
python -m venv ../easykws-env
../easykws-env/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchaudio==2.6.0
python -m pip install -c configs/cpu-constraints.txt ".[preprocess,listen,dev]"
```

POSIX（bash）：

```sh
python3 -m venv ../easykws-env
. ../easykws-env/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchaudio==2.6.0
python -m pip install -c configs/cpu-constraints.txt ".[preprocess,listen,dev]"
```

`configs/cpu-constraints.txt` 只固定 `torch==2.6.0`/`torchaudio==2.6.0` 配对；其余依赖按 pyproject 声明范围从 PyPI 解析。extras 含义：`preprocess`/`training` 为数据与训练必需；`listen` 为麦克风入口（含 sounddevice）；`dev` 为 pytest。只要 `python -m pip check` 通过、且 `python -c "import torch, numpy, scipy, soundfile"` 无错误，即满足完整功能的最小依赖闭包。

只用数据准备（prep）时，无需安装任何 extras：`pip install .` 即可，prep 没有第三方运行依赖。

## wheel 安装与一键冒烟

安装已构建的 wheel（不是 editable）后，冒烟入口随包分发：

先按上文安装 CPU torch/torchaudio 配对，再安装本机 wheel 并显式选择需要的 extra。下面的文件名按实际版本调整，不能只装没有 extras 的基础包就直接执行训练：

```sh
python -m pip install -c configs/cpu-constraints.txt "../easykws-dist/easykws-0.1.0-py3-none-any.whl[preprocess]"
python -m pip check
```

约束文件从源码分发获取；只持有 wheel 时，也可在先安装 CPU 配对后安装带 `[preprocess]` 的 wheel。`listen` 用于麦克风，`dev` 用于运行测试。

```sh
easykws-smoke --output ../easykws-smoke-run
```

该命令在指定新目录内生成虚构音调与元数据，依次执行 v2 准备、VC/公开负样本接口夹具导入与 QC、五类离线增强、fbank 与 train-only CMVN、两 epoch 小模型训练、三会话流式校准、导出部署包，并在全新 Python 进程中加载、push/reset/flush 回放，与校准逐帧分数核对。`summary.json` 记录各阶段计数与验证结论。所有产物都是 `synthetic-test` 测试用途；冒烟通过不代表识别效果。完整链路细节与预算见 [导出与部署说明](reference/export.md) 和仓库内脚本 `src/easykws/smoke.py`。

示例配置（`configs/*.json`）与手动示例生成器（`examples/make_synthetic_recordings.py`）随源码分发提供；wheel 不包含它们，请从源码分发获取。

## 端到端复现（换成自己的数据）

完整手动路径：[录制与标注](recording_guide.md) → [操作教程](tutorial.md)。详细参数按需查阅 [reference](README.md#需要时再查)。每一步都要求新的输出目录，输入只读，产物留在仓库外。

常见问题：

- `torch` 下载了 CUDA 版：安装命令未先指定 CPU 索引；按上文顺序先装配对版本。
- 加载部署包报执行环境不匹配：先设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1` 再启动 Python，或按 [导出说明](reference/export.md#执行环境) 显式应用包内批准配置。
- `--device 0` 找不到设备：数字按 PortAudio 索引解析，先 `easykws-listen --list-devices` 确认索引与名称。
- 麦克风功能导入失败：extras 用的是 `listen`；`--check` 与文件回放不需要它。
- Windows 长路径或权限错误：把输出目录放在用户目录下的短路径中，不要放在仓库内部。

## 发布前检查（维护者）

```sh
python -B tools/check_hygiene.py --tracked
python -B tools/check_history.py --root .
python -m build --outdir ../easykws-dist
python -B tools/check_release_artifacts.py ../easykws-dist/*.tar.gz ../easykws-dist/*.whl
```

这些是启发式公开检查，不是完整脱敏保证；更深入的私有扫描与发布内容审查流程不属于本仓库。远程 CI 未运行时，上述本地结果不能替代双平台结论。
