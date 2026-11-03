# 第三方依赖与来源说明

本仓库目前分发自身的 Python 源码、配置、文档和合成样例生成器；依赖通过包管理器另行安装。下表列直接运行依赖及关键音频底层库的项目主许可证，不是完整的二进制依赖清单。依赖包可能还附带其他组件，应以所安装分发包的许可证文件为准。

| 组件 | 项目主许可证 | 本项目用途与依据 |
| --- | --- | --- |
| NumPy | BSD-3-Clause | 数组及 `.npy` 存储；[v2.2.6 许可证](https://github.com/numpy/numpy/blob/v2.2.6/LICENSE.txt) |
| SciPy | BSD-3-Clause | 显式启用时的多相重采样；[v1.15.3 许可证](https://github.com/scipy/scipy/blob/v1.15.3/LICENSE.txt) |
| SoundFile | BSD-3-Clause | PCM WAV 解码；[官方说明](https://python-soundfile.readthedocs.io/en/latest/) |
| libsndfile | LGPL-2.1-or-later | SoundFile 使用的底层库，某些 SoundFile wheel 随包提供；[1.2.2 源码许可声明](https://github.com/libsndfile/libsndfile/blob/1.2.2/src/sndfile.c)、[许可正文](https://github.com/libsndfile/libsndfile/blob/1.2.2/COPYING) |
| PyTorch | BSD-3-Clause | 特征统计、模型训练与受限 checkpoint 保存/加载；[v2.6.0 许可证](https://github.com/pytorch/pytorch/blob/v2.6.0/LICENSE) |
| torchaudio | BSD-2-Clause | 调用 `compliance.kaldi.fbank`；[v2.6.0 许可证](https://github.com/pytorch/audio/blob/v2.6.0/LICENSE) |
| SoundDevice (python-sounddevice) | MIT | `easykws-listen` 麦克风入口的可选依赖；[v0.5.2 许可证](https://github.com/spatialaudio/python-sounddevice/blob/v0.5.2/LICENSE)；已核对本机安装的 0.5.2 dist-info 元数据与 LICENSE 文件 |

python-sounddevice 官方 wheel 可能随包附带 PortAudio 编译库；PortAudio 采用其上游的 MIT 风格许可证（以实际安装分发包内的许可文件为准），本项目源码不包含该二进制。当前模拟设备验收在已安装 sounddevice 的环境中完成，未实机验证麦克风。

SoundFile 官方说明分别列出 Python 包的 BSD 许可证与 libsndfile 的 LGPL 许可证，两者不能合并写成 BSD。未来若制作包含这些库的安装包、容器或可执行文件，需按实际打包内容重新整理许可证、通知及适用的分发材料；当前源码候选不包含这些库的二进制。

当前预处理的解码、fbank 参数及 CMVN 计算沿用项目留存实现的数值约定，适配代码围绕通用清单和完整性校验重新编写。增强和训练保留已记录的算法语义，旧工程特定输入布局、预设样本名单与硬编码来源分支不迁入。模型来源及无法确认的历史上游版本单独记在 [MODEL_PROVENANCE.md](docs/reference/MODEL_PROVENANCE.md)，不以新建仓库历史替代历史来源证据。
