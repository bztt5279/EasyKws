# EasyKws 文档首页

想做一个属于自己的唤醒词，可以先选好词，找几位朋友录制，再跟着教程训练和验证。第一次接触项目时，不需要逐个翻完这个目录，按下面的顺序阅读即可：

**[录制与标注](recording_guide.md) → [操作教程](tutorial.md) → [技术报告](technical_report.md)**

只想先看看软件是否能运行，跳到[合成演示](tutorial.md#1-先跑一遍合成演示)。演示使用合成音频检查流程，不代表真实语音识别效果。

采集精力放在目标词和容易混淆的对照词上；普通语音、音乐、环境噪声和 RIR 使用合适的公开资源补充。录制指南会说明朋友们怎样分组，以及如何另外准备连续校准和现场验证。

## 主要文档

| 文档 | 读完应该能做什么 |
| --- | --- |
| [录制指南](recording_guide.md) | 组织试录、正负样本、质量审核与独立校准录音 |
| [操作教程](tutorial.md) | 依次执行准备、预处理、训练、校准和导出，检查每一步输出 |
| [技术报告](technical_report.md) | 理解各阶段的作用、方案取舍、已有证据和局限 |
| [ESP32-S3 端侧实践](technical_report.md#81-esp32-s3-端侧部署实践) | 了解已有板端实现、真实采音记录及当前尚未接入的部分 |
| [安装说明](install.md) | 建立环境，了解实际支持范围和常见安装问题 |
| [来源与许可证](DATA_PROVENANCE.md) | 区分上游实现、外部生成工具、数据来源及当前依赖 |

## 需要时再查

`reference/` 保存详细字段与接口约定。它不是另一套教程，不需要按文件名顺序阅读。

- **数据**：[清单与划分](reference/recording_manifest.md)、[多来源 v2 清单](reference/data_contract_v2.md)、[扩充产物导入](reference/data_expansion.md)、[公开负样本](reference/public_negatives.md)。
- **训练**：[特征预处理](reference/preprocessing.md)、[离线增强](reference/augmentation.md)、[采样与恢复](reference/training.md)。
- **运行**：[校准协议](reference/calibration.md)、[导出、流式 API 与麦克风](reference/export.md)。
- **来源细节**：[模型来源记录](reference/MODEL_PROVENANCE.md)、[历史结果边界](reference/HISTORICAL_RESULTS.md)。

所有命令默认从仓库根目录执行，音频、清单和运行输出放在仓库之外。参考文档中的示例路径、哈希占位符和参数均需要按实际数据填写。

## 开发与维护

维护者另读[开发说明](DEVELOPMENT.md)。测试、打包和发布检查属于维护流程；它们通过不能替代真实语音效果评估。文档图中的流程表示已实现的软件衔接，不表示发布了可直接使用的唤醒模型。
