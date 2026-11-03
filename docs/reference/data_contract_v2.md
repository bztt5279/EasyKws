# v2 数据契约

[文档首页](../README.md) · [操作教程](../tutorial.md)

本文描述录音清单与 prepared bundle 的 v2 接口。v2 与 v1 并行；现有 v1 清单、API 和产物继续受支持。未知 schema/version 会明确拒绝，不按旧版猜测或静默降级。完整使用流程见[操作教程](../tutorial.md)；本页只查字段和接口。

## 显式命令与路径

v2 必须通过显式命令选择：`plan-v2` 检查并报告计划，`run-v2` 生成 prepared bundle，`upgrade-v1` 将 v1 输入迁入新目录。迁移不原地修改 v1 清单或产物。命令参数以各命令帮助为准；接口使用 manifest、config、output，以及数据集 ID 到本机目录的 `path-map`。

`path-map` 是用户本机的 `dataset_id`→根目录映射，必须恰好覆盖清单中的全部数据集。音频和资源路径相对各自数据集根目录解析。物理根目录、`path-map` 内容不得写入 manifest、prepared bundle 或后续产物；因此移动数据时，用户可更新本机映射而保持清单中的相对路径不变。

`upgrade-v1` 使用显式音频根目录读取旧输入，在新输出目录写入 v2 manifest 和展开配置；保守地把旧 synthetic 标记映射为 `test-fixture`，并固定旧划分。升级不会改写旧目录。需要核对迁移后的来源、用途和划分，再继续使用新 bundle。

若要用合成音调试用升级接口，先在仓库根目录生成示例，再执行：

```sh
python examples/make_synthetic_recordings.py --output ../easykws-demo-recordings
easykws-prep upgrade-v1 --manifest ../easykws-demo-recordings/recordings.jsonl --audio-root ../easykws-demo-recordings/audio --output ../easykws-demo-v2
```

在仓库之外创建 `../easykws-paths.local.json`，内容为 `{"recordings": "../easykws-demo-recordings/audio"}`。映射中的相对根目录相对于命令的当前工作目录解析；换工作目录时可使用本机绝对根目录，但该映射不得提交到公开仓库。然后运行：

```sh
easykws-prep plan-v2 --manifest ../easykws-demo-v2/manifest.json --config ../easykws-demo-v2/config.json --path-map ../easykws-paths.local.json
easykws-prep run-v2 --manifest ../easykws-demo-v2/manifest.json --config ../easykws-demo-v2/config.json --path-map ../easykws-paths.local.json --output ../easykws-demo-prepared-v2
easykws-preprocess run --prepared ../easykws-demo-prepared-v2 --path-map ../easykws-paths.local.json --output ../easykws-demo-features-v2
```

所有输出目录必须不存在。升级目录中的 `UPGRADE.json` 是转换记录，不是 prepared READY；必须继续运行 `run-v2`。此例仍是测试音调，升级不改变其测试用途。

## Python API

准备阶段提供 `easykws.prep.v2.plan(manifest, path_map, config=None)` 和 `prepare(manifest, path_map, output, config=None)`。`upgrade_v1(manifest, audio_root, output, config=None)` 执行显式迁移。预处理阶段沿用 v1 API 入口读取 prepared bundle；v2 调用为 `plan(prepared, None, config=None, path_map=...)` 与 `run(prepared, None, output, config=None, path_map=...)`。在 CLI 中，`--path-map` 与 `--audio-root` 互斥。

配置沿用 v1 参数，schema 必须为 `easykws-prep-config/v2`；v2 纳入状态只允许 `pass` 和 `manual`。

## 清单结构

v2 清单是严格 JSON 对象，顶层恰含 `schema`、`datasets`、`resources`、`selection`、`rows`；`schema` 必须为 `easykws-recordings/v2`。字段缺失、多余字段、重复或未知版本均报错。

每个 `datasets` 项恰含 `dataset_id`、`source`、`version`、`license`、`split_policy`。`source`、`version` 和 `license` 须为非空字符串；它们只是清单声明，不构成许可认证。`split_policy` 为 `speaker` 或 `official`。

每个 `rows` 项恰含以下字段：

- `sample_id`、`dataset_id`、`speaker_id`、`source_recording_id`、`audio_path`、`audio_sha256`；
- `label`、`group`、`expected_phrase`、`review_status`；
- `origin`、`purpose`、`parent_sample_ids`、`resource_ids`、`requested_split`、`generation`。

ID 在清单中全局唯一；源录音按 `(dataset_id, source_recording_id)` 区分。`audio_path` 是安全的相对 WAV 路径。纳入样本必须提供实际音频文件 SHA256；排除样本缺少文件时哈希可为 null。纳入状态只允许 `pass` 或 `manual`，配置不能放行 `pending` 或 `flagged`。

`origin` 描述来源，只能是 `recorded`、`public`、`vc` 或 `test-fixture`，未列出的来源类型不受支持。`purpose` 独立描述用途，只能是 `training-data` 或 `synthetic-test`。`test-fixture` 必须标为 `synthetic-test`。测试标记沿父样本和资源关系传播，已排除的祖先也参与传播；清单保留原始声明，`effective_purpose` 表示计算后的用途。合法 VC 不会仅因生成方式自动变成测试数据。

说话人已知时，`speaker_id` 表示真实身份；可用别名归并。同一身份不能借别名拆分。只有 `origin=public` 且划分策略为 `official` 时，未知说话人可以为 null；不得伪造独立身份。未知说话人只能按官方划分和来源隔离，统计中单独报告，不能据此声称完成真人隔离。

`parent_sample_ids` 和 `resource_ids` 是唯一 ID 数组，可为空。VC 样本必须且仅有一个父样本，并引用至少一个 `reference-voice` 资源；其 label 为 0、目标词及实际说话人须与父样本一致，派生样本只能进入 train。非 VC 的 `generation` 必须为 null；VC 的 `generation` 含 `engine`、`model`、`config_sha256`、`seed`、`status`，状态为 `success`、`failed` 或 `pending`。后两种状态强制排除，且与人工审核状态分开。

`requested_split` 可为空或为 `train`、`val`、`test`。`official` 数据集的每条记录必须明确指定划分；`speaker` 策略允许锁定原划分。锁定划分、说话人映射和组约束相冲突时必须报错。最终 train/val/test 均须非空并满足类别与组要求；存在固定划分时不得使用 `split_counts`。自动划分只处理未锁定的连通单元；全部记录（包括排除记录）中的共享说话人、源录音和父子关系都参与连通关系，防止通过排除桥接记录拆开数据。资源不合并说话人单元。

## 资源与选择子集

每个 `resources` 项恰含 `resource_id`、`dataset_id`、`audio_path`、`sha256`、`kind`、`purpose`、`split`、`speaker_id`、`source_sample_id`。`kind` 为 `reference-voice`、`noise` 或 `rir`；`purpose` 为 `training-data` 或 `synthetic-test`；`split` 为 `train`、`val`、`test` 或 `external`。`external` 表示声明的外部训练资源，其身份与未知测试说话人是否重合无法证明。

`source_sample_id` 可为空；非空时必须指向存在的样本，并参与用途传播和划分核验。已知 `speaker_id` 也须与样本身份交叉核验。被纳入样本引用的资源须存在且通过 WAV/hash 核验。train 样本不得引用 val/test 资源，也不得引用已知来自 val/test 的样本或说话人资源。资源不代替真实说话人，也不计入说话人数量。全零噪声或 RIR 的效果质量检查属于后续工作。

资源身份与清单中已知 val/test 音频相同时，即使自称 `external` 且没有填写来源样本，也会拒绝用于训练。预处理增强阶段同时利用文件哈希与可核实的 PCM 身份，覆盖只改 WAV 封装的相同内容；可读取且身份通过核验的排除项也参与检查，不可读排除项记录检查边界。已知测试音频不能借资源导入静默丢失测试用途。这不承诺发现重编码、裁剪、变速等所有同源关系。

`selection` 恰含 `kind`、`reason`、`available_count`。`kind` 为 `full`、`fixed-selection` 或 `diagnostic-subset`；`available_count` 不小于清单行数。`full` 时两者必须相等；其他选择必须说明理由。输入集合选择与后续 QC 排除分别计数，不能把 QC 排除说成截断。诊断子集标记会传播；选择集即使含测试夹具也仍带有夹具用途标记。

## 产物与用途边界

v2 prepared bundle 包含原始 v2 `manifest.json`、展开的 `config.json`、`train.jsonl`、`val.jsonl`、`test.jsonl`、`excluded.jsonl` 和最后发布的 `READY.json`。行记录保留声明字段，并记录实际身份、划分、排除原因、`effective_purpose` 与纳入音频元数据。`READY.json` 分开表达用途、`data_flags`（如 `synthetic-test`、`diagnostic-subset`）和选择信息，并记录来源清单、配置、逐文件哈希及统计。

读取 v2 bundle 时，预处理器须重建清单契约和划分，并核对 READY、索引、配置及实际音频；只检查哈希不足以确认逻辑一致性。预处理产物继续携带清单、资源和用途信息，并使用相应 v2 schema。不会自动迁入历史音频或权重。

来源、用途和子集标记用于追溯与阻止工具链静默丢失信息。哈希和 READY 检查不能证明来源真实、许可有效，也不能抵抗有人同步篡改全部记录；这些标记不是防伪或可信来源认证。

## 当前实现边界

VC 产物可通过[扩充导入接口](data_expansion.md)接入；外部 VC 引擎不包含在本仓库。自动测试以小型合成音频检查兼容、升级、用途传播、资源边界、未知说话人和重算哈希后的逻辑矛盾；模拟的 VC 声明不代表执行或验证过 VC 引擎。Windows 命令链已验证显式升级、v2 准备与 clean 预处理，以及搬迁音频根目录后的产物一致性；Linux、真实数据适配及识别效果不在本阶段验证范围内。

预处理可通过显式配置产生[离线增强变体](augmentation.md)。特征索引的 `origin` 和录音哈希描述源波形；新增 `augmentation` 记录实际变体及操作链，不能把变体的源录音哈希当作增强波形哈希。`input_count`、`source_included_count` 计数原输入，`included_count` 与 split 的 samples 计数实际输出特征条目。增强不改写 prepared 清单。
