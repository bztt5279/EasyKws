# 导入扩充产物

[文档首页](../README.md) · [操作教程](../tutorial.md)

扩充入口把已生成、已标注的数据加入现有 v2 prepared bundle。它固定原数据的划分，复用 v2 校验，再输出能直接交给预处理的 `prepared/`。不调用 VC 引擎、不下载模型或数据、不执行离线增强。

## VC 与用户新增录音

准备一个只包含新增候选的 [v2 清单](data_contract_v2.md)。候选的 datasets/resources 可以复用原 ID，但同 ID 的声明必须完全相同。新 sample_id 不能与原集合重合；父样本必须来自原 prepared。VC 的父样本必须是原集合中已纳入的 train 正样本。其输出行保留实际源说话人和目标词，生成器、模型、配置哈希及 seed 按实际填写。

所有新增行应明确 `requested_split`；VC 可填 null，导入器固定为 train。新增难负样本用自录数据行、明确的负标签及组名描述，不能靠文件名猜标签。

`--qc` 接收按候选 sample_id 索引的 JSON 对象。例如下列哈希占位符必须替换成实际值：

```json
{
  "converted_001": {
    "source_sha256": "<原 train 源 WAV 文件 SHA256>",
    "output_sha256": "<本候选输出 WAV 文件 SHA256>",
    "resource_sha256": {"reference_001": "<参考资源文件 SHA256>"},
    "status": "ok",
    "reasons": []
  }
}
```

`resource_sha256` 必须恰好对应候选引用的全部资源。QC 只适用于 VC 候选，状态为 `ok/flagged/failed/pending`；原始 QC 内容和摘要保存在导入报告。QC 的源/输出/参考哈希不匹配会拒绝本次导入，不能把另一条音频的审核结果套过来。

VC 同时满足 generation=success、清单审核通过和 QC=ok 才进入候选通过集合。缺 QC 按 pending 记录，flagged 不自动放行。通过者还须实际通过 WAV/hash 检查及现有预处理解码入口的 16 kHz 单声道、非空、非全零、非满幅检查。因此 VC 导入需安装 `easykws[preprocess]`。

这些检查**不自动判断发音正确或合成语音是否可用**。时长异常、疑似截断、长静音、尾能量变化及听审结果应由外部 QC/人工审核给出，再填写状态和理由；本入口没有重新实现历史全部声学 QC 启发式。不要根据文件存在自动生成 `ok`。源和参考数据的取得、授权及实际生成版本仍需自行记录。

```sh
easykws-prep expand-plan --prepared ../base-prepared --candidates ../candidates.json --path-map ../paths.local.json --qc ../qc.json --expected-ids ../planned-ids.json
easykws-prep expand-run --prepared ../base-prepared --candidates ../candidates.json --path-map ../paths.local.json --qc ../qc.json --expected-ids ../planned-ids.json --output ../expanded
easykws-preprocess run --prepared ../expanded/prepared --path-map ../paths.local.json --output ../expanded-features
```

本机 path-map 须覆盖合并后全部数据集，留在仓库外。`planned-ids.json` 是独立计划中的 sample_id 数组；缺少或多出 ID 均报错。不提供时报告明确标为 `declared_manifest_only`，只核验候选清单内部，不声称完成整个生成计划。

## 去重与数量限制

导入先审核、核验文件，再按 PCM 内容去重，最后限量。报告逐条列出 accepted、flagged、failed、pending、duplicate、limited 及原因。生成缺失、审核排除与数量限额分别计数。已有原始数据不因新增候选去重而删除。

同侧重复内容保留确定的一个候选；跨划分的重复 PCM 拒绝导入。参考资源也与已知评估音频的 PCM 比较，不能只换 WAV 头就当成新资源。该检查不等于检测所有重采样、变速或其他同源关系。

`--max-total` 和 `--max-per-source` 默认不限，只影响新增 train 数据，保留全部新增 val/test。`--seed` 固定选择顺序；每源限制对 VC 按父样本计，对普通数据按数据集与原录音 ID 计。真正截断时传播 `diagnostic-subset`，不把 QC 排除当作截断。即使限制值未触发，来源中已有的测试/诊断标记也不能消失。

输出目录必须全新且在输入目录外。保存候选合并清单、展开配置、`report.json` 和 `prepared/`；失败时写 FAILED，失败目录不作为成功结果交接。实际纳入数据以 `prepared/READY.json` 及报告 ID 清单为准。保留原 QC 报告和外部生成记录，文件哈希只是内容绑定，不是来源认证。

## 公开负样本

提供本机 Mobvoi 六 JSON 的适配入口；见 [公开负样本说明](public_negatives.md)。先明确每个原 keyword_id 相对你的目标词是负类还是应排除，再生成 v2 候选：

```sh
easykws-prep import-mobvoi --metadata ../public-metadata --audio-root ../public-audio --keyword-labels ../keyword-labels.json --dataset-id public-source --source user-supplied-source --version user-checked-release --license user-checked-license --output ../public-candidates
easykws-prep expand-run --prepared ../base-prepared --candidates ../public-candidates/manifest.json --path-map ../paths.local.json --output ../with-public-data
```

类别映射值仅支持整数 -1 或字符串 `exclude`，没有默认“全部为负类”的映射。保留原 UID、keyword_id、官方集合和元数据哈希的报告与清单一起保存；dev 映射为 val，未知说话人保持未知。`full` 只表示覆盖输入的六份本机元数据，不证明官方完整版本已经下载或核验。

本阶段验证使用少量合成 WAV、模拟 VC/QC 与虚构公开元数据；不代表运行过外部生成器、恢复过真实数据集或验证过识别效果。
