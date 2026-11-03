# 录音清单与划分

[文档首页](../README.md) · [操作教程](../tutorial.md)

当前接口只负责校验 PCM WAV、应用人工审核规则、建立隔离划分和输出索引。它不做重采样、混音、裁剪、响度处理或特征提取，也不自动判断录音质量。

## 输入清单

UTF-8 JSONL，每行一个对象，可有空行。字段名重复、未知字段或缺少必填字段均报错。标识符由 ASCII 字母、数字、下划线、连字符或点组成，首字符必须是字母或数字，总长不超过 128 字符。

| 字段 | 要求与语义 |
| --- | --- |
| `sample_id` | 必填，稳定且全清单唯一的样本标识符。重新排列清单不能改变它 |
| `speaker_id` | 必填，说话人标识符。同一个人的不同别名必须通过配置归并 |
| `source_recording_id` | 必填，原始录音标识符。从同一原始录音裁剪或处理而来的样本共享此值；不同录音不能误用同一个值 |
| `audio_path` | 必填，相对 `--audio-root` 的 WAV 路径，使用 `/`。不允许绝对路径、盘符、反斜杠、`..` 或解析后逃出根目录的链接 |
| `label` | 必填，整数 `0` 表示目标词，整数 `-1` 表示非目标。不会根据文件名、文本或 group 猜标签 |
| `group` | 必填，非空字符串，由用户定义语义，例如 `near_field`。可配置每个划分必须包含的组 |
| `expected_phrase` | 必填，字符串。正样本必须非空；负样本可为空，也可记录实际说话文本。该字段不会覆盖 label |
| `review_status` | 必填，`pending`、`pass`、`manual` 或 `flagged`。默认纳入 `pass`、`manual`；`manual` 表示人工审核通过 |
| `origin` | 必填，`recorded` 或 `synthetic`。当前合成示例必须填 synthetic；不能用录音标记伪装流程测试数据 |
| `parent_sample_id` | 可选，派生样本的直接父样本 ID，须在同一清单中；无父样本可省略或填 null，不允许循环关系 |

格式示例（单行，不是一份足以划分的数据集）：

```json
{"sample_id":"clip_001","speaker_id":"person_001","source_recording_id":"session_001","audio_path":"person_001/clip_001.wav","label":0,"group":"near_field","expected_phrase":"你好小A","review_status":"pass","origin":"recorded"}
```

同源派生关系必须保留；父样本可以因为质量原因被排除，但仍应留在清单中用于追溯。由多个输入混合产生的样本目前没有多父节点接口，请留到后续增强阶段处理。

所有纳入样本须为 Python `wave` 可读取的非空、未压缩整数 PCM WAV（8/16/24/32 位）；当前步骤读取原采样率和通道数，不转换为 16 kHz 单声道。程序检查音频负载长度，并同时记录整个文件与解码出的 PCM 字节哈希。内容完全相同的纳入音频会报错，要求先显式去重；不会依赖不同文件名掩盖重复。

被排除样本仍校验字段、路径和关系，但不要求音频存在，不读取其音频内容。使用错误的 ID 或别名可能造成漏隔离；哈希不能发现不同编码、加噪或变速后的所有近似重复。

## 配置

参考 [prepare.example.json](../../configs/prepare.example.json)。配置为 JSON 对象；缺省项使用默认值，未知项报错。嵌套对象整体替换，必须提供其全部必要字段。

| 配置 | 默认与作用 |
| --- | --- |
| `schema` | `easykws-prep-config/v1` |
| `seed` | 非负整数，默认 0；影响独立单元的确定性排序 |
| `speaker_aliases` | 默认 `{}`，别名到真实说话人 ID 的映射，支持链式归并，禁止循环 |
| `exclude_sample_ids` | 默认 `[]`，额外显式排除的 ID；引用不存在的 ID 报错 |
| `include_review_status` | 默认 `["pass", "manual"]`；其他状态记入排除索引 |
| `split_ratios` | 默认 train/val/test 为 0.6/0.2/0.2；各项须大于零、合计为 1 |
| `split_counts` | 默认 null；可指定 train/val/test 的独立单元数量，覆盖比例计算。各项正整数，合计须等于单元总数 |
| `speaker_splits` | 默认 `{}`；按归并后的说话人显式分配，须恰好覆盖所有纳入说话人，不能与 split_counts 同时使用 |
| `require_both_labels` | 默认 true；每个划分必须包含正负两类，可显式关闭以准备单类诊断集 |
| `required_groups` | 默认 train/val/test 各为 `[]`；要求相应划分覆盖指定 group |

任何共享实际说话人、原始录音或父子关系的样本都会合并为同一个**独立单元**。排除样本也参与关系连接，避免通过排除中间样本把同源数据分到两侧。一个单元可能包含多人，因此 split_counts 按单元计数，并非始终等于人数。

至少需要 3 个独立单元。自动模式按 seed 与单元内说话人 ID 计算稳定排序，按比例的最大余数法分配数量；同样内容的行顺序改变不会改变划分。增加或删除说话人可能改变划分，不保证增量稳定。

自动模式不会搜索满足类别和组覆盖的所有组合。如果选定划分缺少所需类别或组，会直接失败；应补录数据、调整 seed，或提供明确的 speaker_splits。显式配置同样不得打破同源隔离约束。测试集应在实验前固定，不应根据模型表现反复调整。

## 输出与完整性

`plan` 仅校验并输出统计。`run` 要求输出目录不存在且位于音频根目录之外，生成：

- `train.jsonl`、`val.jsonl`、`test.jsonl`：纳入样本、原相对路径、归并后的 `actual_speaker_id`、split、音频信息与哈希。
- `excluded.jsonl`：排除样本、原因、`included=false`、`split=null`，无音频信息。
- `config.json`：展开默认值后的实际配置。
- `READY.json`：schema `easykws-prepared/v1`、数量统计、输入清单和配置哈希、各产物文件哈希、purpose。

产物写入后逐文件回读核验，最后以重命名方式发布 READY.json。写入失败会尽力留下 FAILED.json；缺少 READY.json 的目录不得作为完成产物消费。后续读取方还必须复核其中的文件哈希；这个标记不防止写入后的人为篡改，也不替代真实性判断。

只要纳入样本中含 synthetic，purpose 就是 `synthetic-test`；否则为 `recorded-data`。后者只说明清单声明的数据类型，**不代表模型已可发布或质量达标**。这些标记和来源关系需要在后续预处理、训练、导出中继续传递。

不将机器绝对音频路径写入索引。消费索引时仍须显式提供 audio_root，并核验当前音频哈希。目录移动可以改变根路径，不能改变相对路径和音频内容。

## v2 契约

v1 清单继续受支持。v2 字段、显式迁移和兼容规则见 [v2 数据契约](data_contract_v2.md)；`path-map` 仅用于本机解析，不进入产物。
