# 本机公开负样本适配器

[文档首页](../README.md) · [操作教程](../tutorial.md) · [扩充导入](data_expansion.md)

本页只说明 Mobvoi 格式的本机元数据转换。适配器不下载数据、不替用户判断关键词含义，也不认证数据许可。来源、版本、许可说明和类别映射都必须显式提供。

官方数据入口：[MobvoiHotwords / OpenSLR 87（小问唤醒词数据集）](https://openslr.org/87/)。音频与标注元数据分别下载，不能只有音频文件而缺少下面所需的元数据。这里的「关键词/非关键词」是原始任务标注，接入其他自定义词时仍需重新确认类别映射。

## 输入目录与类别映射

元数据目录必须包含六个 JSON 数组：`n_train.json`、`n_dev.json`、`n_test.json`、`p_train.json`、`p_dev.json`、`p_test.json`。每项需要全局唯一的 `utt_id` 和整数或字符串 `keyword_id`，可有 `speaker_id`；缺少说话人时保留未知。

音频只按 `audio_root/<utt_id>.wav` 读取，不递归搜索同名文件，也不重采样。UID 必须是安全的 ASCII 文件名主体，不带路径或 WAV 后缀，最长 128 字符。纳入文件必须是可读取、非空、未压缩整数 PCM WAV；适配器记录音频信息、文件及 PCM 哈希。

类别映射只接受整数 `-1` 或字符串 `"exclude"`，必须覆盖全部实际出现的类别。没有默认「全部负类」映射，也不提供正类转换。应根据官方类别含义和自己的目标词判断，不能从原始文件的 p/n 前缀推断标签。

例如下面的数值只是虚构示例，不是官方类别推荐：

```json
{"7": -1, "8": "exclude"}
```

## 命令与 Python 接口

在仓库根目录执行，目录及声明替换为实际内容：

```sh
easykws-prep import-mobvoi --metadata ../public-metadata --audio-root ../public-audio --keyword-labels ../keyword-labels.json --dataset-id public-source --source user-supplied-source --version user-checked-release --license user-checked-license --output ../public-candidates
```

输出 `manifest.json` 可以交给[扩充入口](data_expansion.md)，本机路径映射应包含新增 dataset。Python 接口如下：

```python
from easykws.prep.mobvoi import build

document, report = build(
    metadata_dir, audio_root,
    dataset_id="my-public-source", source="user-supplied source description",
    version="user-supplied release", license="user-checked license declaration",
    keyword_labels={"7": -1, "8": "exclude"},
)
```

上面的 `metadata_dir`、`audio_root` 由调用者指定，返回清单和报告都应保存。

## 官方划分、排除与未知说话人

全部本机元数据行都保留，包括明确排除的类别。排除项标为 `flagged`，音频哈希可为空，不打开其音频。每条报告保留原 UID、keyword ID、官方集合、元数据文件及其 SHA256。官方 dev 映射为 val，数据集声明 `split_policy="official"`。

已知安全字符串说话人 ID 原样保留，其他已知标量转为带类型的稳定哈希，原值留在报告中。只有未知身份使用 null；null 不表示这些录音来自同一个人，也不证明不存在同一人跨集合。

样本来源为 `public`，用途默认 `training-data`，没有生成器、父样本或资源关系。后续仍需通过 v2 的身份冲突和 PCM 去重检查。单独准备仅含负类的清单时，需要显式设置 `require_both_labels=false`，但每个划分仍必须有纳入样本；全排除输入不能生成训练集。

`selection.kind="full"` 只说明覆盖这六份本机元数据，不能证明已经获取完整官方版本、达到官方条数、解释正确或声学质量合格。自动测试用的是虚构元数据和生成音调，不代表真实公开语料验收。
