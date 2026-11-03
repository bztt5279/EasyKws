# 验证评分与流式校准

[文档首页](../README.md) · [操作教程](../tutorial.md)

训练完成后，使用独立验证录音选择候选模型和部署阈值。当前支持 CPU/FP32 的文件回放；导出与部署加载见[导出与部署说明](export.md)。测试集不能参与模型或阈值选择，合成音调只验证接口。

## 两种评估口径

`evaluate` 对特征目录中的 val 样本计算整句最大分，报告 AUC、样本召回和分段负样本误判比例。它不计算连续音频 FAH，也不直接确定部署阈值。

```sh
easykws-train evaluate --run ../demo-training --features ../demo-features --output ../demo-evaluation --thresholds 0.1 0.5 0.9
```

`calibrate` 将原始会话音频逐帧送入同一个 `FreshFrameTrigger`，在显式正例窗口和负音频区间上统计事件。FAH 的分母是清单中负区间的实际时长，包括数字静音；未标注区间单独记录，不擅自当成负例。每个正例窗口的首次事件算命中，其余算重复事件。区间采用左闭右开规则。

```sh
easykws-train calibrate --run ../demo-training --features ../demo-features --manifest ../calibration/sessions.json --config configs/calibration.example.json --output ../demo-calibration
```

音频路径默认相对于清单目录，也可用 `--path-root` 指定根目录。输入只读，输出放在新的独立目录，不覆盖已有产物。

## 会话清单

清单格式如下。SHA256 和样本数须填入实际值；示例不是可直接执行的数据文件。

```json
{
  "schema": "easykws-calibration-sessions/v1",
  "purpose": "training-data",
  "split": "calibration",
  "sample_rate": 16000,
  "channels": 1,
  "clock": "samples",
  "boundary_policy": "reset-each-session",
  "gap_policy": "split-session",
  "provenance": {"source_id": "user-validation", "description": "Independent labeled recording"},
  "sessions": [{
    "id": "session-01",
    "path": "session-01.wav",
    "sha256": "REPLACE_WITH_FILE_SHA256",
    "duration_samples": 160000,
    "recording_kind": "continuous-recording",
    "speaker_ids": ["validation-speaker"],
    "positive_windows": [{"id": "target-01", "start_sec": 3.0, "end_sec": 5.0}],
    "negative_intervals": [{"start_sec": 0.0, "end_sec": 3.0}, {"start_sec": 5.0, "end_sec": 10.0}]
  }]
}
```

音频必须是 16 kHz、单声道 PCM WAV，支持 8/16/24/32 bit。标注区间不能重叠或超出实际时长。`split` 仅接受 `val` 或 `calibration`。`recording_kind` 可为 `continuous-recording`、`offline-replay` 或 `synthetic-concatenation`；人工合成拼接必须标为 `synthetic-test`，不能包装成真实连续录音。测试用途训练模型也不能通过校准降级为普通模型。

说话人与文件内容检查只能识别已有记录中的冲突，不能验证新录音的身份、标注真伪或发现所有近重复。请提供真实的说话人及来源信息，并独立划分校准集。

## 触发与计时

默认规则为连续 3 帧确认、50 帧热身、低分持续 0.3 秒重新武装、2 秒冷却，同一激活段只报告一次。比较符 `ge` 表示大于等于阈值，`gt` 表示严格大于。完整规则写入校准绑定记录，不沿用历史模型的固定阈值。

前端为 40 维 fbank、25 ms 窗、10 ms 帧移。事件时间取对应完整帧的结束时间；尾部不足一帧不补零，实际输入时长仍按样本数计。零值块正常推进缓存、热身、冷却和音频时钟；空块不推进时间。录音断流须拆成不同 session，各 session 重置状态，不能把未观测间隔补成静音。

## 候选选择与绑定

配置显式给出候选 checkpoint、阈值网格及至少一个约束：`min_recall` 或 `max_fah`。示例配置只演示格式，参数不是效果建议。缺少正例或负区间时，相应比率为 null；不能满足约束的候选不会被自动兜底选中。可行候选依次按召回降序、FAH 升序、阈值降序、候选名升序选择。

校准输出与训练目录分离。训练 `status.json` 中的 `selected_checkpoint` 不被回写；校准目录的 `selected_checkpoint.json` 才记录本次选择。无可行候选时保留报告，不生成选择记录。成功产物通过 `READY.json` 清单核验；执行失败保留失败标记。

绑定包含实际推理权重和持久缓冲区的内容摘要、结构、CMVN、前端、阈值与触发规则、实现摘要、校准输入和选择协议，以及执行精度与环境。原 checkpoint 文件哈希是来源记录；推理内容摘要用于判断重打包是否保持内容一致。同结构不同权重、不同 CMVN、规则或评分环境不能直接复用阈值。其他平台或精度需要重新校准或另行提供兼容性证据。

用途标记防止工具链内静默丢失测试属性，不是签名或可信来源认证。短时回放的协议 FAH 不代表长期现场误触发率。
