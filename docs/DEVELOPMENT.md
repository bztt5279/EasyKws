# 开发与本地发布检查

[文档首页](README.md) · 本文面向维护者。普通使用者从[操作教程](tutorial.md)开始。

当前仓库提供数据准备、扩充产物导入、离线增强、特征预处理、CPU 训练、流式校准、模型导出与部署包加载、流式推理 API 与模拟设备验收的麦克风入口；实机麦克风行为未验证。开发应使用新建的独立虚拟环境，原始归档保持只读，不将私有数据或环境复制到仓库。

```powershell
python -m pip install -e '.[preprocess,dev]'
python -m pytest
```

项目声明 Python >=3.10，目前实测仅覆盖 Windows Python 3.12。Python 3.14 不在当前测试覆盖内；安装成功、声明的最低版本和跨平台运行证据是不同的结论。完整流程及数据契约见 [preprocessing.md](reference/preprocessing.md)。测试使用小型合成输入，不能据此声称模型效果或部署性能。

## 提交前检查

先脱敏、核对来源与许可，再暂存和提交。不要把原始音频、模型权重或压缩交付包加入待公开文件。检查新增文档、配置、日志与测试中的身份信息、用户目录、私钥及凭据；测试也不应包含真实敏感样例。

在仓库根目录执行：

```powershell
python tools/check_hygiene.py --tracked
python tools/check_hygiene.py --files docs/reference/preprocessing.md docs/DEVELOPMENT.md
git diff
git diff --cached
python tools/check_hygiene.py --staged
```

`--tracked` 扫描当前工作树中 Git 已跟踪文件的内容；尚未加入 Git 的文件需使用 `--files` 显式列出，路径相对 `--root`（默认当前目录）。`--staged` 扫描暂存区 blob，覆盖下一次提交的新增或修改内容；它与工作树内容可能不同。三个选择模式必须选一个，不能合用。显式文件模式无需 Git。

检查脚本拒绝常见音频、权重和压缩扩展名、越界路径、符号链接与 submodule，并用规则检测绝对用户目录、私钥和明显凭据。非 UTF-8 内容需要人工审阅。测试中的用户路径例外仅限 `tests/` 下带 `SYNTHETIC_EXAMPLE` 标记且用户名为 `synthetic-user` 或 `example-user` 的单个匹配；它不会豁免同一行的其他用户路径或凭据。

## 分发与 CI 检查

构建分发物并审查其成员与元数据（输出在仓库外）：

```sh
python -m pip install build
python -m build --outdir ../easykws-dist
python -B tools/check_release_artifacts.py ../easykws-dist/*.tar.gz ../easykws-dist/*.whl
```

`tools/check_history.py --root .` 对全部可达提交树运行同一套公开卫生规则；远程 CI（`.github/workflows/ci.yml`）覆盖 Windows/Linux × Python 3.10/3.11/3.12 的安装、全仓测试、合成冒烟与打包审查。workflow 文件存在不等于远程执行已通过。安装包黑盒验收位于 `tests/installation/`，只在 wheel 安装环境中运行（源码树中跳过，不作为安装证据）。

这是启发式检查，不是完整脱敏保证：改后缀、未覆盖的二进制格式、业务身份信息或其他隐私内容仍可能漏检。`.gitignore` 不能替代发布文件审阅，也不能让已经跟踪的文件自动退出提交。不要为了通过规则而改写实际敏感内容的表示方式；应移除该内容并检查其来源及 Git 历史。

## 验收边界

报告验证时说明实际解释器、平台、依赖、输入类型与命令，区分静态审阅、合成输入测试、数值对照、真实录音评估和硬件运行。文档中的命令是复现入口，不表示每次变更已经执行或通过。

当前工作以本地候选仓库和本地验收为范围，没有进行远程发布。添加远程、推送或公开发布应独立核对待发布文件及历史；本地测试通过不能代替该审阅。
