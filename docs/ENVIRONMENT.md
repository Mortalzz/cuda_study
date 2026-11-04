# 环境与验证记录

日期：2026-11-01。项目使用 CMake + NMake，无完整 Visual Studio IDE，无项目运行脚本。

| 组件 | 版本 | 实际位置 |
|---|---|---|
| Build Tools | 2026 / 18.11.3 | `D:\Software\BuildTools` |
| MSVC / NMake | 工具集 14.51.36231 | `D:\Software\BuildTools\VC\Tools\MSVC\14.51.36231` |
| Windows SDK | 10.0.26100.9457，目录版本 10.0.26100.0 | `D:\Software\WindowsSDK` |
| CUDA Toolkit | 13.4.92 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4` |
| CMake | 4.4.4 | `C:\Program Files\CMake` |
| GPU / 驱动 | RTX 5070 Ti / 616.64 | 架构 `sm_120` |

Build Tools 约 1.73 GiB，Windows SDK 约 1.33 GiB；原完整 VS 目录约 6.64 GiB。此为目录文件大小，不包含系统缓存。新增组件主体在 D 盘，沿用已有 CUDA、CMake 和 Nsight 安装。

显式安装的 Build Tools 组件为 `VC.CoreBuildTools` 和 `VC.Tools.x86.x64`；SDK 选择 `OptionId.DesktopCPPx64`，包含安装器要求的公共依赖。没有为 NMake 切换安装新组件。

Visual Studio Installer 仍在 `C:\Program Files (x86)\Microsoft Visual Studio\Installer`。Windows Installer 缓存、系统运行库与临时文件由安装器维护在系统盘。

安装记录：`D:\Software\BuildToolsInstaller`、`D:\Software\WindowsSDK\installer`。安装返回码均为 0，无需重启。

## 构建结构

- 根 `CMakeLists.txt` 集中注册 `reduction` 和 `transpose`。
- `add_cuda_operator()` 统一公共 include、C++17、CUDA 架构、行号和 UTF-8 编译选项。
- CMake 生成 NMake Makefile，Release/Debug 使用独立目录。
- VS Code 直接调用 CMake Tools，运行路径取自当前选中目标。
- 已移除 `scripts/build.ps1`、`scripts/profile.ps1`；性能分析直接调用 ncu。
- 算法源文件未改动，备用实现和历史报告仍保留。

## 备份

整理前完整备份：`D:\Software\researh\Project-before-cleanup-20261101.zip`。

切换 NMake 前的备份：`D:\Software\researh\Project-before-nmake-20261101.zip`。

旧生成产物分别暂存于项目外的 `Project-generated-before-cleanup-20261101`、`Project-generated-before-nmake-20261101`，均不参与当前构建。

## 验证边界

构建与内存检查通过不等于 softmax 数值正确。原 shared kernel 仍存在指数溢出、NaN 漏检和尾部行保护缺失；本次只调整构建和运行方式。

## 本次实际验证

在 x64 Build Tools 开发者环境中执行，以下检查退出码均为 0：

| 检查 | 结果 |
|---|---|
| `cmake --preset release` | 生成 `build/release/Makefile`，生成器为 `NMake Makefiles` |
| Release 目录执行 `nmake reduction transpose` | 两个目标均编译、链接成功 |
| 直接运行 Release 的两个 exe | reduction、transpose 均打印原程序的 PASS |
| `cmake --preset debug` 后在 Debug 目录执行 `nmake` | 两个目标均编译、链接成功；reduction 运行成功 |
| Release 目录执行 `nmake help` | 列出 reduction、transpose 和清理等目标 |
| Release 目录再次执行 `nmake` | 直接复用已完成目标，无重复编译 |
| 根目录执行 `cmake --build --preset release --target transpose` | 通过 CMake 调用同一 NMake，成功 |
| Compute Sanitizer memcheck 检查 Release reduction | 固定内置输入下 `ERROR SUMMARY: 0 errors` |
| 直接调用 ncu 采集 Release reduction | 成功生成 `reports/nmake-verification.ncu-repz` |

Nsight 验证只在当前进程中将 TEMP/TMP 指向项目 `build` 目录，没有修改系统环境变量。VS Code 配置已检查 JSON 和所用 CMake Tools 命令；未进行编辑器界面内的交互调试验证。

## VS Code 终端环境修正

普通 PowerShell 不会自动获得 CMake Tools 内部的开发环境。工作区新增默认 **CUDA Developer PowerShell (x64)** 终端配置，通过 Windows PowerShell 导入 Build Tools 官方开发模块，再调用 `Enter-VsDevShell` 加载 x64 环境；未更改系统 PATH 或用户 PowerShell profile。该配置只影响新终端。

用户自行创建的 `run.sh` 使用 CMD 风格命令，在 Bash 下存在路径转义和程序查找问题；文件保留，不作为构建入口。直接在开发终端使用 CMake/NMake 和 PowerShell 的 `.\程序.exe` 形式即可。

验证时读取工作区终端配置，使用相同可执行文件和初始化命令启动 Windows PowerShell（去掉自动验证不需要的 `-NoExit`），成功执行 `cmake --preset release`、Release 目录的 `nmake` 及两个 exe，退出码均为 0。未在 VS Code 界面内操作；已有终端仍需用户关闭后重建。
