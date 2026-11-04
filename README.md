# CUDA Lab

CUDA C++17 算子练习。根目录 `CMakeLists.txt` 管理全部目标：CMake 生成 Makefile，Windows 使用 Build Tools 自带的 NMake 构建，直接运行生成的 exe。项目不需要自定义脚本或完整 Visual Studio。

## 编译与运行

从开始菜单打开 **x64 Native Tools Command Prompt for VS**（Build Tools 安装的开发者命令提示符），它负责加载 `cl`、`nmake` 和 Windows SDK 环境。以下命令使用 CMD 语法：

```bat
cd /d D:\Software\researh\Project
cmake --preset release
cd build\release
nmake reduction
reduction.exe
```

在同一个构建目录中：

```bat
nmake transpose
transpose.exe
nmake
nmake help
nmake clean
```

分别表示构建转置目标、运行转置程序、构建所有目标、查看可用目标、清理产物。修改源码后重新执行 `nmake 目标名`，只重编译受影响的文件。

也可以在项目根目录让 CMake 调用同一个 NMake：

```bat
cmake --build --preset release --target transpose
build\release\transpose.exe
```

Debug 使用独立目录：

```bat
cmake --preset debug
cmake --build --preset debug --target reduction
build\debug\reduction.exe
```

不使用 preset 时的等价命令：

```bat
cmake -S . -B build\release -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
```

## 新增算子

例如新增 `02_scan/scan.cu`（包含自己的 `main()`），只需在根 `CMakeLists.txt` 末尾添加：

```cmake
add_cuda_operator(scan 02_scan/scan.cu)
```

多个源文件可写在同一行：

```cmake
add_cuda_operator(scan 02_scan/main.cu 02_scan/scan.cu)
```

两个例子择一使用。公共头文件路径、C++17、CUDA 架构、`-lineinfo` 和 MSVC UTF-8 选项统一配置，无需重复。每个可执行目标只有一个 `main()`。如需链接库，仍在根文件中按目标添加，例如 `target_link_libraries(scan PRIVATE ...)`。

然后在项目根目录执行：

```bat
cmake --preset release
cmake --build --preset release --target scan
build\release\scan.exe
```

不需要改运行配置或增加脚本。CMake 显式登记源文件，不会把草稿或归档源码自动纳入构建。

## VS Code

工作区默认终端为 **CUDA Developer PowerShell (x64)**，启动时加载 Build Tools 官方开发环境，提供 `cl`、`nmake` 和 SDK 路径。配置更新不会改变已经打开的终端：先用终端面板垃圾桶关闭旧终端，再点击 `+` 新建；若仍打开其他终端，用下拉菜单选择该配置。

在新建终端中直接执行以下 PowerShell 命令：

```powershell
cmake --preset release
cmake --build --preset release
.\build\release\reduction.exe
.\build\release\transpose.exe
```

也可以在配置完成后进入 `build\release`，执行 `nmake reduction` 或 `nmake transpose`，然后用 `.\reduction.exe` 或 `.\transpose.exe` 运行。PowerShell 运行当前目录程序必须使用 `.\` 前缀。

如果要继续使用已有的普通 PowerShell 终端，手动加载一次当前会话的开发环境即可：

```powershell
Import-Module 'D:\Software\BuildTools\Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
Enter-VsDevShell -VsInstallPath 'D:\Software\BuildTools' -SkipAutomaticLocation -Arch amd64 -HostArch amd64
```

`cmake.useVsDeveloperEnvironment` 仅配置 CMake Tools 扩展启动的进程，不会配置手动打开的普通 PowerShell。不要使用 `bash run.sh` 执行这里的 CMD/PowerShell 示例：Bash 会把 `build\release` 中的反斜杠作为转义符；这也不是本项目的构建入口。

安装推荐的 C/C++ 与 CMake Tools 扩展：

1. `CMake: Select Configure Preset` 选择 `release` 或 `debug`，执行 Configure。
2. 选择对应 Build Preset，并用 `CMake: Set Build Target` 选择目标；Ctrl+Shift+B 构建。
3. `CMake: Set Debug Target` 选择 `reduction` 或 `transpose`，使用 `CMake: Run Without Debugging` 运行。
4. 调试主机代码时选择 `debug` 的 configure/build preset，再按 F5。程序路径由 CMake Tools 提供，新增目标无需修改 `launch.json`。

工作区已启用 CMake Tools 的开发者环境加载。若编辑器找不到独立 CMake，重新打开 VS Code 以继承安装后的系统 PATH，或从上述开发者命令提示符启动它。

## 目录与环境

| 路径 | 用途 |
|---|---|
| `CMakeLists.txt` | 公共编译配置和全部算子目标登记 |
| `CMakePresets.json` | NMake Release/Debug 配置 |
| `00_transpose/transpose.cu` | `transpose` 可执行目标 |
| `01_reduction/reduction.cu` | `reduction` 目标，默认执行 shared NHWC softmax |
| `common/` | CPU 参考实现、CUDA 检查、计时与基准测试 |
| `docs/` | 环境说明、历史教程与验证记录 |
| `archive/` | 不参与构建的备用实现 |
| `reports/` | Nsight Compute 报告 |
| `build/release`、`build/debug` | Makefile、对象文件与可执行程序 |

依赖：Build Tools 的 MSVC/NMake、Windows SDK、CUDA Toolkit 13.4、CMake 3.28+。当前 GPU 是 RTX 5070 Ti，架构默认 `120`；换 GPU 可在配置命令追加 `-DCMAKE_CUDA_ARCHITECTURES=native`。

[环境配置](docs/SETUP.md) · [本机环境与验证](docs/ENVIRONMENT.md) · [历史 CMake 教程](docs/CMake_Tutorial.md)（旧路径与构建方式以本 README 为准）。

## 验证与性能分析

从项目根目录直接调用工具，以下仍为 CMD 语法：

```bat
"%CUDA_PATH%\compute-sanitizer\compute-sanitizer.exe" --tool memcheck --error-exitcode 1 build\release\reduction.exe

"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.1\target\windows-desktop-win7-x64\ncu.exe" --set basic --kernel-name regex:shared_mem_softmax_NHWC --launch-skip 3 --launch-count 1 --export reports\softmax build\release\reduction.exe
```

如果 ncu 报临时目录锁错误，可先在当前 CMD 中执行 `set "TEMP=%CD%\build"` 和 `set "TMP=%CD%\build"`。报告查看、指标选择均使用 Nsight 自身参数。

当前没有独立测试套件。原 shared softmax 存在指数溢出、NaN 比较漏检及尾部行保护缺失，因此打印 `PASS` 或 memcheck 通过不能单独证明数值正确。算法源码保持原样。性能测试使用 Release，不将 Sanitizer/Nsight 下的耗时与普通执行混用。
