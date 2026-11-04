# 最小 Windows 开发环境

本项目使用 CMake + NMake，不需要完整 Visual Studio，也不需要项目自定义脚本。

| 组件 | 本机位置 |
|---|---|
| MSVC、NMake（Build Tools 2026） | `D:\Software\BuildTools` |
| Windows SDK | `D:\Software\WindowsSDK` |
| CUDA Toolkit 13.4 | 沿用 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4` |
| CMake 4.4.4 | 沿用 `C:\Program Files\CMake` |

新增组件优先安装在 `D:\Software\组件名称`。没有新增 GNU Make、Ninja 或 CMake 副本；Windows 原生 CUDA 编译使用 MSVC，NMake 已随其安装。SDK 只选择桌面 C++ x64 功能及安装器要求的依赖。安装器和部分系统缓存固定在 C 盘。

## 开发者命令提示符

从开始菜单打开 **x64 Native Tools Command Prompt for VS**。这是 Build Tools 提供的环境入口，会配置编译器、NMake、头文件和库路径，项目自身没有脚本。

普通 CMD 中也可以手动加载一次官方环境：

```bat
call D:\Software\BuildTools\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64
```

随后检查：

```bat
where cl
where nmake
where rc
where cmake
where nvcc
```

系统 PATH 应包含 CMake 的 `bin` 与 CUDA 的 `bin`；`CUDA_PATH` 和 `CUDA_PATH_V13_4` 指向 Toolkit 根目录。安装后重新打开终端以继承系统变量，避免把特定 MSVC 版本的目录永久写入 PATH。

## 构建规则

NMake 是单配置生成器，Release/Debug 分别放在 `build/release` 与 `build/debug`，由 `CMAKE_BUILD_TYPE` 决定。切换生成器时必须使用新构建目录，不能复用旧 MSBuild 缓存。

NMake 直接使用 `cl`、`nvcc`、SDK 头文件/库和链接工具，不依赖 CUDA 的 Visual Studio/MSBuild `.props` 集成文件。已有集成文件无需为本次切换再次修改。

根 `CMakeLists.txt` 的 `add_cuda_operator()` 负责创建可执行目标和统一编译选项，新增算子在此登记。无需为每个算子建立脚本或编辑器任务。

VS Code 工作区使用 CMake Tools 的 `cmake.useVsDeveloperEnvironment=always` 加载编译环境；preset 中的 x64 为编辑器提示，CMake CLI 不会对 NMake 传入不支持的 `-A` 参数。

手动打开的集成终端不继承上述 CMake Tools 环境。本工作区另设默认终端 **CUDA Developer PowerShell (x64)**，使用 Windows 自带 PowerShell 导入 Build Tools 的 `Microsoft.VisualStudio.DevShell.dll` 并进入 x64 开发环境，不增加项目构建脚本。首次应用配置后须关闭旧终端并新建终端；当前目录 exe 在 PowerShell 中使用 `.\程序.exe` 运行。普通 PowerShell 的手动加载命令见 [README](../README.md#vs-code)。

官方资料：[CMake NMake 生成器](https://cmake.org/cmake/help/latest/generator/NMake%20Makefiles.html)、[微软 NMake 说明](https://learn.microsoft.com/en-us/cpp/build/reference/nmake-reference)、[CUDA Windows 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)。
