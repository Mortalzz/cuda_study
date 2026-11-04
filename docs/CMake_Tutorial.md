# CMake + CUDA 入门教程

> 历史教程：下文使用旧 Visual Studio 生成器、旧 preset 和旧路径，不是当前项目的运行步骤。当前采用 CMake + NMake，新增算子在根 `CMakeLists.txt` 登记；请按 [README](../README.md) 和 [环境配置](SETUP.md) 操作。

> 基于本机环境：Windows 11 + Visual Studio 2026 + CUDA 13.4 + RTX 5070 Ti。  
> 教程中的路径以 `D:\Software\CUDAC\Project` 为例。

---

## 目录

1. [先建立正确的心智模型](#1-先建立正确的心智模型)
2. [三个目录，三个动作](#2-三个目录三个动作)
3. [最小 CMake 工程：C++](#3-最小-cmake-工程c)
4. [最小 CMake 工程：CUDA](#4-最小-cmake-工程cuda)
5. [逐行读懂你的 Project](#5-逐行读懂你的-project)
6. [Target：CMake 的核心](#6-targetcmake-的核心)
7. [include / link / options](#7-include--link--options)
8. [PRIVATE / PUBLIC / INTERFACE](#8-private--public--interface)
9. [CMake 语言速成](#9-cmake-语言速成)
10. [CUDA 专用配置](#10-cuda-专用配置)
11. [CMakePresets](#11-cmakepresets)
12. [Debug/Release 与生成器](#12-debugrelease-与生成器)
13. [新增一个 CUDA 练习](#13-新增一个-cuda-练习)
14. [公共头文件库](#14-公共头文件库)
15. [VS Code 工作流](#15-vs-code-工作流)
16. [Visual Studio 工作流](#16-visual-studio-工作流)
17. [常用命令](#17-常用命令)
18. [常见错误](#18-常见错误)
19. [最佳实践](#19-最佳实践)
20. [练习](#20-练习)
21. [下一步学什么](#21-下一步学什么)
22. [速查表](#22-速查表)

---

## 1. 先建立正确的心智模型

### 1.1 CMake 不是编译器

CMake 本身不编译 C++，也不编译 CUDA。它做的是：

> 读取 `CMakeLists.txt`，检测编译器和依赖，生成另一套构建文件；真正编译的是 MSBuild、Ninja、nvcc、cl.exe。

流程：

```text
CMakeLists.txt
      │
      │  cmake configure
      ▼
Visual Studio 工程 / Ninja 文件
      │
      │  cmake --build
      ▼
MSBuild / Ninja
      │
      ├── nvcc  → CUDA 代码
      ├── cl    → 主机 C++ 代码
      └── link  → exe
```

所以：

- `CMakeLists.txt` 是“构建配方”
- `cmake --preset ...` 是“读配方，生成工程”
- `cmake --build ...` 是“执行工程，编译链接”
- `.vcxproj` / `.slnx` 是生成物，不是手写源文件

### 1.2 CMake 不做什么

CMake 不是：

- 编译器
- 包管理器
- 代码生成器
- 调试器
- 运行器

它不负责安装 CUDA，不负责调试 kernel，也不负责运行 exe。  
它只负责：**把“要编译哪些文件、用什么选项、链接哪些库”描述清楚。**

### 1.3 CMakeLists 在 configure 阶段执行

`CMakeLists.txt` 不是 Makefile，也不是在每次 build 时逐行执行。  
它是在 **configure 阶段** 被 CMake 解释的脚本。

所以：

- 改了 `CMakeLists.txt` → 要重新 configure
- 只改了 `.cu` / `.cpp` → 直接 build
- `message(STATUS "...")` 只在 configure 时打印
- `if()`、`foreach()`、`function()` 也是 configure 时执行

---

## 2. 三个目录，三个动作

### 2.1 三个目录

| 目录 | 例子 | 作用 |
|---|---|---|
| 源码目录 | `D:\Software\CUDAC\Project` | 放 `CMakeLists.txt`、`.cu`、头文件 |
| 构建目录 | `D:\Software\CUDAC\Project\build` | 放生成的工程、obj、exe、缓存 |
| 安装目录 | `D:\Software\CUDAC\Project\install` | 可选，`cmake --install` 用 |

推荐始终“out-of-source build”：

```text
Project/
  CMakeLists.txt
  File.cu
  common/
  build/          ← 所有生成物
```

不要把生成物丢在源码目录里。

### 2.2 三个动作

```powershell
# 1. configure：生成构建文件
cmake --preset vs2026-x64

# 2. build：编译链接
cmake --build --preset release

# 3. run：运行
.\build\Release\transpose.exe
```

### 2.3 build 目录可以随时删

`build/` 里的东西都是生成物。  
遇到 CMake 缓存、编译器路径、生成器问题时，最省事的处理往往是：

```powershell
Remove-Item -Recurse -Force .\build
cmake --preset vs2026-x64
cmake --build --preset release
```

或者：

```powershell
cmake --fresh --preset vs2026-x64
```

`--fresh` 会删除缓存并重新 configure，但不会删除源码。

---

## 3. 最小 CMake 工程：C++

目录：

```text
hello/
  CMakeLists.txt
  main.cpp
```

`CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.28)
project(hello LANGUAGES CXX)

add_executable(hello main.cpp)
```

`main.cpp`：

```cpp
#include <cstdio>

int main()
{
    std::printf("hello cmake\n");
    return 0;
}
```

配置和构建：

```powershell
cd D:\Software\CUDAC\hello
cmake -S . -B build
cmake --build build --config Release
.\build\Release\hello.exe
```

### 命令解释

```text
-S .        源码目录是当前目录
-B build    构建目录是 build
--build     让 CMake 调用底层构建系统
--config    多配置生成器选择 Release
```

### 为什么有 `build\Release`

因为这里用的是 Visual Studio 生成器，它是 **多配置生成器**：

```text
build\
  Debug\
    hello.exe
  Release\
    hello.exe
```

所以必须用 `--config Release` 选配置。  
如果是 Ninja 这种单配置生成器，则是在 configure 时用 `-DCMAKE_BUILD_TYPE=Release` 决定。

---

## 4. 最小 CMake 工程：CUDA

目录：

```text
cuda_hello/
  CMakeLists.txt
  main.cu
```

`CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.28)

set(CMAKE_CUDA_ARCHITECTURES 120)

project(cuda_hello LANGUAGES CXX CUDA)

add_executable(cuda_hello main.cu)
```

`main.cu`：

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel()
{
    printf("hello from GPU, block %d thread %d\n",
           blockIdx.x, threadIdx.x);
}

int main()
{
    hello_kernel<<<1, 4>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
```

构建：

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release
.\build\Release\cuda_hello.exe
```

### 关键点

- `LANGUAGES CXX CUDA`：让 CMake 启用 CUDA，并识别 `.cu`
- `CMAKE_CUDA_ARCHITECTURES 120`：目标 GPU 是 `sm_120`
- 不需要手动写 `-gencode=arch=compute_120,code=sm_120`
- CMake 会自动把 `.cu` 交给 `nvcc`

验证：

```powershell
& "$env:CUDA_PATH\bin\cuobjdump.exe" -sass .\build\Release\cuda_hello.exe |
  Select-String "arch = sm_"
```

应该看到 `arch = sm_120`。

---

## 5. 逐行读懂你的 Project

### 5.1 目录结构

```text
D:\Software\CUDAC\Project\
  CMakeLists.txt
  CMakePresets.json
  README.md
  File.cu
  common\
    cuda_check.cuh
    timer.cuh
    reference.h
  01_reduction\
    naive_reduction.cu
  02_scan\
  03_stencil\
  04_sgemm\
  capstone\
  build\                ← CMake 生成
```

目前真正参与构建的只有：

```text
File.cu  →  target: transpose
```

`01_reduction` 等目录先放着，等源码有 `main()` 再加 target。

### 5.2 当前 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.28)

set(CMAKE_CUDA_ARCHITECTURES 120)

project(CudaLab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(transpose File.cu)
target_include_directories(transpose PRIVATE common)
target_compile_options(transpose PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)
```

逐行解释：

#### `cmake_minimum_required(VERSION 3.28)`

要求 CMake 至少 3.28。  
本机 CMake 是 4.3.1，满足要求。

#### `set(CMAKE_CUDA_ARCHITECTURES 120)`

设置所有 CUDA target 的默认架构为 `sm_120`。  
放在 `project()` 前，是为了在启用 CUDA 时就用这个值。

CMake 会根据它生成类似：

```text
--generate-code=arch=compute_120,code=[compute_120,sm_120]
```

这表示同时生成：

- `compute_120` 的 PTX
- `sm_120` 的原生 SASS

所以不需要再手写 `-gencode`。

#### `project(CudaLab LANGUAGES CXX CUDA)`

项目名 `CudaLab`，启用 C++ 和 CUDA。  
`CXX` 负责 `.cpp`，`CUDA` 负责 `.cu`。

#### `set(CMAKE_CXX_STANDARD 17)` / `set(CMAKE_CUDA_STANDARD 17)`

主机 C++ 和 CUDA C++ 都用 C++17。  
C++20 也可以，但学习阶段 C++17 足够。

#### `add_executable(transpose File.cu)`

创建一个可执行 target：

```text
target 名：transpose
源文件：File.cu
输出：build\Release\transpose.exe
```

#### `target_include_directories(transpose PRIVATE common)`

只给 `transpose` 添加 `common` 头文件目录。  
`File.cu` 里以后写：

```cpp
#include "cuda_check.cuh"
```

就能找到 `common\cuda_check.cuh`。

#### `target_compile_options(...)`

```cmake
$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
```

只有编译 CUDA 文件时才加 `-lineinfo`。

```cmake
$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
```

只有编译 CUDA 文件时，把 `/utf-8` 传给主机编译器 MSVC，消除 C4819 警告。

### 5.3 当前 CMakePresets.json

```json
{
  "version": 6,
  "cmakeMinimumRequired": {
    "major": 3,
    "minor": 28,
    "patch": 0
  },
  "configurePresets": [
    {
      "name": "vs2026-x64",
      "displayName": "Visual Studio 2026 x64",
      "generator": "Visual Studio 18 2026",
      "architecture": "x64",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "CMAKE_CUDA_ARCHITECTURES": "120"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "release",
      "configurePreset": "vs2026-x64",
      "configuration": "Release"
    },
    {
      "name": "debug",
      "configurePreset": "vs2026-x64",
      "configuration": "Debug"
    }
  ]
}
```

它等价于手写：

```powershell
cmake -S . -B build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -DCMAKE_CUDA_ARCHITECTURES=120

cmake --build build --config Release
```

Presets 只是把这些参数保存下来，方便复用。

### 5.4 生成物在哪里

configure 之后：

```text
build\
  CudaLab.slnx
  transpose.vcxproj
  ALL_BUILD.vcxproj
  ZERO_CHECK.vcxproj
  CMakeCache.txt
```

build 之后：

```text
build\Release\transpose.exe
```

注意：

- `CudaLab.slnx`、`transpose.vcxproj` 是 CMake 生成的
- 不要手改这些生成文件
- 要改设置，改 `CMakeLists.txt` 或 `CMakePresets.json`

---

## 6. Target：CMake 的核心

### 6.1 什么是 target

Target 是 CMake 管理的“构建单元”，通常是一个：

- 可执行文件
- 静态库
- 动态库
- 头文件接口库

每个 target 有自己的：

- 源文件
- include 路径
- 编译选项
- 宏定义
- 链接库

### 6.2 创建可执行文件

```cmake
add_executable(transpose File.cu)
```

一个 `main()` 对应一个 executable。  
不要把一个没有 `main()` 的 `.cu` 单独做成 executable，否则链接会报：

```text
unresolved external symbol main
```

### 6.3 添加多个源文件

```cmake
add_executable(app
    main.cu
    kernel.cu
    utils.cu
)
```

或者：

```cmake
add_executable(app main.cu)

target_sources(app PRIVATE
    kernel.cu
    utils.cu
)
```

两种都可以。  
学习阶段推荐把源文件直接写在 `add_executable` 里，直观。

### 6.4 添加一个 reduction target

等 `01_reduction/naive_reduction.cu` 写好 `main()` 后，在根 `CMakeLists.txt` 加：

```cmake
add_executable(naive_reduction
    01_reduction/naive_reduction.cu
)

target_include_directories(naive_reduction PRIVATE common)

target_compile_options(naive_reduction PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)
```

然后：

```powershell
cmake --preset vs2026-x64
cmake --build --preset release
.\build\Release\naive_reduction.exe
```

### 6.5 用子目录管理

如果每个主题都有自己的 `CMakeLists.txt`：

```text
Project/
  CMakeLists.txt
  01_reduction/
    CMakeLists.txt
    naive_reduction.cu
```

根文件：

```cmake
add_subdirectory(01_reduction)
```

`01_reduction/CMakeLists.txt`：

```cmake
add_executable(naive_reduction naive_reduction.cu)

target_include_directories(naive_reduction PRIVATE
    ${PROJECT_SOURCE_DIR}/common
)

target_compile_options(naive_reduction PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)
```

`${PROJECT_SOURCE_DIR}` 指向根项目目录。  
因为子目录里的相对路径是相对于 `01_reduction` 的，不能直接写 `common`。

---

## 7. include / link / options

### 7.1 头文件目录

```cmake
target_include_directories(app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
```

- `PRIVATE`：只有 `app` 能用这个目录
- `PUBLIC`：`app` 和依赖 `app` 的 target 都能用
- `INTERFACE`：只有依赖 `app` 的 target 能用

### 7.2 链接库

```cmake
target_link_libraries(app PRIVATE mylib)
```

`target_link_libraries` 不只是“加一个 .lib”，它还会传播：

- include 路径
- 编译选项
- 依赖库
- 链接顺序

这就是“现代 CMake”的核心：**依赖关系挂在 target 上，而不是全局变量上。**

### 7.3 编译选项

```cmake
target_compile_options(app PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
```

常用：

```cmake
target_compile_options(app PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g>
)
```

但 CUDA 的优化级别通常由 build type 自动处理，不需要手写 `-O3`。

### 7.4 宏定义

```cmake
target_compile_definitions(app PRIVATE
    NDEBUG
    CUDA_LAB=1
)
```

对应代码：

```cpp
#ifdef CUDA_LAB
...
#endif
```

### 7.5 不要用全局命令

不推荐：

```cmake
include_directories(common)
link_libraries(mylib)
add_definitions(-DFOO)
```

这些会影响后面所有 target，容易造成隐式依赖。

推荐：

```cmake
target_include_directories(app PRIVATE common)
target_link_libraries(app PRIVATE mylib)
target_compile_definitions(app PRIVATE FOO)
```

---

## 8. PRIVATE / PUBLIC / INTERFACE

| 关键字 | 当前 target 使用 | 依赖它的 target 也使用 |
|---|---|---|
| `PRIVATE` | 是 | 否 |
| `PUBLIC` | 是 | 是 |
| `INTERFACE` | 否 | 是 |

### 8.1 可执行文件

```cmake
add_executable(app main.cu)
target_include_directories(app PRIVATE include)
target_link_libraries(app PRIVATE mylib)
```

可执行文件不对外提供接口，所以几乎都用 `PRIVATE`。

### 8.2 静态库

```cmake
add_library(mylib STATIC mylib.cu)

target_include_directories(mylib PUBLIC include)
target_link_libraries(mylib PRIVATE some_dep)
```

- `PUBLIC include`：因为使用 `mylib` 的人也需要这些头文件
- `PRIVATE some_dep`：`some_dep` 只是 `mylib` 内部使用，不暴露给使用者

### 8.3 头文件库

`common/` 里只有 `.cuh` / `.h`，没有 `.cu`，适合做成 interface library：

```cmake
add_library(cuda_common INTERFACE)

target_include_directories(cuda_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
```

然后：

```cmake
target_link_libraries(transpose PRIVATE cuda_common)
```

`cuda_common` 不生成 `.lib`，只是把 include 路径传给使用者。  
这就是 `INTERFACE` 的典型用法。

### 8.4 CUDA 静态库的额外注意

如果静态库里跨多个 `.cu` 文件调用 `__device__` 函数，可能需要：

```cmake
set_target_properties(mylib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
```

学习阶段先不要拆 CUDA 库。  
**一个主题一个 executable** 最简单，等确实需要复用再拆。

---

## 9. CMake 语言速成

### 9.1 命令

CMake 命令长这样：

```cmake
command(arg1 arg2 arg3)
```

命令名大小写不敏感，但习惯全小写。  
变量区分大小写。

### 9.2 变量

```cmake
set(MY_SOURCES main.cu kernel.cu)
add_executable(app ${MY_SOURCES})
```

`${VAR}` 展开变量。  
列表用分号分隔，但也可以多行写。

### 9.3 缓存变量

```cmake
set(MY_OPTION ON CACHE BOOL "enable something")
```

缓存变量会写进 `build/CMakeCache.txt`。  
命令行可以覆盖：

```powershell
cmake -S . -B build -DMY_OPTION=OFF
```

普通 `set(VAR VALUE)` 不会持久化。

### 9.4 option

```cmake
option(CUDA_LAB_BUILD_TESTS "build tests" OFF)

if(CUDA_LAB_BUILD_TESTS)
    enable_testing()
endif()
```

命令行：

```powershell
cmake --preset vs2026-x64 -DCUDA_LAB_BUILD_TESTS=ON
```

### 9.5 if / foreach / function

```cmake
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
endif()

foreach(src IN LISTS MY_SOURCES)
    message(STATUS "source: ${src}")
endforeach()
```

函数：

```cmake
function(add_cuda_exe name)
    add_executable(${name} ${ARGN})

    target_link_libraries(${name} PRIVATE cuda_common)

    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endfunction()
```

调用：

```cmake
add_cuda_exe(transpose File.cu)
add_cuda_exe(naive_reduction 01_reduction/naive_reduction.cu)
```

`ARGN` 是函数除第一个参数外的其余参数。

### 9.6 生成器表达式

```cmake
$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
```

读法：

> 如果当前编译语言是 CUDA，就加 `-lineinfo`。

常用：

```cmake
$<$<CONFIG:Release>:-O3>
$<$<CONFIG:Debug>:-g>
$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
```

生成器表达式在 **generate 阶段** 求值，不是 configure 阶段。

### 9.7 configure-time 和 build-time

| 文件 | 什么时候处理 |
|---|---|
| `CMakeLists.txt` | configure |
| `CMakePresets.json` | configure 前 |
| `.cu` / `.cpp` | build |
| `CMakeCache.txt` | configure 生成，build 读取 |

所以：

- 改 `CMakeLists.txt` → 重新 configure
- 改 `.cu` → 直接 build
- 改 `CMakePresets.json` → 重新选 preset / configure

### 9.8 常用路径变量

| 变量 | 含义 |
|---|---|
| `CMAKE_SOURCE_DIR` | 最顶层源码目录 |
| `PROJECT_SOURCE_DIR` | 当前 project() 的源码目录 |
| `CMAKE_CURRENT_SOURCE_DIR` | 当前 CMakeLists 所在目录 |
| `CMAKE_BINARY_DIR` | 最顶层 build 目录 |
| `CMAKE_CURRENT_BINARY_DIR` | 当前 CMakeLists 对应的 build 目录 |

在根项目里，`PROJECT_SOURCE_DIR` 和 `CMAKE_SOURCE_DIR` 通常一样。  
在子目录里，`CMAKE_CURRENT_SOURCE_DIR` 会变。

---

## 10. CUDA 专用配置

### 10.1 启用 CUDA

```cmake
project(CudaLab LANGUAGES CXX CUDA)
```

只写 `CXX` 不会编译 `.cu`。  
必须加 `CUDA`。

### 10.2 设置架构

```cmake
set(CMAKE_CUDA_ARCHITECTURES 120)
```

RTX 5070 Ti 是 Blackwell，Compute Capability 12.0，对应 `sm_120`。

也可以只给某个 target 设置：

```cmake
set_target_properties(transpose PROPERTIES
    CUDA_ARCHITECTURES 120
)
```

多个架构：

```cmake
set(CMAKE_CUDA_ARCHITECTURES "90;120")
```

学习阶段单卡用 `120` 就够。

### 10.3 CUDA 标准

```cmake
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
```

主机 C++ 标准用 `CMAKE_CXX_STANDARD`，CUDA 用 `CMAKE_CUDA_STANDARD`，两个都设。

### 10.4 `-lineinfo` 和 `-G`

| 选项 | 作用 | 对性能影响 |
|---|---|---|
| `-lineinfo` | 行号映射，保留优化 | 很小 |
| `-G` | 设备端完整调试信息，关闭优化 | 很大 |

性能测试和 ncu 用 Release + `-lineinfo`。  
需要设备断点时用 Debug + `-G`。

### 10.5 使用 cuBLAS 等 CUDA 库

```cmake
find_package(CUDAToolkit REQUIRED)

add_executable(sgemm 04_sgemm/sgemm.cu)

target_link_libraries(sgemm PRIVATE
    CUDA::cublas
)
```

常用 imported target：

```text
CUDA::cudart
CUDA::cublas
CUDA::cufft
CUDA::curand
CUDA::cusolver
CUDA::cusparse
```

如果 CUDA 不在默认路径：

```cmake
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4")
find_package(CUDAToolkit REQUIRED)
```

或者在 configure 时：

```powershell
cmake --preset vs2026-x64 `
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4"
```

### 10.6 验证原生架构

```powershell
& "$env:CUDA_PATH\bin\cuobjdump.exe" -sass .\build\Release\transpose.exe |
  Select-String "arch = sm_"
```

看到 `arch = sm_120` 才算原生。

---

## 11. CMakePresets

### 11.1 为什么用 presets

没有 presets 时，每次都要敲：

```powershell
cmake -S . -B build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -DCMAKE_CUDA_ARCHITECTURES=120

cmake --build build --config Release
```

Presets 把这些参数存进 JSON，变成：

```powershell
cmake --preset vs2026-x64
cmake --build --preset release
```

### 11.2 configurePresets

```json
{
  "name": "vs2026-x64",
  "generator": "Visual Studio 18 2026",
  "architecture": "x64",
  "binaryDir": "${sourceDir}/build",
  "cacheVariables": {
    "CMAKE_CUDA_ARCHITECTURES": "120"
  }
}
```

字段：

- `name`：preset 名字
- `generator`：生成器
- `architecture`：`x64`
- `binaryDir`：build 目录
- `cacheVariables`：等价于 `-D...`

### 11.3 buildPresets

```json
{
  "name": "release",
  "configurePreset": "vs2026-x64",
  "configuration": "Release"
}
```

对应：

```powershell
cmake --build build --config Release
```

### 11.4 常用命令

```powershell
# 列出 configure presets
cmake --list-presets

# 配置
cmake --preset vs2026-x64

# 构建 Release
cmake --build --preset release

# 构建 Debug
cmake --build --preset debug

# 清理后重建
cmake --build --preset release --clean-first

# 删除缓存重新配置
cmake --fresh --preset vs2026-x64
```

### 11.5 CMakeUserPresets.json

个人本机路径、临时选项可以放：

```text
CMakeUserPresets.json
```

它不应该提交到仓库。  
团队共享的配置放 `CMakePresets.json`。

### 11.6 必须在 Project 目录运行

你的 `CMakePresets.json` 在：

```text
D:\Software\CUDAC\Project\CMakePresets.json
```

所以命令要在 `Project` 目录执行：

```powershell
cd D:\Software\CUDAC\Project
cmake --preset vs2026-x64
```

如果在 `D:\Software\CUDAC` 执行，会报：

```text
Could not read presets ... File not found
```

---

## 12. Debug/Release 与生成器

### 12.1 多配置生成器：Visual Studio

```text
build\
  Debug\
  Release\
```

构建时选：

```powershell
cmake --build build --config Release
```

`CMAKE_BUILD_TYPE` 对 VS 生成器通常无效。  
不要写：

```cmake
if(CMAKE_BUILD_TYPE STREQUAL "Release")  # 在 VS 多配置下可能为空
```

要在多配置下判断，用生成器表达式：

```cmake
$<$<CONFIG:Release>:...>
```

### 12.2 单配置生成器：Ninja

```powershell
cmake -S . -B build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-ninja
```

Ninja 一个 build 目录只对应一个配置。  
如果要在 Windows 上用 Ninja + MSVC，最好从：

```text
x64 Native Tools Command Prompt for VS 2026
```

启动，或者确保 `cl.exe` 在 PATH 里。

### 12.3 Debug 和 Release 的区别

| 配置 | 典型选项 | 用途 |
|---|---|---|
| Debug | `/Od`、`-G`、`_DEBUG` | 断点调试 |
| Release | `/O2`、`NDEBUG` | 性能测试、ncu、交付 |

CUDA 里：

- Debug 可能带 `-G`，关闭优化
- Release 优化打开
- 需要源码对应时，Release 加 `-lineinfo`

### 12.4 清理

```powershell
# 只清理构建产物，不删配置
cmake --build --preset release --clean-first

# 删除整个 build 目录，重新 configure
Remove-Item -Recurse -Force .\build
cmake --preset vs2026-x64
```

`--fresh` 和 `--clean-first` 不是一回事：

- `--fresh`：重新 configure，清缓存
- `--clean-first`：保留配置，先 clean 再 build

---

## 13. 新增一个 CUDA 练习

以 `01_reduction/naive_reduction.cu` 为例。

### 13.1 先确保源码有 main

```cpp
int main()
{
    // ...
    return 0;
}
```

没有 `main()` 不能做 executable。

### 13.2 在根 CMakeLists 添加 target

```cmake
add_executable(naive_reduction
    01_reduction/naive_reduction.cu
)

target_include_directories(naive_reduction PRIVATE common)

target_compile_options(naive_reduction PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)
```

### 13.3 重新配置和构建

```powershell
cd D:\Software\CUDAC\Project
cmake --preset vs2026-x64
cmake --build --preset release
```

### 13.4 运行

```powershell
.\build\Release\naive_reduction.exe
```

### 13.5 如果编译多个文件

```cmake
add_executable(naive_reduction
    01_reduction/main.cu
    01_reduction/naive_reduction.cu
)
```

一个 target 只能有一个 `main()`。

### 13.6 如果用子目录

根：

```cmake
add_subdirectory(01_reduction)
```

`01_reduction/CMakeLists.txt`：

```cmake
add_executable(naive_reduction naive_reduction.cu)

target_include_directories(naive_reduction PRIVATE
    ${PROJECT_SOURCE_DIR}/common
)

target_compile_options(naive_reduction PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)
```

注意：子目录里的 `common` 相对路径不对，要用 `${PROJECT_SOURCE_DIR}/common` 或 `cuda_common`。

---

## 14. 公共头文件库

### 14.1 定义 interface library

在根 `CMakeLists.txt`：

```cmake
add_library(cuda_common INTERFACE)

target_include_directories(cuda_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
```

### 14.2 让 target 使用它

```cmake
target_link_libraries(transpose PRIVATE cuda_common)
```

之后每个新 target：

```cmake
add_executable(naive_reduction 01_reduction/naive_reduction.cu)
target_link_libraries(naive_reduction PRIVATE cuda_common)
```

就不需要重复写 `target_include_directories`。

### 14.3 标准设置

统一在根文件设置：

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
```

这样所有 target 默认使用 C++17。

### 14.4 统一编译选项

可以写一个函数：

```cmake
function(add_cuda_exe name)
    add_executable(${name} ${ARGN})

    target_link_libraries(${name} PRIVATE cuda_common)

    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endfunction()
```

调用：

```cmake
add_cuda_exe(transpose File.cu)
add_cuda_exe(naive_reduction 01_reduction/naive_reduction.cu)
```

这就是一个小型“项目框架”。  
先理解显式写法，再用函数减少重复。

---

## 15. VS Code 工作流

### 15.1 打开正确目录

打开：

```text
D:\Software\CUDAC\Project
```

不是：

```text
D:\Software\CUDAC
```

因为 `CMakeLists.txt` 和 `CMakePresets.json` 在 `Project` 里。

### 15.2 选择 preset

CMake Tools 会自动读取 `CMakePresets.json`。  
状态栏或命令面板：

```text
CMake: Select Configure Preset → vs2026-x64
```

### 15.3 选择 Release

状态栏 Variant 选：

```text
Release
```

或者命令面板：

```text
CMake: Select Variant → Release
```

### 15.4 构建

```text
CMake: Build
```

或：

```text
Terminal → Run Task → CMake: build Release
```

### 15.5 运行

```text
Terminal → Run Task → Run transpose (Release)
```

或者终端：

```powershell
.\build\Release\transpose.exe
```

### 15.6 调试

`F5` 使用 `.vscode/launch.json`：

```text
Debug transpose (host only)
```

它只调试主机代码。  
CUDA 设备端断点需要 Visual Studio + Nsight VSE。

### 15.7 常见现象

- `.cu` 里的 `<<< >>>` 可能被 C/C++ 扩展标红  
  这是 IntelliSense 限制，不一定影响编译。
- CMake Tools 找不到 preset  
  确认打开的是 `Project` 目录。
- CMake 不在 PATH  
  `tasks.json` 已经写死 VS 自带 CMake 的完整路径。
- `code` 命令不在 PATH  
  用完整路径或把 `D:\Software\Vscode\Microsoft VS Code\bin` 加进 PATH。

---

## 16. Visual Studio 工作流

CMake 用 VS 生成器时，会生成：

```text
build\CudaLab.slnx
build\transpose.vcxproj
```

需要设备端调试时，可以用 Visual Studio 打开：

```text
D:\Software\CUDAC\Project\build\CudaLab.slnx
```

然后：

- 选择 `Release | x64` 或 `Debug | x64`
- 用 Nsight VSE 调试 CUDA
- 不要手改生成的 `.vcxproj`

如果改了 `CMakeLists.txt`，VS 里的 `ZERO_CHECK` 会检测到并重新 configure。  
但更稳的做法是回 VS Code / 命令行重新 configure。

**CMake 是唯一事实来源。**  
你在 VS 里改的生成工程设置，可能会被下一次 configure 覆盖。

---

## 17. 常用命令

```powershell
# 版本
cmake --version

# 配置
cmake --preset vs2026-x64

# 构建 Release
cmake --build --preset release

# 构建 Debug
cmake --build --preset debug

# 只构建某个 target
cmake --build --preset release --target transpose

# 清理后构建
cmake --build --preset release --clean-first

# 删除缓存并重新配置
cmake --fresh --preset vs2026-x64

# 列出 configure presets
cmake --list-presets

# 列出 build presets
cmake --list-presets=build

# 打开生成的 VS 工程
cmake --open build

# 运行
.\build\Release\transpose.exe
```

手动配置（不用 preset）：

```powershell
cmake -S . -B build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -DCMAKE_CUDA_ARCHITECTURES=120

cmake --build build --config Release
```

---

## 18. 常见错误

### 18.1 找不到 CUDA 编译器

```text
No CMAKE_CUDA_COMPILER could be found
```

原因：

- `nvcc` 不在 PATH
- CUDA 没装
- VS Code 没继承环境变量

处理：

```powershell
where.exe nvcc
```

如果找不到，确认：

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin
```

在 PATH 里。

也可以显式设置：

```powershell
cmake --preset vs2026-x64 `
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe"
```

### 18.2 找不到源文件

```text
Cannot find source file
```

原因：`add_executable` 里的路径相对于当前 `CMakeLists.txt`。

处理：

```cmake
add_executable(app 01_reduction/main.cu)
```

不是相对于 `build`，也不是相对于 VS Code 打开的目录。

### 18.3 没有设置 CUDA 架构

```text
Target requires CUDA architecture
```

处理：

```cmake
set(CMAKE_CUDA_ARCHITECTURES 120)
```

### 18.4 找不到 exe

多配置 VS 生成器的输出在：

```text
build\Release\transpose.exe
```

不是：

```text
build\transpose.exe
```

### 18.5 改了生成工程没生效

你可能改的是：

```text
build\transpose.vcxproj
```

这是生成物。  
应该改：

```text
CMakeLists.txt
```

然后重新 configure。

### 18.6 缓存导致怪问题

```powershell
cmake --fresh --preset vs2026-x64
```

或者：

```powershell
Remove-Item -Recurse -Force .\build
cmake --preset vs2026-x64
```

### 18.7 配置成功但构建失败

说明 CMake 已经找到编译器，问题在：

- 源码语法
- 编译选项
- 链接库
- 缺少 `main()`

看构建日志里的 `nvcc` / `cl` 命令和第一条 error。

### 18.8 Target 重名

```text
add_executable(foo ...)
add_executable(foo ...)
```

target 名必须唯一。  
不同目录里的 target 也在同一个全局命名空间。

### 18.9 没有 main

```text
unresolved external symbol main
```

一个 executable 必须有一个 `main()`。  
没有 `main()` 的文件可以做成 library，或先不要加入 executable。

### 18.10 路径有空格

CMake 里用引号：

```cmake
set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4")
```

CMake 接受 Windows 正斜杠 `/`，比反斜杠更省事。

---

## 19. 最佳实践

1. **一个 executable 一个 `main()`**
2. **一个主题一个 target**
3. **优先用 `target_*` 命令**
4. **include 路径跟着 target 走**
5. **公共头文件做成 `INTERFACE` library**
6. **源文件显式列出，不要用 `file(GLOB *.cu)`**
7. **build 目录独立，不提交**
8. **用 presets 固定生成器、架构、配置**
9. **不要手改生成的 `.vcxproj` / `.slnx`**
10. **性能测试用 Release**
11. **需要源码对应时用 `-lineinfo`，不要用 `-G`**
12. **CUDA 架构用 `CMAKE_CUDA_ARCHITECTURES`，不要手写 `-gencode`**
13. **CMakeLists 改动后 reconfigure**
14. **遇到怪问题先删 build**
15. **CMakeLists 保持小，重复多了再写 function**

### 关于 `file(GLOB)`

不推荐：

```cmake
file(GLOB SOURCES "*.cu")
add_executable(app ${SOURCES})
```

新增文件时 CMake 不一定会自动重新 configure，容易出现“文件明明加了但没编译”。  
显式列出源文件更可靠。

### 关于全局选项

不推荐：

```cmake
include_directories(common)
link_libraries(cublas)
add_definitions(-DFOO)
```

推荐：

```cmake
target_include_directories(app PRIVATE common)
target_link_libraries(app PRIVATE CUDA::cublas)
target_compile_definitions(app PRIVATE FOO)
```

---

## 20. 练习

### 练习 1：添加 reduction target

把 `01_reduction/naive_reduction.cu` 写成带 `main()` 的程序，然后在根 `CMakeLists.txt` 加：

```cmake
add_executable(naive_reduction
    01_reduction/naive_reduction.cu
)

target_include_directories(naive_reduction PRIVATE common)
```

构建并运行：

```powershell
cmake --preset vs2026-x64
cmake --build --preset release
.\build\Release\naive_reduction.exe
```

### 练习 2：抽取 `cuda_common`

```cmake
add_library(cuda_common INTERFACE)

target_include_directories(cuda_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
```

然后：

```cmake
target_link_libraries(transpose PRIVATE cuda_common)
target_link_libraries(naive_reduction PRIVATE cuda_common)
```

### 练习 3：改成子目录

创建：

```text
01_reduction\CMakeLists.txt
```

根文件加：

```cmake
add_subdirectory(01_reduction)
```

子目录里定义 `naive_reduction` target。

### 练习 4：链接 cuBLAS

创建一个 `04_sgemm` target：

```cmake
find_package(CUDAToolkit REQUIRED)

add_executable(sgemm 04_sgemm/sgemm.cu)

target_link_libraries(sgemm PRIVATE
    cuda_common
    CUDA::cublas
)
```

### 练习 5：加一个测试

根文件：

```cmake
enable_testing()

add_test(NAME transpose_smoke
    COMMAND transpose
)
```

运行：

```powershell
cmake --preset vs2026-x64
cmake --build --preset release
ctest --test-dir build -C Release --output-on-failure
```

### 练习 6：验证原生架构

```powershell
& "$env:CUDA_PATH\bin\cuobjdump.exe" -sass .\build\Release\transpose.exe |
  Select-String "arch = sm_"
```

必须看到 `sm_120`。

---

## 21. 下一步学什么

先把下面这些用熟，就足够支撑整个 CUDA 学习项目：

- `add_executable`
- `add_library`
- `target_sources`
- `target_include_directories`
- `target_link_libraries`
- `target_compile_options`
- `target_compile_definitions`
- `CMakePresets.json`
- `find_package(CUDAToolkit)`

之后可以再学：

- `CTest`：自动化 correctness 测试
- `install()` / `export()`：打包自己的库
- `FetchContent`：拉取第三方依赖
- `CUDA_SEPARABLE_COMPILATION`：多 `.cu` 文件的 device linking
- `CMakeUserPresets.json`：本机个人配置
- `compile_commands.json`：给 clangd/IntelliSense 用

但不要现在学完。  
**CMake 是工具，不是学习目标。**  
你的主线仍然是 CUDA kernel、性能分析和项目实践。

---

## 22. 速查表

### 配置与构建

```powershell
cmake --preset vs2026-x64
cmake --build --preset release
cmake --build --preset debug
cmake --build --preset release --target transpose
cmake --build --preset release --clean-first
cmake --fresh --preset vs2026-x64
cmake --list-presets
cmake --list-presets=build
```

### CMakeLists 常用命令

```cmake
cmake_minimum_required(VERSION 3.28)
project(CudaLab LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(app main.cu)
add_library(mylib STATIC mylib.cu)
add_library(cuda_common INTERFACE)

target_sources(app PRIVATE kernel.cu)
target_include_directories(app PRIVATE include)
target_link_libraries(app PRIVATE mylib CUDA::cublas)
target_compile_options(app PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
target_compile_definitions(app PRIVATE FOO=1)

add_subdirectory(01_reduction)
```

### CUDA 相关

```cmake
project(CudaLab LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(app PRIVATE CUDA::cublas)

target_compile_options(app PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)
```

### 输出路径

```text
build\Release\transpose.exe
build\Debug\transpose.exe
```

### 一句话

> **CMakeLists 描述 target；configure 生成工程；build 调用编译器和链接器。**  
> 你只需要记住：目标是什么、源文件在哪、头文件在哪、链接什么库。

---

## 参考

- [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/)
- [cmake-presets(7)](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html)
- [CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)
- [FindCUDAToolkit](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
