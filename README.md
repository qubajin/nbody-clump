# NBodyClump 核心代码包

## 目录

- `src/`：C++/CUDA 求解器，入口为 `main.cpp`，当前调用 `DEMClumpMain::RunSimulationClump()`。
- `src/include/`：随项目提供的辅助头文件及历史 CUB 头文件，完整保留以避免遗漏依赖。
- `src/world.par`：原项目自带的运行参数。
- `src/ph.00000000.bt`、`src/ch.00000000.bt`：原有粒子和球簇初始条件。
- `src/Rocks/`：边界与侵入器输入文件。
- `CMakeLists.txt`：原有构建入口。
- `BINARY_OUTPUT.md`：原有二进制输出格式说明。
- `EXTRACTION_MANIFEST.json`：156 个复制文件的原始路径、大小和 SHA-256 校验值；新增说明文件不在此清单中。
- `.gitignore`：编译产物、缓存和运行结果的排除规则。

没有提取历史算例、历史结果、视频、压缩包、编译目录、颗粒生成器和外部分析脚本。
这些仍保留在原项目中。本包保留所有现有求解器模块，没有裁剪暂未使用的计算分支。

## 编译

需要 CMake 3.16+、NVIDIA CUDA Toolkit、兼容的 C++ 编译器，以及 Boost filesystem/system 开发库。
本版本使用 C++14/CUDA C++14，仍然依赖 Boost。

在本包目录执行（将 `<GPU架构>` 替换为目标显卡对应的 CUDA 架构编号）：

```text
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=<GPU架构>
cmake --build build --config Release --parallel
```

构建完成后，CMake 会把自带输入文件复制到可执行程序所在目录。
程序使用当前工作目录读取 `world.par`、初始条件和 `Rocks/`，运行前必须切换到可执行程序所在目录。
Linux 常见位置是 `build/`；Windows 多配置生成器常见位置是 `build/Release/`。

## 参数与验证范围

自带 `world.par` 当前设置为 576 个子球、144 个球簇、时间步长 1e-6 秒、1000000 步，侵入器开启。
这是原 `src` 自带输入，不代表历史大床体算例，也不是此次新确定的团队基准实验；启动前应检查它是否符合实验目的。
参数尚未统一集中到 JSON：除输入文件外，还应留意 `dem_parameters.h`、`dem_constant.h` 及其他源码中的常量。

此次完成复制文件的 SHA-256 一致性检查和本地引用完整性检查，未重新编译或执行 GPU 模拟，未重新验证自带算例的物理有效性。
历史文档中的验证记录不应视为此次重新运行的结果。
原目录的源码、参数和结果均未修改。后续建立仓库时应以本目录为仓库根目录。
