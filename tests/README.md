# CinderX UT 详细说明书

本文档说明当前仓库统一 UT 入口、构建方式、覆盖率口径、测试分类、扩展方式，以及当前已知需要持续关注的问题。

适用仓库：
- [`C:\Users\z30059427\Desktop\cinderx\cinderx`](C:\Users\z30059427\Desktop\cinderx\cinderx)

统一入口脚本：
- [`C:\Users\z30059427\Desktop\cinderx\cinderx\tests\run_test_suites.py`](C:\Users\z30059427\Desktop\cinderx\cinderx\tests\run_test_suites.py)

## 1. 入口脚本如何使用

### 1.1 基本命令

统一入口：

```bash
python tests/run_test_suites.py [options]
```

常见示例：

```bash
# 跑全量
python tests/run_test_suites.py -t all --gcc-root /opt/openEuler/gcc-toolset-14/root

# 只跑 pythonlib
python tests/run_test_suites.py -t pythonlib --gcc-root /opt/openEuler/gcc-toolset-14/root

# 只跑 runtime
python tests/run_test_suites.py -t runtime --gcc-root /opt/openEuler/gcc-toolset-14/root

# 只列出测试，不执行
python tests/run_test_suites.py -t pythonlib -l

# 只跑单个 pythonlib 模块
python tests/run_test_suites.py -t pythonlib -f test_cinderx.test_jit_frame

# 只跑某些 RuntimeTests
python tests/run_test_suites.py -t runtime -f CmdLineTest.BasicFlags

# 跑覆盖率
python tests/run_test_suites.py -t all -c --gcc-root /opt/openEuler/gcc-toolset-14/root
```

### 1.2 主要参数

来自 [`C:\Users\z30059427\Desktop\cinderx\cinderx\tests\run_test_suites.py`](C:\Users\z30059427\Desktop\cinderx\cinderx\tests\run_test_suites.py) 的 `parse_args()`：

- `-t, --target`
  - 可选：`all` / `pythonlib` / `runtime`
  - 默认：`all`
- `-o, --output`
  - 输出目录
  - 默认落在 `cov/ut/<timestamp>`
- `-f, --filter`
  - 可重复指定
  - `pythonlib` 按模块名过滤
  - `runtime` 按 gtest filter 过滤
- `-l, --list`
  - 只列测试，不执行
- `-c, --coverage`
  - 启用覆盖率收集
- `--python-exe`
  - 指定 Python 可执行文件
  - 默认：`python3`
- `--runtime-build-dir`
  - 覆盖 runtime build 目录
- `--runtime-binary`
  - 覆盖 RuntimeTests 二进制路径
- `--runtime-cwd`
  - 覆盖 RuntimeTests 工作目录
- `--keep-going`
  - 出错后继续跑
  - 当前默认启用
- `--no-build`
  - 跳过构建
  - 对 `runtime` 表示不重新 build RuntimeTests
  - 对 `pythonlib -c` 表示不重新安装 coverage 版临时前缀
- `--gcc-root`
  - GCC14 toolchain 根目录

### 1.3 依赖与环境要求

当前入口依赖：

- Python 3.14
- GCC14 toolchain
- `pip install .` 可用
- 远端/本地运行环境可访问：
  - `gcc`
  - `g++`
  - `gcov`

在 Linux/ARM 远端当前常用的是：

```bash
--gcc-root /opt/openEuler/gcc-toolset-14/root
```

### 1.4 `pythonlib` 普通模式与 `pythonlib -c` 覆盖率模式

这是当前最容易混淆的地方。

#### 普通 `pythonlib`

- 不再 build 当前仓库 native 产物
- 直接使用当前环境里已经安装好的 `cinderx` / `_cinderx`
- 入口会先执行安装检查：

```python
import cinderx, _cinderx
```

如果未安装，会报错并提示先执行：

```bash
python -m pip install .
```

#### `pythonlib -c`

- 仍然使用“安装版”口径
- 但不是覆盖默认安装，而是：
  1. 先用 coverage flags 做一次 `pip install .`
  2. 安装到临时前缀目录：
     - `<output>/pythonlib-install-prefix`
  3. 运行时临时把这个前缀的 `site-packages` 放到 `PYTHONPATH` 最前
  4. 跑完后回收 coverage
  5. 删除临时前缀目录

这样做的好处：
- 不覆盖默认环境里的普通安装版
- `pythonlib` 普通模式和 `pythonlib -c` 完全隔离

coverage build 基线目录：
- 普通安装 build base：`scratch`
- coverage 安装 build base：`scratch-pythonlib-cov`

对应逻辑见：
- [`C:\Users\z30059427\Desktop\cinderx\cinderx\setup.py`](C:\Users\z30059427\Desktop\cinderx\cinderx\setup.py)
- [`C:\Users\z30059427\Desktop\cinderx\cinderx\tests\run_test_suites.py`](C:\Users\z30059427\Desktop\cinderx\cinderx\tests\run_test_suites.py)

## 2. 当前 UT 的结构和分类

### 2.1 总体分类

当前统一 UT 分两大类：

- `pythonlib`
- `runtime`

对应目录：

- `pythonlib`
  - [`C:\Users\z30059427\Desktop\cinderx\cinderx\cinderx\PythonLib`](C:\Users\z30059427\Desktop\cinderx\cinderx\cinderx\PythonLib)
  - 主要包含：
    - `test_cinderx.*`
    - 通过 cinder runner 运行的一部分 stdlib `test.*`

- `runtime`
  - [`C:\Users\z30059427\Desktop\cinderx\cinderx\cinderx\RuntimeTests`](C:\Users\z30059427\Desktop\cinderx\cinderx\cinderx\RuntimeTests)
  - gtest 风格 C++ UT
  - 包括：
    - 普通 C++ runtime tests
    - HIR parser / optimizer 相关文本测试

### 2.2 `pythonlib` 是如何运行的

执行器：
- [`C:\Users\z30059427\Desktop\cinderx\cinderx\cinderx\TestScripts\cinder_test_runner312.py`](C:\Users\z30059427\Desktop\cinderx\cinderx\cinderx\TestScripts\cinder_test_runner312.py)

当前口径：
- 使用安装版 `cinderx`
- 通过 `run_test_suites.py` 调度模块执行
- 对少数特殊模块做模块级环境覆盖

当前已知模块级覆盖：
- `test_cinderx.test_jit_frame`
  - `PYTHONJITLIGHTWEIGHTFRAME=0`
- `test.test_code`
  - `CINDERX_DISABLE_SAVE_ENV_JIT_SUPPRESS=1`

对应逻辑：
- [`C:\Users\z30059427\Desktop\cinderx\cinderx\tests\run_test_suites.py`](C:\Users\z30059427\Desktop\cinderx\cinderx\tests\run_test_suites.py) 中的 `pythonlib_module_env()`

### 2.3 `runtime` 是如何构建和运行的

`runtime` 使用单独 build：

- 非覆盖率：
  - `build-runtime-tests-gcc14`
- 覆盖率：
  - `build-runtime-tests-gcc14-cov`

默认产物：
- 二进制：
  - `build-runtime-tests-gcc14*/cinderx/RuntimeTests/RuntimeTests`
- 工作目录：
  - `build-runtime-tests-gcc14*/cinderx/RuntimeTests/runtime_test_root`

当前 `runtime` 构建口径：
- 直接 CMake build
- `BUILD_RUNTIME_TESTS=ON`
- 覆盖率模式下同样走单独 cov build 目录

### 2.4 覆盖率统计口径

当前总表覆盖率是 native product coverage 口径。

#### `pythonlib -c`

- 用临时前缀安装 coverage 版
- `.gcno/.gcda` 从 `scratch-pythonlib-cov/...` 回收
- 不再是 `0%`

#### `runtime -c`

- 用 `build-runtime-tests-gcc14-cov`
- 直接从 cov build 回收

#### `combined`

- 总表覆盖率会把选中的 suite 结果汇总到产品源码根上：
  - `cinderx/Jit`
  - `cinderx/Interpreter`
  - `cinderx/Common`
  - 等产品源码目录

## 3. 如何新增 UT

### 3.1 新增 `pythonlib` UT

推荐位置：

- CinderX Python 层测试：
  - [`C:\Users\z30059427\Desktop\cinderx\cinderx\cinderx\PythonLib\test_cinderx`](C:\Users\z30059427\Desktop\cinderx\cinderx\cinderx\PythonLib\test_cinderx)

做法：
- 新建 `test_*.py`
- 按现有 `unittest` 风格编写
- 尽量避免：
  - 直接修改全局 `os.environ`
  - 手写整份 `dict(os.environ)` / `os.environ.copy()` 后到处传
- 如果必须起子进程，优先复用文件里已有 helper

新增后验证：

```bash
python tests/run_test_suites.py -t pythonlib -f test_cinderx.test_xxx --gcc-root /opt/openEuler/gcc-toolset-14/root
```

如果是覆盖率验证：

```bash
python tests/run_test_suites.py -t pythonlib -c -f test_cinderx.test_xxx --gcc-root /opt/openEuler/gcc-toolset-14/root
```

### 3.2 新增 `runtime` UT

推荐位置：

- C++/gtest：
  - [`C:\Users\z30059427\Desktop\cinderx\cinderx\cinderx\RuntimeTests`](C:\Users\z30059427\Desktop\cinderx\cinderx\cinderx\RuntimeTests)
- HIR 文本测试：
  - [`C:\Users\z30059427\Desktop\cinderx\cinderx\cinderx\RuntimeTests\hir_tests`](C:\Users\z30059427\Desktop\cinderx\cinderx\cinderx\RuntimeTests\hir_tests)

做法：
- 新增 gtest case 或扩展现有 `.txt` 测试数据
- 如果新测试依赖 Python runtime / `_cinderx` 模块状态，优先复用现有 fixture

新增后验证：

```bash
python tests/run_test_suites.py -t runtime -f SomeTest.SomeCase --gcc-root /opt/openEuler/gcc-toolset-14/root
```

### 3.3 新增测试时的实践建议

- 优先先单模块/单 case 跑通
- 行为变化要补测试，不要只改实现
- 对平台能力未到位的情况：
  - 先明确是“真实产品缺口”还是“测试断言过强”
- 不要把测试入口修复和产品行为大改混在一个提交里

## 4. 当前 UT 现状与后续关注点

### 4.1 当前整体现状

最近一轮全量对比报告：
- [`C:\Users\z30059427\Desktop\cinderx\cov\ut-final-diff-2.md`](C:\Users\z30059427\Desktop\cinderx\cov\ut-final-diff-2.md)

基线：
- [`C:\Users\z30059427\Desktop\cinderx\cov\ut-all-fbinc-4db05fc`](C:\Users\z30059427\Desktop\cinderx\cov\ut-all-fbinc-4db05fc)

当时结论：
- 总用例数：`1371 -> 1387`
- 新增用例：`16`
- 新增用例全部 `PASSED`
- 相对基线：
  - `pythonlib` 旧用例退化：`0`
  - `runtime` 旧用例退化：`0`
- 修复旧红项：
  - `pythonlib`: `4`
  - `runtime`: `592`

说明：
- 上述全量结果生成时，`pythonlib -c` 还处在“安装版 coverage 口径刚打通前”的阶段，因此当时文档里的 `pythonlib coverage = 0%`
- 现在 `pythonlib -c` 已经通过最新逻辑打通，后续如果重新跑全量 `-c`，`pythonlib coverage` 应该不再是 `0%`

### 4.2 当前已知需要持续关注的点

#### 1. `test.test_getpath`

状态：
- 在最近全量结果里仍是 `FAILED`

说明：
- 它在基线里也是 `FAILED`
- 不属于这轮新引入退化
- 但如果后面要继续提升 `pythonlib` 总体绿率，这是仍需单独跟进的一项

#### 2. `test_cinderx.test_jit_frame`

现状：
- 目前通过模块级环境覆盖恢复：
  - `PYTHONJITLIGHTWEIGHTFRAME=0`

说明：
- 当前 lightweight frame 模式下，`sys._getframe()/f_back` 相关路径在 OSS 3.14/ARM 口径下仍有产品缺口
- 当前是测试入口侧最小收口，不等于产品根因已彻底解决

#### 3. `test.test_code`

现状：
- 当前通过模块级覆盖恢复：
  - `CINDERX_DISABLE_SAVE_ENV_JIT_SUPPRESS=1`

说明：
- 当前问题与 `libregrtest.save_env` 的 JIT suppress patch 交互有关
- 已经不再阻塞 UT
- 但从产品/runner 语义上，后续仍值得继续收敛

#### 4. `test_cinderx.test_oss_quick`

现状：
- 已修通

说明：
- 这条线暴露过 compat policy 与实际 feature enablement 的不一致
- 后续如果再改：
  - `lightweight_frames`
  - `adaptive_static_python`
  - `static_python`
 之间的策略关系，需要优先回归这个模块

#### 5. `test_cinderx.test_arm_runtime`

现状：
- 已修通

说明：
- 这套测试在 ARM 上对优化形态很敏感
- 当前已经收掉一批“断言过强”的 case，并对 3 个 double-lowering 能力点做了 skip
- 后续如果补齐 ARM JIT 的：
  - `DoubleBinaryOp`
  - `DoubleSqrt`
 相关 lowering，可以优先回看这套测试，把 skip 收回来

### 4.3 当前最推荐的日常回归组合

开发中建议先跑：

```bash
# 脚本参数/入口回归
python -m unittest tests.test_run_test_suites -v

# pythonlib 代表模块
python tests/run_test_suites.py -t pythonlib -f test.test_call --gcc-root /opt/openEuler/gcc-toolset-14/root
python tests/run_test_suites.py -t pythonlib -f test_cinderx.test_oss_quick --gcc-root /opt/openEuler/gcc-toolset-14/root
python tests/run_test_suites.py -t pythonlib -f test_cinderx.test_arm_runtime --gcc-root /opt/openEuler/gcc-toolset-14/root

# runtime 代表模块
python tests/run_test_suites.py -t runtime -f CmdLineTest.BasicFlags --gcc-root /opt/openEuler/gcc-toolset-14/root
```

做覆盖率 smoke 时建议：

```bash
python tests/run_test_suites.py -t pythonlib -c -f test.test_call --gcc-root /opt/openEuler/gcc-toolset-14/root
python tests/run_test_suites.py -t runtime -c -f CmdLineTest.BasicFlags --gcc-root /opt/openEuler/gcc-toolset-14/root
```

## 5. 总结

当前统一 UT 入口已经稳定支持：

- `pythonlib`
  - 安装版运行
  - 安装版 coverage
  - 临时前缀隔离
- `runtime`
  - 独立构建
  - 独立 coverage
- 全量结果统一汇总
- 相对基线做新增/退化/修复统计

当前最重要的使用约束是：

- 普通 `pythonlib` 先确保执行过：
  ```bash
  python -m pip install .
  ```
- `pythonlib -c` 不再覆盖默认环境，而是临时安装 coverage 版
- `runtime` 仍然依赖 GCC14 toolchain 和单独 build 目录

