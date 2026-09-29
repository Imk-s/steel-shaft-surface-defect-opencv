# Steel Shaft Surface Defect Detection

基于 OpenCV 的钢轴表面划痕与凹点检测项目。

## 检测目标

- 检测宽度 ≥ 0.5 mm 的划痕
- 检测直径 ≥ 0.5 mm 的凹点

## 算法流程

当前版本采用传统计算机视觉方法完成钢轴表面划痕与凹点检测，整体流程如下：

```text
原始图像
   ↓
ROI 提取
   ↓
灰度化与 Gaussian 滤波
   ↓
双尺度背景差分增强
   ↓
固定阈值 / 局部自适应阈值
   ↓
形态学处理
   ↓
轮廓提取
   ↓
候选区域特征计算
   ├─ 面积
   ├─ 长宽比
   ├─ 圆度
   ├─ 骨架宽度
   └─ 轮廓长度
   ↓
Scratch / Pit 分类
   ↓
像素尺寸测量
   ↓
px → mm 标定
   ↓
0.5 mm 阈值筛选
   ↓
结果标注与统计输出
```

## 桌面界面（阶段 1～10）

PySide6 界面通过独立入口启动，已接入原有检测和尺寸测量流程。
图片导入、等比例预览和鼠标 ROI 继续使用阶段 1～4 的实现。

### 运行

在项目目录的 PowerShell 中执行：

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gui.txt
.\.venv\Scripts\python.exe -m gui
```

如果已经创建并安装好了 `.venv`，只执行最后一行。
原命令行检测入口仍为 `python main.py`。

### 如何检测

1. 导入图片，左侧显示原图预览和原始分辨率；文件选择起始目录复用
   `config.IMAGE_PATH.parent`，支持中文路径。
2. 点击“选择 ROI”，在左侧拖拽。ROI 的左右边缘应贴紧钢轴两侧；
   下方显示原图中的 `(x, y, w, h)`。再次点击可重新选择，Esc 可取消。
3. 调整窗口大小，图片保持比例，ROI 保持原图坐标并正确重绘。
4. 输入钢轴真实直径，例如 `30`，单位 mm。点击“开始生成”后，后台
   执行原有预处理、分类、标定和尺寸筛选；按钮显示“处理中……”。
5. 成功后右侧显示最近一次 Result。“查看中间结果”和“查看 Debug 图”
   打开两列、可滚动的动态图库；“查看后台日志”显示本轮完整日志。
   各页面都可用“← 返回”回到主页面。
6. 运行期间仍可缩放窗口、查看上一轮图库和日志。重复生成和修改输入
   暂时禁用；若此时关闭窗口，会等待本轮任务结束后安全关闭。

没有图片、没有 ROI、直径为空/非数字/非正数时，界面会提示检查输入，
不会启动任务。导入失败或生成失败也不会清空最近一次成功结果。

### 图片与日志保存

GUI 每次运行在 `config.DEBUG_OUTPUT_ROOT` 下创建独立的 `run_*` 目录，
默认位于项目的 `debug/_picture`，不覆盖之前的运行结果。目录中包含：

- `result.jpg`、`mask.jpg` 和 `shaft_mask.jpg`。
- 按顺序编号的中间图 PNG：Gray、Small Scale、Background、Enhanced、
  Binary Mask、Clean Mask、Shaft Mask。
- 原有轮廓 Debug 图；是否生成仍由 `config.DEBUG_ALL_CONTOURS` 控制。
- `pipeline.log`：UTF-8 日志，包含标定值、轮廓信息、测量结果和统计。

GUI 不调用 `cv2.selectROI`、`cv2.imshow` 或 `cv2.waitKey`。
原命令行入口及其交互窗口、结果保存位置和 Debug 功能继续保留。

### 文件职责与状态

- `gui/__main__.py`：桌面入口。
- `gui/main_window.py`：保留原布局、导入和输入验证；新增任务管理、
  完整输出提交、图库页面和日志页面。
- `gui/image_view.py`：原始 ndarray 与 Qt 预览分离；灰度/BGR/BGRA
  图片转换到 Qt 自有存储，QPixmap 使用 `KeepAspectRatio` 缩放并居中。
- `gui/state.py`：`InputState` 保存当前图片、`Path`、原图 ROI 和直径；
  `OutputState` 复用 `pipeline.PipelineResult`，保存最近成功运行的
  Result、中间图字典、Debug 字典、日志和本轮保存目录。
- `pipeline.py`：不依赖 Qt 的高层接口 `run_pipeline(image, roi, diameter_mm)`，
  只编排已有算法函数；打印同时输出到终端并捕获为本轮日志。
- `gui/worker.py`：`QObject` Worker 在 `QThread` 中调用 pipeline，通过
  signal/slot 把成功结果或错误交回主线程，不直接操作任何 QWidget。
- `gui/image_gallery.py`：中间图和 Debug 图复用的两列动态图库。
- `debug_utils.py`：仅新增可选保存目录和内存模式；默认调用方式保留，
  轮廓特征、拒绝原因和绘制逻辑没有重写。

导入图片时只更新 `InputState` 并清空旧 ROI；修改直径和 ROI 也只更新
输入。`apply_output_state()` 是完整成功输出的提交入口；先准备所有新
预览和图库，再统一替换。运行失败或预览转换失败时，保留上一次成功
Result、中间图、Debug 图和日志。失败弹窗可展开详细错误信息。

鼠标使用控件的逻辑坐标，映射方式为：

```text
image_x = (mouse_x - offset_x) / display_scale
image_y = (mouse_y - offset_y) / display_scale
```

拖拽端点转换后限制在原图边界，较小端点向下取整、较大端点向上取整。
覆盖层绘制时用相反转换，控件 resize 时重新计算比例和居中偏移。
缩放和绘制均不改变原始 ndarray。

算法始终接收原分辨率图片和原图 ROI；点击生成时复制输入快照交给
后台线程，GUI 缩放图不会参与检测或测量。
`main.py`、`preprocess.py`、`detector.py`、`measurement.py`、`config.py`
和原有 `gui/image_view.py` 本次均未修改。
当前代码参数以 `config.py` 为准（`MIN_DEFECT_SIZE_MM` 当前为 `0.25`），
本次界面开发没有调整参数或文档中原有的目标标准。

### 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test*.py' -v
```

测试使用 Qt `offscreen`，不会弹出本机窗口。覆盖 1920×1080、4000×3000、
竖图、缩放比例、真实鼠标拖拽、反向/越界 ROI、Esc、中文路径、A/B 输入
切换、成功输出整批提交、失败保留和非法参数提示。也覆盖真实 pipeline
的 Worker 调用、线程归属、界面心跳、图库数量变化、中文日志、独立保存
目录及错误预览回退。共 28 项测试。

原命令行完整流程通过屏蔽交互窗口和写盘进行回归，并与 pipeline 的
Result/Mask/Shaft Mask 逐像素比较，原有结果文件不会被测试覆盖。
另已用项目中的 4096×2304 实图完成一次后台检测和四个页面的离屏检查。
