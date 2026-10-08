# Steel Shaft Surface Defect Detection

基于 OpenCV 的钢轴表面划痕、凹点与片状锈斑/污渍检测项目。

## 检测目标

- 检测宽度 ≥ 0.5 mm 的划痕
- 检测直径 ≥ 0.5 mm 的凹点
- 检测片状锈斑/污渍；其跨度也按当前 `config.MIN_DEFECT_SIZE_MM` 筛选

## 算法流程

当前版本采用传统计算机视觉方法。划痕/凹点与片状污渍走两条候选通道：

```text
原始图像
   ↓
ROI 提取
   ↓
灰度化与 Gaussian 滤波
   ↓
双尺度背景差分增强 ───────────── Lab 局部色差 + 纹理异常
   ↓
局部自适应阈值                       污渍候选 Mask
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
Scratch / Pit 分类                   Stain 区域筛选
   ↓
像素尺寸测量
   ↓
px → mm 标定
   ↓
按 config.MIN_DEFECT_SIZE_MM 筛选
   ↓
结果标注与统计输出
```

### 采集图中的边缘和粘连防误判

ROI 内另设有效检测区：上下各排除 4%，左右只保留钢轴中心 `±0.6R`。
被有效区截断的轮廓不作最终检测，Debug 日志保留其编号和拒绝原因。
Binary → Clean 先按尺度轻度去噪、删除小连通域，仅对剩余细长候选
做短距离水平/竖直连接，不再对整张 Mask 统一闭运算。

Scratch 的候选形状仍用骨架中位宽和外接矩形筛选；最终宽度改为
`2 × 最宽骨架点到边缘的距离`，再沿局部法线映射到圆柱表面。
不再沿法线穿过粘连纹理追边界，避免测量宽度远超骨架直径。
标准恢复为 `config.MIN_DEFECT_SIZE_MM = 0.5 mm`。标定后先得
`px_per_mm = D_px / D_mm`，滤波尺度、面积、长度、去噪和连接核尺寸
均由毫米参数换算成像素；最终是否达标仍以圆柱表面毫米测量为准。
当前分割仍可能把宽痕切碎，0.5 mm 下某些照片可能没有达标划痕；
这不等于照片没有肉眼可见的异常。
Debug 的 Clean Contours 按最终结果着色：达标划痕为红色、达标凹点为绿色，
边界、形状、不可测或尺寸不足的候选为黄色；日志保留轮廓编号及拒绝原因。

### 片状锈斑/污渍

现有 Scratch/Pit 的几何条件分别偏向细长线条和近圆形小坑，无法把不规则的
片状锈斑当作一个完整缺陷。新增的 Stain 通道不改变原 Mask 和测量公式：
在 ROI 内对比 Lab 色度与较大尺度的局部背景，并结合原增强图的局部纹理；
对无明显色差的深色污渍，再加入更严格的局部变暗、面积和形状条件，
形成独立 Stain Mask。排除 ROI 顶部反光、底缘和侧边附近的截断候选后，
再用已有圆柱表面映射及最小包围圆计算区域跨度。Result 用橙色轮廓标示，
标签 `S编号` 对应后台日志；Stain Debug 图中青色为入选、黄色为拒绝。
这是一种针对颜色/纹理异常的传统视觉启发式，不能保证不同照明、材质及
低对比、平滑的污渍都被检出；需要更多标注样本来验证和调整。

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
  Binary Mask、Clean Mask、Stain Mask、Detection Area、Shaft Mask。
- 原有轮廓 Debug 图及独立 Stain Contours 图；是否生成仍由
  `config.DEBUG_ALL_CONTOURS` 控制。
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
  编排划痕、凹点和污渍通道；打印同时输出到终端并捕获为本轮日志。
- `preprocess.py` / `detector.py`：分别生成污渍候选 Mask 和筛选片状轮廓，
  原 Scratch/Pit 预处理与判定条件保持独立。
- `gui/worker.py`：`QObject` Worker 在 `QThread` 中调用 pipeline，通过
  signal/slot 把成功结果或错误交回主线程，不直接操作任何 QWidget。
- `gui/image_gallery.py`：中间图和 Debug 图复用的两列动态图库。
- `debug_utils.py`：保留原轮廓编号和拒绝原因，并新增独立的 S 编号污渍图。

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
GUI 只调用高层 pipeline；预处理、轮廓分类和尺寸测量仍在各自的算法模块中，
原有 `gui/image_view.py` 的缩放与 ROI 坐标映射没有改动。
当前尺寸判定门槛以 `config.MIN_DEFECT_SIZE_MM` 为准，现为 `0.5 mm`，
低于门槛的候选仍会在日志中
留下测量值，但不会绘制到 Result。

### 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test*.py' -v
```

测试使用 Qt `offscreen`，不会弹出本机窗口。覆盖 1920×1080、4000×3000、
竖图、缩放比例、真实鼠标拖拽、反向/越界 ROI、Esc、中文路径、A/B 输入
切换、成功输出整批提交、失败保留和非法参数提示。也覆盖真实 pipeline
的 Worker 调用、线程归属、界面心跳、图库数量变化、中文日志、独立保存
目录及错误预览回退，也检查污渍通道和原命令行结果一致性。

原命令行完整流程通过屏蔽交互窗口和写盘进行回归，并与 pipeline 的
Result/Mask/Shaft Mask 逐像素比较，原有结果文件不会被测试覆盖。
另已用项目中的 4096×2304 实图完成一次后台检测和四个页面的离屏检查。
