# Development Log

##2026-09-23

### Baseline

完成钢轴表面划痕与凹点的第一版传统 CV 检测流程。

### 算法

当前流程：

- 灰度化
- 3x3 Gaussian Blur 降噪
- 大尺度 Gaussian Blur 估计背景
- background - gray 提取暗异常
- 固定阈值二值化
- Opening 去除小噪声
- Closing 连接断裂区域
- findContours 提取候选
- 根据几何特征区分 Scratch / Pit

### 改进

早期采用 normalize + Otsu 时，
干净样品中的弱表面纹理会被相对放大，产生较多误检。

改为保留原始背景差分值，并使用固定 threshold 后，
干净区域误检明显下降。

### 后序目标

- 双尺度缺陷增强
- 解决较大暗缺陷漏检
- 像素-mm 标定
- 根据 0.5 mm 标准过滤缺陷