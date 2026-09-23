# Steel Shaft Surface Defect Detection

基于 OpenCV 的钢轴表面划痕与凹点检测项目。

## 检测目标

- 检测宽度 ≥ 0.5 mm 的划痕
- 检测直径 ≥ 0.5 mm 的凹点

## 当前算法流程

原图
→ ROI
→ 灰度化
→ Gaussian Blur 降噪
→ Gaussian Blur 背景估计
→ background - gray
→ 固定阈值二值化
→ 形态学处理
→ 轮廓提取
→ Scratch / Pit 初步分类

## 当前进度

- [x] 基础图像读取与 ROI
- [x] 灰度化与高斯滤波
- [x] 背景差分增强
- [x] 固定阈值二值化
- [x] 划痕与凹点候选检测
- [ ] 双尺度增强
- [ ] 像素-mm 标定
- [ ] 0.5 mm 阈值筛选
- [ ] 专业照明环境测试
- [ ] GUI