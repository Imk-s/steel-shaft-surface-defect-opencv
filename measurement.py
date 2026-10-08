"""由钢轴主体 Mask 计算像素尺度与中心轴线。"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import (
    IMAGE_PATH,
    MEASURABLE_RADIUS_RATIO,
    MIN_DEFECT_SIZE_MM,
    PROJECT_ROOT,
    SCRATCH_PCA_RADIUS_MM,
)


DEFAULT_IMAGE_PATH = IMAGE_PATH
DEFAULT_SHAFT_MASK_PATH = PROJECT_ROOT / "shaft_mask.jpg"


@dataclass(frozen=True)
class ShaftCalibration:
    """钢轴标定结果，坐标均相对于输入 Mask。"""

    diameter_mm: float
    diameter_px: float
    center_x: float
    radius_mm: float
    radius_px: float
    mm_per_px: float
    bounding_box: tuple[int, int, int, int]
    measurement_y_range: tuple[int, int]
    sampled_rows: int
    valid_rows: int


@dataclass(frozen=True)
class SurfacePoint:
    """圆柱展开坐标：s 为圆周弧长，z 为轴向距离。"""

    s_mm: float
    z_mm: float


@dataclass(frozen=True)
class ScratchMeasurement:
    """划痕宽度测量结果。"""

    measurable: bool
    valid: bool
    width_mm: float | None
    width_px: float | None
    max_radius_px: float | None
    max_point_xy: tuple[float, float] | None
    boundary_a_xy: tuple[float, float] | None
    boundary_b_xy: tuple[float, float] | None
    tangent_xy: tuple[float, float] | None
    normal_xy: tuple[float, float] | None
    reason: str | None


@dataclass(frozen=True)
class PitMeasurement:
    """凹点展开轮廓最小外接圆测量结果。"""

    measurable: bool
    valid: bool
    diameter_mm: float | None
    center_sz_mm: tuple[float, float] | None
    radius_mm: float | None
    reason: str | None


def _as_binary_mask(mask):
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("钢轴主体 Mask 必须是单通道二维图像")

    return (mask > 0).astype(np.uint8)


def prepare_shaft_mask(mask):
    """保留并填充 Mask 中最大的外轮廓，作为钢轴主体。"""
    binary = _as_binary_mask(mask)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise ValueError("钢轴主体 Mask 中没有前景")

    largest_contour = max(contours, key=cv2.contourArea)
    body_mask = np.zeros_like(binary)
    cv2.drawContours(body_mask, [largest_contour], -1, 255, cv2.FILLED)
    return body_mask


def create_shaft_mask_from_roi(image, roi):
    """将用户精确框选的钢轴 ROI 转为主体 Mask。

    v1 使用人工贴合钢轴左右边缘的 ROI，避免反光金属与桌面颜色相近
    时自动分割错误。返回值是 ROI 局部坐标下的全前景 Mask。
    """
    x, y, width, height = map(int, roi)

    if width <= 0 or height <= 0:
        raise ValueError("钢轴 ROI 的宽高必须大于 0")

    image_height, image_width = image.shape[:2]

    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError("钢轴 ROI 超出图像范围")

    return np.full((height, width), 255, dtype=np.uint8)


def _robust_inliers(values, mad_z_threshold):
    """使用 MAD 过滤异常值；MAD 为零时保留中位数附近 1 像素。"""
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    deviation = np.abs(values - median)
    mad = np.median(deviation)

    if mad <= np.finfo(np.float64).eps:
        return deviation <= 1.0

    robust_z = 0.6745 * deviation / mad
    return robust_z <= mad_z_threshold


def calibrate_shaft_mask(
    shaft_mask,
    diameter_mm,
    middle_height_ratio=0.60,
    mad_z_threshold=3.5,
):
    """根据钢轴主体 Mask 标定 D_px、x_c、R_px、R_mm 与 k。

    处理流程：最大连通域 → bounding box 高度中间 60% → 逐行左右
    边界 → MAD 异常值过滤 → 中位数。像素直径按前景覆盖像素数
    ``x_right - x_left + 1`` 计算。
    """
    if diameter_mm <= 0:
        raise ValueError("钢轴真实直径 diameter_mm 必须大于 0")
    if not 0 < middle_height_ratio <= 1:
        raise ValueError("middle_height_ratio 必须在 (0, 1] 内")
    if mad_z_threshold <= 0:
        raise ValueError("mad_z_threshold 必须大于 0")

    body_mask = prepare_shaft_mask(shaft_mask)
    foreground_y, foreground_x = np.nonzero(body_mask)
    x0 = int(foreground_x.min())
    y0 = int(foreground_y.min())
    width = int(foreground_x.max() - x0 + 1)
    height = int(foreground_y.max() - y0 + 1)

    margin_ratio = (1.0 - middle_height_ratio) / 2.0
    y_start = y0 + int(np.floor(height * margin_ratio))
    y_end = y0 + int(np.ceil(height * (1.0 - margin_ratio)))

    diameters = []
    centers = []

    for row_y in range(y_start, y_end):
        foreground_columns = np.flatnonzero(body_mask[row_y])

        if foreground_columns.size < 2:
            continue

        x_left = float(foreground_columns[0])
        x_right = float(foreground_columns[-1])
        diameters.append(x_right - x_left + 1.0)
        centers.append((x_left + x_right) / 2.0)

    if not diameters:
        raise ValueError("钢轴 bounding box 中间区域没有有效测量行")

    diameters = np.asarray(diameters, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    diameter_inliers = _robust_inliers(diameters, mad_z_threshold)
    center_inliers = _robust_inliers(centers, mad_z_threshold)
    valid = diameter_inliers & center_inliers

    if np.count_nonzero(valid) == 0:
        raise ValueError("异常值过滤后没有有效测量行")

    diameter_px = float(np.median(diameters[valid]))
    center_x = float(np.median(centers[valid]))
    radius_px = diameter_px / 2.0
    radius_mm = float(diameter_mm) / 2.0
    mm_per_px = float(diameter_mm) / diameter_px

    return ShaftCalibration(
        diameter_mm=float(diameter_mm),
        diameter_px=diameter_px,
        center_x=center_x,
        radius_mm=radius_mm,
        radius_px=radius_px,
        mm_per_px=mm_per_px,
        bounding_box=(x0, y0, width, height),
        measurement_y_range=(y_start, y_end),
        sampled_rows=int(diameters.size),
        valid_rows=int(np.count_nonzero(valid)),
    )


def is_x_measurable(
    x_px,
    calibration,
    radius_ratio=MEASURABLE_RADIUS_RATIO,
):
    """判断横坐标是否位于圆柱投影中央有效测量区域。"""
    if not 0 < radius_ratio <= 1:
        raise ValueError("radius_ratio 必须在 (0, 1] 内")

    return abs(float(x_px) - calibration.center_x) <= (
        radius_ratio * calibration.radius_px
    )


def map_x_to_arc(
    x_px,
    calibration,
    require_measurable=True,
    radius_ratio=MEASURABLE_RADIUS_RATIO,
):
    """将图像横坐标 x 映射为圆柱表面的圆周弧长坐标 s。"""
    x_px = float(x_px)
    normalized_x = (x_px - calibration.center_x) / calibration.radius_px

    if abs(normalized_x) > 1.0 + 1e-9:
        raise ValueError(
            f"x={x_px} 位于钢轴投影外，"
            f"有效范围为 "
            f"[{calibration.center_x - calibration.radius_px}, "
            f"{calibration.center_x + calibration.radius_px}]"
        )
    if require_measurable and not is_x_measurable(
        x_px,
        calibration,
        radius_ratio,
    ):
        raise ValueError(
            f"x={x_px} 超出中央有效测量区域："
            f"|x-x_c| <= {radius_ratio}R_px"
        )

    normalized_x = float(np.clip(normalized_x, -1.0, 1.0))
    return calibration.radius_mm * float(np.arcsin(normalized_x))


def map_y_to_axial(y_px, calibration):
    """将图像纵坐标 y 映射为轴向坐标 z。"""
    return calibration.mm_per_px * float(y_px)


def map_image_point(
    x_px,
    y_px,
    calibration,
    require_measurable=True,
    radius_ratio=MEASURABLE_RADIUS_RATIO,
):
    """将 ROI/Mask 坐标系中的图像点 (x, y) 映射为 (s, z)。"""
    return SurfacePoint(
        s_mm=map_x_to_arc(
            x_px,
            calibration,
            require_measurable=require_measurable,
            radius_ratio=radius_ratio,
        ),
        z_mm=map_y_to_axial(y_px, calibration),
    )


def map_image_points(
    points_xy,
    calibration,
    require_measurable=True,
    radius_ratio=MEASURABLE_RADIUS_RATIO,
):
    """批量映射 N×2 图像点，返回 N×2 的 (s_mm, z_mm) 数组。"""
    points = np.asarray(points_xy, dtype=np.float64)

    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy 必须是形状为 (N, 2) 的坐标数组")
    if not 0 < radius_ratio <= 1:
        raise ValueError("radius_ratio 必须在 (0, 1] 内")

    normalized_x = (
        points[:, 0] - calibration.center_x
    ) / calibration.radius_px

    outside_projection = np.abs(normalized_x) > 1.0 + 1e-9
    if np.any(outside_projection):
        indexes = np.flatnonzero(outside_projection).tolist()
        raise ValueError(f"以下点位于钢轴投影外: indexes={indexes}")

    if require_measurable:
        outside_measurement = np.abs(normalized_x) > radius_ratio
        if np.any(outside_measurement):
            indexes = np.flatnonzero(outside_measurement).tolist()
            raise ValueError(
                "以下点超出中央有效测量区域: "
                f"indexes={indexes}, limit={radius_ratio}R_px"
            )

    normalized_x = np.clip(normalized_x, -1.0, 1.0)
    surface_points = np.empty_like(points, dtype=np.float64)
    surface_points[:, 0] = (
        calibration.radius_mm * np.arcsin(normalized_x)
    )
    surface_points[:, 1] = calibration.mm_per_px * points[:, 1]
    return surface_points


def _extract_skeleton(binary_mask):
    """使用 Zhang-Suen 细化得到单像素骨架。"""
    work = np.pad(
        (np.asarray(binary_mask) > 0).astype(np.uint8),
        1,
        mode="constant",
        constant_values=0,
    )

    while True:
        changed = False

        for step in (0, 1):
            center = work[1:-1, 1:-1]
            p2 = work[:-2, 1:-1]
            p3 = work[:-2, 2:]
            p4 = work[1:-1, 2:]
            p5 = work[2:, 2:]
            p6 = work[2:, 1:-1]
            p7 = work[2:, :-2]
            p8 = work[1:-1, :-2]
            p9 = work[:-2, :-2]
            neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )

            if step == 0:
                preserve_1 = p2 * p4 * p6 == 0
                preserve_2 = p4 * p6 * p8 == 0
            else:
                preserve_1 = p2 * p4 * p8 == 0
                preserve_2 = p2 * p6 * p8 == 0

            remove = (
                (center == 1)
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
                & preserve_1
                & preserve_2
            )

            if np.any(remove):
                center[remove] = 0
                changed = True

        if not changed:
            break

    return (work[1:-1, 1:-1] * 255).astype(np.uint8)


def _contour_mask(contour, padding=3):
    bx, by, width, height = cv2.boundingRect(contour)
    offset_x = bx - padding
    offset_y = by - padding
    local_contour = contour.copy()
    local_contour[:, 0, 0] -= offset_x
    local_contour[:, 0, 1] -= offset_y
    mask = np.zeros(
        (height + 2 * padding, width + 2 * padding),
        dtype=np.uint8,
    )
    cv2.drawContours(mask, [local_contour], -1, 255, cv2.FILLED)
    return mask, offset_x, offset_y


def _maximum_distance_skeleton_point(skeleton, distance_map):
    points_yx = np.argwhere(skeleton > 0)

    if points_yx.size == 0:
        raise ValueError("划痕 Mask 无法提取骨架")

    radii = distance_map[points_yx[:, 0], points_yx[:, 1]]
    max_radius = float(np.max(radii))
    candidates = points_yx[np.isclose(radii, max_radius)]
    skeleton_center = np.mean(points_yx, axis=0)
    candidate_index = int(np.argmin(
        np.sum((candidates - skeleton_center) ** 2, axis=1)
    ))
    point_y, point_x = candidates[candidate_index]
    return np.array([float(point_x), float(point_y)]), max_radius


def _estimate_skeleton_direction(
    skeleton,
    point_xy,
    neighborhood_radius,
):
    points_yx = np.argwhere(skeleton > 0)
    points_xy = points_yx[:, ::-1].astype(np.float64)
    distances = np.linalg.norm(points_xy - point_xy, axis=1)
    local_points = points_xy[distances <= neighborhood_radius]

    if local_points.shape[0] < 3:
        nearest_count = min(7, points_xy.shape[0])
        local_points = points_xy[np.argsort(distances)[:nearest_count]]
    if local_points.shape[0] < 2:
        raise ValueError("骨架点过少，无法估计局部方向")

    centered = local_points - np.mean(local_points, axis=0)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    return tangent, normal


def measure_scratch_contour(
    contour,
    calibration,
    min_width_mm=MIN_DEFECT_SIZE_MM,
):
    """按骨架最大点、局部法线和圆柱映射测量划痕真实宽度。"""
    mask, offset_x, offset_y = _contour_mask(contour)
    skeleton = _extract_skeleton(mask)
    padded_mask = np.pad(mask, 1, mode="constant", constant_values=0)
    distance_map = cv2.distanceTransform(
        padded_mask,
        cv2.DIST_L2,
        5,
    )[1:-1, 1:-1]

    try:
        max_point_local, max_radius = _maximum_distance_skeleton_point(
            skeleton,
            distance_map,
        )
        tangent, normal = _estimate_skeleton_direction(
            skeleton,
            max_point_local,
            neighborhood_radius=SCRATCH_PCA_RADIUS_MM / calibration.mm_per_px,
        )
    except ValueError as error:
        return ScratchMeasurement(
            False, False, None, None, None, None,
            None, None, None, None, str(error),
        )

    offset = np.array([offset_x, offset_y], dtype=np.float64)
    max_point = max_point_local + offset
    # 最宽骨架点的内切圆半径就是中心线到边缘的距离。沿局部法线
    # 取 2*r 的宽度，不再让追边界误穿过分叉/粘连纹理。
    boundary_a = max_point - max_radius * normal
    boundary_b = max_point + max_radius * normal

    if not is_x_measurable(max_point[0], calibration):
        return ScratchMeasurement(
            measurable=False,
            valid=False,
            width_mm=None,
            width_px=float(np.linalg.norm(boundary_b - boundary_a)),
            max_radius_px=max_radius,
            max_point_xy=tuple(max_point),
            boundary_a_xy=tuple(boundary_a),
            boundary_b_xy=tuple(boundary_b),
            tangent_xy=tuple(tangent),
            normal_xy=tuple(normal),
            reason="划痕最大宽度点超出中央有效测量区域",
        )

    try:
        surface_a = map_image_point(
            boundary_a[0],
            boundary_a[1],
            calibration,
            require_measurable=False,
        )
        surface_b = map_image_point(
            boundary_b[0],
            boundary_b[1],
            calibration,
            require_measurable=False,
        )
    except ValueError as error:
        return ScratchMeasurement(
            False, False, None, None, max_radius,
            tuple(max_point), tuple(boundary_a), tuple(boundary_b),
            tuple(tangent), tuple(normal), str(error),
        )

    width_mm = float(np.hypot(
        surface_b.s_mm - surface_a.s_mm,
        surface_b.z_mm - surface_a.z_mm,
    ))
    width_px = float(np.linalg.norm(boundary_b - boundary_a))
    return ScratchMeasurement(
        measurable=True,
        valid=width_mm >= min_width_mm,
        width_mm=width_mm,
        width_px=width_px,
        max_radius_px=max_radius,
        max_point_xy=tuple(max_point),
        boundary_a_xy=tuple(boundary_a),
        boundary_b_xy=tuple(boundary_b),
        tangent_xy=tuple(tangent),
        normal_xy=tuple(normal),
        reason=None,
    )


def measure_pit_contour(
    contour,
    calibration,
    min_diameter_mm=MIN_DEFECT_SIZE_MM,
):
    """将凹点轮廓映射到 (s,z) 平面并计算最小外接圆直径。"""
    points_xy = contour.reshape(-1, 2).astype(np.float64)

    if points_xy.shape[0] == 0:
        return PitMeasurement(False, False, None, None, None, "凹点轮廓为空")

    (center_x, _), _ = cv2.minEnclosingCircle(contour)
    center_x = float(center_x)
    if not is_x_measurable(center_x, calibration):
        return PitMeasurement(
            False,
            False,
            None,
            None,
            None,
            "凹点中心超出中央有效测量区域",
        )

    try:
        surface_points = map_image_points(
            points_xy,
            calibration,
            require_measurable=False,
        )
    except ValueError as error:
        return PitMeasurement(False, False, None, None, None, str(error))

    (center_s, center_z), radius_mm = cv2.minEnclosingCircle(
        surface_points.astype(np.float32).reshape(-1, 1, 2)
    )
    diameter_mm = float(2.0 * radius_mm)
    return PitMeasurement(
        measurable=True,
        valid=diameter_mm >= min_diameter_mm,
        diameter_mm=diameter_mm,
        center_sz_mm=(float(center_s), float(center_z)),
        radius_mm=float(radius_mm),
        reason=None,
    )


def print_calibration(calibration):
    """将标定结果打印到终端。"""
    x0, y0, width, height = calibration.bounding_box
    y_start, y_end = calibration.measurement_y_range

    print("=== 钢轴标定结果 ===")
    print(f"主体 bounding box: x0={x0}, y0={y0}, w={width}, h={height}")
    print(f"中间测量行: y=[{y_start}, {y_end})")
    print(f"有效测量行: {calibration.valid_rows}/{calibration.sampled_rows}")
    print(f"D_mm = {calibration.diameter_mm:.6f} mm")
    print(f"D_px = {calibration.diameter_px:.6f} px")
    print(f"x_c  = {calibration.center_x:.6f} px")
    print(f"R_px = {calibration.radius_px:.6f} px")
    print(f"R_mm = {calibration.radius_mm:.6f} mm")
    print(f"k    = {calibration.mm_per_px:.9f} mm/px")


def read_mask(path):
    """读取 Mask，兼容 Windows 中文路径。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def read_image(path):
    """读取彩色图片，兼容 Windows 中文路径。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, image):
    """保存图片，兼容 Windows 中文路径。"""
    success, encoded = cv2.imencode(path.suffix, image)

    if not success:
        raise RuntimeError(f"图片编码失败: {path}")

    encoded.tofile(str(path))


def prompt_diameter_mm():
    """在终端中读取钢轴真实直径，直到输入有效正数。"""
    while True:
        raw_value = input("请输入钢轴真实直径 D_mm（单位 mm）: ").strip()

        try:
            diameter_mm = float(raw_value)
        except ValueError:
            print("输入无效，请输入数字，例如 50 或 50.0")
            continue

        if diameter_mm <= 0:
            print("真实直径必须大于 0，请重新输入")
            continue

        return diameter_mm


def main():
    parser = argparse.ArgumentParser(description="从钢轴照片或主体 Mask 计算尺度")
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help=f"钢轴原始照片，默认 {DEFAULT_IMAGE_PATH}",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        help="已有的钢轴主体 Mask；指定后跳过图片读取和 ROI 选择",
    )
    parser.add_argument(
        "--diameter-mm",
        type=float,
        help="钢轴真实直径，单位 mm；省略时在终端中输入",
    )
    args = parser.parse_args()

    roi_x = 0

    if args.mask is not None:
        if not args.mask.is_file():
            raise SystemExit(f"钢轴主体 Mask 不存在: {args.mask}")

        shaft_mask = read_mask(args.mask)

        if shaft_mask is None:
            raise SystemExit(f"钢轴主体 Mask 读取失败: {args.mask}")
    else:
        if not args.image.is_file():
            raise SystemExit(f"钢轴图片不存在: {args.image}")

        image = read_image(args.image)

        if image is None:
            raise SystemExit(f"钢轴图片读取失败: {args.image}")

        print("请框选钢轴：左右边缘贴紧钢轴两侧，上下覆盖主体即可")
        roi = cv2.selectROI("Select Shaft ROI", image, False, False)
        cv2.destroyWindow("Select Shaft ROI")
        roi_x, _, roi_width, roi_height = map(int, roi)

        if roi_width == 0 or roi_height == 0:
            raise SystemExit("没有选择钢轴 ROI")

        shaft_mask = create_shaft_mask_from_roi(image, roi)
        write_image(DEFAULT_SHAFT_MASK_PATH, shaft_mask)
        print("钢轴主体 Mask 已保存:", DEFAULT_SHAFT_MASK_PATH)

    diameter_mm = (
        args.diameter_mm
        if args.diameter_mm is not None
        else prompt_diameter_mm()
    )
    calibration = calibrate_shaft_mask(shaft_mask, diameter_mm)
    print_calibration(calibration)

    if args.mask is None:
        print(f"x_c（原图坐标）= {roi_x + calibration.center_x:.6f} px")


if __name__ == "__main__":
    main()
