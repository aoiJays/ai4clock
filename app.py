from shutil import rmtree

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import yaml
import os
from skimage.morphology import dilation, erosion, disk, skeletonize
from sklearn.cluster import KMeans
import time
from skimage.filters import median

def load_specific_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # 提取特定配置项
    image_path = config.get('image', {}).get('path')
    candidate_points = config.get('candidate_points', [])
    zone_point = config.get('scan_zone', {}).get('center_point', [0, 0])
    zone_radius = config.get('scan_zone', {}).get('radius', 0)
    zone_theta = config.get('scan_zone', {}).get('angle', 0)
    valid_threshold = config.get('scan_zone', {}).get('valid_threshold', 0)
    step = config.get('scale', {}).get('step', 5)
    start_point = config.get('scale', {}).get('start', 0.0)
    end_point = config.get('scale', {}).get('end', 0.0)
    precision = config.get('scale', {}).get('precision', 0)
    unit = config.get('scale', {}).get('unit', '')

    return (
        image_path,
        candidate_points,
        zone_point,
        zone_radius,
        zone_theta,
        valid_threshold,
        step,
        start_point,
        end_point,
        precision,
        unit
    )

def my_fun(parameters, x_samples, y_samples):
    # Unpack parameters: two focus points and the target distance sum
    x_focus_1, y_focus_1, x_focus_2, y_focus_2, sum_of_target_distance = parameters

    # Calculate the actual distances from the points to the two foci
    distance_to_focus_1 = np.sqrt((x_samples - x_focus_1) ** 2 + (y_samples - y_focus_1) ** 2)
    distance_to_focus_2 = np.sqrt((x_samples - x_focus_2) ** 2 + (y_samples - y_focus_2) ** 2)

    # Return the difference between actual and target distances for each point
    return distance_to_focus_1 + distance_to_focus_2 - sum_of_target_distance

def fit_ellipse(x_samples, y_samples):

    # Compute the centroid of the samples
    centroid_x = np.mean(x_samples)
    centroid_y = np.mean(y_samples)

    # Find the point farthest from the centroid
    distances_from_centroid = np.sqrt((x_samples - centroid_x) ** 2 + (y_samples - centroid_y) ** 2)
    min_distance_index = np.argmin(distances_from_centroid)


    # Estimate the sum of distances from any point on the ellipse to the foci
    sum_of_target_distance_init = 2 * distances_from_centroid[min_distance_index]

    # Optimize to fit the ellipse using initial guesses for the parameters
    initial_guess = np.array([centroid_x, centroid_y, centroid_x, centroid_y, sum_of_target_distance_init])
    res_optimized = least_squares(fun=my_fun, x0=initial_guess, args=(x_samples, y_samples))

    if res_optimized.success:
        # Unpack optimized parameters
        x1_res, y1_res, x2_res, y2_res, l2_res = res_optimized.x

        # Calculate the angle of the ellipse based on the foci
        alpha_res = np.arctan2(y2_res - y1_res, x2_res - x1_res)

        # Calculate the distance between the foci
        l_ab = np.sqrt((y2_res - y1_res) ** 2 + (x2_res - x1_res) ** 2)

        # Calculate semi-major and semi-minor axes
        a_res = l2_res / 2  # Semi-major axis length
        b_res = np.sqrt(a_res**2 - (l_ab / 2)**2)  # Semi-minor axis length

        return a_res, b_res, (x1_res+x2_res)/2, (y1_res+y2_res)/2, alpha_res
    else:
        print('Fail to fit ellipse')
        return None


# (x, y) rotate around (h,k) with alpha
def rotate(x, y, alpha):
    x_rot = x * np.cos(alpha) - y * np.sin(alpha)
    y_rot = x * np.sin(alpha) + y * np.cos(alpha)
    return x_rot, y_rot
def get_Point_in_ellipse(a, b, theta):
    return a * np.cos(theta) , b * np.sin(theta)

def Point_in_ellipse(h, k, a, b, alpha, theta):
    x0, y0 = np.cos(theta), np.sin(theta)
    x_real_rot, y_real_rot = rotate(x0, y0, -alpha)
    x_real_rot, y_real_rot = get_Point_in_ellipse(a, b, np.arctan2(y_real_rot, x_real_rot))
    x_res, y_res = rotate(x_real_rot, y_real_rot, alpha)
    return x_res + h, y_res + k


def SIFT4H(img1, img2, debug):

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    # 使用SIFT找到关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN 参数设计
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # 或传递一个空字典

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 准备计算单应性矩阵的数据
    good_matches = []
    pts1 = []
    pts2 = []

    # 应用比例测试
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.float32(pts1).reshape(-1, 2)
    pts2 = np.float32(pts2).reshape(-1, 2)

    # 获取单应性矩阵
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # 绘制匹配结果
    draw_params = dict(matchColor=(0, 255, 0),  # 使用绿色绘制匹配项
                       singlePointColor=None,
                       matchesMask=mask.ravel().tolist(),  # 只绘制内部点
                       flags=2)


    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

    if not debug: return H
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title('Matches')
    plt.show()

    return H

def save_cv_image_with_plt(cv_image, output_path):
    # 如果图像是彩色的，则将其从BGR转换为RGB
    if len(cv_image.shape) == 3:
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = cv_image  # 灰度图像不需要转换

    # 创建一个新的图形，并关闭坐标轴
    plt.figure()
    plt.axis('off')  # 关闭坐标轴
    plt.gca().set_axis_off()  # 隐藏边框

    # 显示图像
    plt.imshow(rgb_image, cmap='gray' if len(cv_image.shape) == 2 else None)

    # 保存图像到文件，bbox_inches='tight' 和 pad_inches=0 用来去掉边缘空白
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()  # 关闭当前图形以释放内存

# 使用单应矩阵 对点进行转换
# 图像坐标系
def transform_point_with_H(x_point, y_point, H):
    point = np.array([x_point, y_point], dtype=np.float32).T
    projected_points = (cv2.perspectiveTransform(point.reshape(-1,1,2), H).reshape(-1,2).T)
    return projected_points[0], projected_points[1]


def warp_image_and_points_with_H(H, img2, x_point, y_point):
    # 获取img2的尺寸
    height, width = img2.shape[:2]

    # 定义img2四个角点的位置
    corners = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)

    # 使用perspectiveTransform来找到四个角点变换后的位置
    transformed_corners = cv2.perspectiveTransform(corners, H)

    # 找到变换后的最小外接矩形
    x_min = min(transformed_corners[:, 0, 0])
    y_min = min(transformed_corners[:, 0, 1])
    x_max = max(transformed_corners[:, 0, 0])
    y_max = max(transformed_corners[:, 0, 1])

    # 计算新图像的宽度和高度
    new_width = int(np.round(x_max - x_min))
    new_height = int(np.round(y_max - y_min))

    # 调整H矩阵，使得变换后的图像不被裁剪
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    adjusted_H = np.dot(translation_matrix, H)

    # 应用透视变换到图像
    warped_img2 = cv2.warpPerspective(img2, adjusted_H, (new_width, new_height))

    # 将输入的点转换为正确的格式，并应用相同的变换矩阵
    points = np.array([x_point, y_point], dtype=np.float32).T.reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, translation_matrix).reshape(-1, 2).T
    # 返回变换后的图像和点集
    return warped_img2, transformed_points[0], transformed_points[1]

def mask_in_ellipse(img, a, b, x0, y0, alpha):

    # Generate meshgrid for efficient computation
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    # Compute rotated coordinates
    x_rot = (X - x0) * cos_alpha + (Y - y0) * sin_alpha
    y_rot = -(X - x0) * sin_alpha + (Y - y0) * cos_alpha

    # Compute the ellipse equation and create a mask
    ellipse_mask = ((x_rot ** 2) / (a ** 2)) + ((y_rot ** 2) / (b ** 2)) <= 1

    # print('debug mask size', ellipse_mask.shape, img.shape)
    return ellipse_mask

def crop_with_mask(img, mask, x0, y0, crop=True):
    # Apply the mask to the image
    img_masked = img.copy()

    if img.ndim == 3:
        img_masked[~mask] = [0, 0, 0]
    else :
        img_masked[~mask] = 0

    if not crop: return img_masked

    # Find bounding box of the non-zero elements in the masked image
    rows = np.any(img_masked, axis=1)
    cols = np.any(img_masked, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Crop the image using the bounding box
    cropped_img = img_masked[ymin:ymax+1, xmin:xmax+1]

    # Calculate new center coordinates relative to the cropped image
    new_x0 = x0 - xmin
    new_y0 = y0 - ymin

    return cropped_img, new_x0, new_y0



def kmeans_binarization(gray_img, mask):
    """
    对输入图像应用中值滤波去噪和K-means聚类进行二值化。
    仅对 mask 为 True 的像素点进行处理。

    参数:
        gray_img (numpy.ndarray): 输入的灰度图像。
        mask (numpy.ndarray): 布尔类型掩码，True 表示需要处理的像素，False 表示保留原始值。

    返回:
        numpy.ndarray: 二值化后的灰度图像。
    """
    # 确保 mask 和图像尺寸一致
    assert gray_img.shape == mask.shape, "gray_img 和 mask 必须尺寸相同"

    # 提取需要处理的像素
    pixel_vals = gray_img[mask].reshape((-1, 1))

    # 转换为浮点数类型
    pixel_vals = np.float32(pixel_vals)

    # 执行 K-means 聚类
    kmeans = KMeans(n_clusters=2, random_state=0)  # 设置随机种子以获得可重复的结果
    labels = kmeans.fit_predict(pixel_vals)

    # 获取质心并排序，以确定哪个是背景，哪个是前景
    quantized = kmeans.cluster_centers_[labels]
    sorted_centroids = np.sort(kmeans.cluster_centers_, axis=0)

    # 创建二值图像
    threshold_value = sorted_centroids.mean()  # 取两个质心的中间值作为阈值
    binary_vals = np.where(quantized < threshold_value, 255, 0).astype(np.uint8)

    # 创建与原图像同尺寸的二值图像
    binary_img = gray_img.copy()
    binary_img[mask] = binary_vals.flatten()  # 仅更新 mask 中的像素

    return binary_img

def enhance_contrast(gray_img, gamma=2.0):
    """
    增强灰度图像的暗部和亮部对比度。

    参数:
        gray_img (numpy.ndarray): 输入的灰度图像。
        gamma (float): 非线性调整的Gamma值，默认值为2.0。
                       大于1增强亮部，小于1增强暗部。

    返回:
        numpy.ndarray: 对比度增强后的图像。
    """

    # 线性拉伸对比度
    min_val, max_val = np.min(gray_img), np.max(gray_img)
    stretched_img = ((gray_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # 非线性调整，增强对比
    adjusted_img = np.power(stretched_img / 255.0, gamma) * 255
    return stretched_img.astype(np.uint8)


def convert_to_color_and_draw_lines(gray_image, zone_radius):

    # 使用Canny边缘检测器找出边缘
    edges = gray_image.copy()

    # 使用HoughLinesP函数进行概率霍夫变换以检测线段
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=zone_radius*1.2, maxLineGap=10)

    # 将灰度图像转换为三通道彩色图像
    color_image = cv2.cvtColor(gray_image.copy(), cv2.COLOR_GRAY2BGR)

    # 如果有检测到线段，则在彩色图上绘制这些线段
    line_res = []

    if lines is None:
        print('Falied')
        exit(0)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 在彩色图上绘制检测到的线
        line_res.append( line[0] )
        # print( (x1, y1), (x2, y2) )
        # show_img(color_image)
    return color_image, line_res

def cross(x1, y1, x2, y2):
    return x1 * y2 - x2 * y1

def dot(x1, y1, x2, y2):
    return x1 * x2 + y1 * y2


def point_in_zone(x0, y0, x, y, vec_x, vec_y):
    x,y,vec_x, vec_y = x - x0, y - y0, vec_x - x0, vec_y - y0
    return cross(x,y,vec_x, vec_y) >= 0 and dot(x,y,vec_x, vec_y) >= 0

def vectors_angle(ux, uy, vx, vy):
    # 计算每个向量与正X轴之间的角度（以弧度为单位）
    angle_u = np.arctan2(uy, ux)
    angle_v = np.arctan2(vy, vx)

    # 计算两个角度之间的差值，并调整到[-pi, pi]范围内
    angle_diff = angle_v - angle_u
    if angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2 * np.pi

    # 将角度差转换为度数
    return np.degrees(abs(angle_diff))

def get_scan_zone_mask(img, x0, y0, xp, yp, alpha, scan_zone_r):
    # 生成与 img 相同大小的掩码
    alpha = np.radians(alpha)
    mask = np.zeros_like(img, dtype=bool)

    # 生成网格，这里 X 对应列，Y 对应行
    Y, X = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')

    X_shifted = X - x0
    Y_shifted = Y - y0

    # 计算圆形掩码
    round_mask = X_shifted ** 2 + Y_shifted ** 2 <= scan_zone_r ** 2

    # 限制点到圆形范围后再计算
    X_shifted = X_shifted[round_mask]
    Y_shifted = Y_shifted[round_mask]

    # 计算旋转后的点
    xp1, yp1 = rotate(xp - x0, yp - y0, alpha)
    xp2, yp2 = rotate(xp - x0, yp - y0, -alpha)

    # 计算向量 (x0, y0) -> (xp1, yp1) 和 (x0, y0) -> (xp2, yp2)
    v1_x, v1_y = xp1, yp1
    v2_x, v2_y = xp2, yp2

    # 计算交叉乘积
    cross_prod1 = cross(v1_x, v1_y, X_shifted, Y_shifted)
    cross_prod2 = cross(v2_x, v2_y, X_shifted, Y_shifted)
    reversed_cross_prod1 = cross(-v1_x, -v1_y, X_shifted, Y_shifted)
    reversed_cross_prod2 = cross(-v2_x, -v2_y, X_shifted, Y_shifted)

    # 创建角度区域掩码
    angle_mask = (reversed_cross_prod1 >= 0) & (reversed_cross_prod2 <= 0) | (cross_prod1 >= 0) & (cross_prod2 <= 0)

    # 合并掩码，更新原始的 mask
    mask[round_mask] = angle_mask

    return mask

def draw_text(img, text):

    position = (50, 50)  # 文字左下角的位置
    # 定义字体、缩放比例、颜色和线宽
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    color = (255, 255, 0)
    thickness = 2

    # 使用 cv2.putText() 方法绘制文字
    cv2.putText(img, text, position, font, scale, color, thickness, lineType=cv2.LINE_AA)
    return img

def apply_gamma_correction(img, gamma=2.2):
    """
    应用伽马校正来调整图像亮度
    """
    # 创建伽马校正的映射表
    inv_gamma = 1.0 / gamma
    lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")

    # 应用伽马校正
    corrected_image = cv2.LUT(img, lookup_table)

    return corrected_image

def canny_edge_detection(img, low_threshold=50, high_threshold=150):
    """
    使用 Canny 算法检测边缘。

    参数:
    - img: 输入的灰度图像（numpy array）。
    - low_threshold: Canny 算法的低阈值。
    - high_threshold: Canny 算法的高阈值。

    返回:
    - 边缘图像（numpy array）。
    """
    # 使用Canny算法进行边缘检测
    edges = cv2.Canny(img, low_threshold, high_threshold)
    return edges

def process_image(source_path, test_path, output_path):

    config_file_path = os.path.join(source_path, 'config.yaml')

    (
        image_path,
        candidate_point,
        zone_point,
        zone_radius,
        zone_theta,
        valid_threshold,
        step,
        START_POINT,
        END_POINT,
        PRESICISION,
        UNIT
    ) = load_specific_config(config_file_path)

    # 打印提取的配置项以检查是否正确加载
    # print(f"Image Path: {image_path}")
    # print(f"Candidate Points: {candidate_point}\n")
    # print(f"Zone Point: {zone_point}")
    # print(f"Zone Radius: {zone_radius}")
    # print(f"Zone Theta: {zone_theta}")
    # print(f"Valid Threshold: {valid_threshold}")
    # print(f"step {step}")
    # print(f"Start Point: {START_POINT}\nEnd Point: {END_POINT}")
    # print(f"Precision: {PRESICISION}\nUnit: {UNIT}")

    # 读取图像
    img1 = cv2.imread(image_path)
    img2 = cv2.imread(test_path)

    candidate_point = np.array(candidate_point)
    candidate_point = np.vstack((candidate_point, zone_point))

    x_samples = candidate_point.T[0]
    y_samples = candidate_point.T[1]

    # 对待测图进行矫正 将标注点标注在新的图像上

    H_img2_img1 = SIFT4H(img2, img1, False)
    work_img, x_img_correct, y_img_correct = warp_image_and_points_with_H(H_img2_img1, img2, x_samples, y_samples)

    zone_point_x, zone_point_y = x_img_correct[-1], y_img_correct[-1]

    x_img_correct = x_img_correct[:-1]
    y_img_correct = y_img_correct[:-1]

    # 求解椭圆
    a_res, b_res, x0, y0, alpha_res = fit_ellipse(x_img_correct, y_img_correct)

    theta_res = np.linspace(0, 2 * np.pi, 50)
    x_res, y_res = Point_in_ellipse(x0, y0, a_res, b_res, alpha_res, theta_res)

    zone_vec1 = np.array([zone_point_x, zone_point_y]) - np.array([x0, y0])
    zone_vec2 = np.array([x_res[0], y_res[0]]) - np.array([x0, y0])
    s = np.linalg.norm(zone_vec1) / np.linalg.norm(zone_vec2)  # 根据向量长度求出缩放因子


    # # 扫描区域可视化
    s_l = (np.linalg.norm(zone_vec1) - zone_radius) / np.linalg.norm(zone_vec2)  # 根据向量长度求出缩放因子
    s_r = (np.linalg.norm(zone_vec1) + zone_radius) / np.linalg.norm(zone_vec2)  # 根据向量长度求出缩放因子
    # x_zone_l, y_zone_l = Point_in_ellipse(x0, y0, a_res * s_l, b_res * s_l, alpha_res, theta_res)
    # x_zone_r, y_zone_r = Point_in_ellipse(x0, y0, a_res * s_r, b_res * s_r, alpha_res, theta_res)
    #
    # show_point(work_img, np.concatenate((x_zone_l, x_zone_r)), np.concatenate((y_zone_l, y_zone_r)), (0, 0, 255))
    #

    theta_img_correct = np.arctan2(y_img_correct - y0, x_img_correct - x0)
    work_img_with_masked, x0, y0 = crop_with_mask(work_img, mask_in_ellipse(work_img, a_res, b_res, x0, y0, alpha_res),
                                                  x0, y0)
    theta_crop = np.linspace(0, 2 * np.pi, 50)
    x_crop, y_crop = Point_in_ellipse(x0, y0, a_res, b_res, alpha_res, theta_res)
    x_small_correct, y_small_correct = Point_in_ellipse(x0, y0, a_res, b_res, alpha_res, theta_img_correct)

    # 灰度图像增强对比度
    gray_img = cv2.cvtColor(work_img_with_masked, cv2.COLOR_BGR2GRAY)

    mask_inside = ~mask_in_ellipse(gray_img, a_res * s_l, b_res * s_l, x0, y0, alpha_res)
    mask_outside = mask_in_ellipse(gray_img, a_res * s_r, b_res * s_r, x0, y0, alpha_res)
    test_img = crop_with_mask(gray_img.copy(), mask_inside & mask_outside, x0, y0, crop=False)

    test_img = median(test_img, disk(3))
    test_img = enhance_contrast(test_img)

    binary_img = kmeans_binarization(test_img, mask_inside & mask_outside)
    scan_zone_img = binary_img.copy()

    scan_debug_img = gray_img.copy()
    output_res, max_pixel_value_mean, res_mask = None, 0, None
    res_x, res_y = None, None

    for i in range(len(x_small_correct) - 1):

        s_point_x, s_point_y = x_small_correct[i], y_small_correct[i]
        t_point_x, t_point_y = x_small_correct[i + 1], y_small_correct[i + 1]

        # # 两个相邻标点不应该超过90度
        theta_s = np.arctan2(s_point_y - y0, s_point_x - x0)
        theta_t = np.arctan2(t_point_y - y0, t_point_x - x0)

        # 响邻点角度差不能超过90
        while np.abs(theta_t - theta_s) > np.pi:
            if theta_s < theta_t:
                theta_s += np.pi * 2
            else:
                theta_t += np.pi * 2

        # debug 切分出step + 1个点
        # theta_small_point = np.linspace(theta_s, theta_t, step + 1)
        # x_small, y_small = Point_in_ellipse(x0, y0, a_res * s , b_res * s, alpha_res, theta_small_point)
        # show_point(scan_debug_img, x_small, y_small, (255, 255, 0))

        # 切分出step * 10 + 1个点
        theta_small_point = np.linspace(theta_s, theta_t, step * 10 + 1)
        x_small, y_small = Point_in_ellipse(x0, y0, a_res * s, b_res * s, alpha_res, theta_small_point)

        min_x = max(0, int(x_small.min()))
        max_x = min(scan_zone_img.shape[0], int(x_small.max()))
        min_y = max(0, int(y_small.min()))
        max_y = min(scan_zone_img.shape[1], int(y_small.max()))
        # print(min_x, min_y, max_x, max_y)
        if scan_zone_img[min_y:max_y + 1, min_x:max_x + 1].max() == 0: continue
        # show_point(scan_debug_img, x_small, y_small, (255, 255, 0))

        L_res = ((END_POINT - START_POINT) / (len(x_small_correct) - 1)) * i + START_POINT
        R_res = ((END_POINT - START_POINT) / (len(x_small_correct) - 1)) * (i + 1) + START_POINT

        # print(L_res, R_res)
        for j, (xp, yp) in enumerate(zip(x_small, y_small)):
            scan_zone_mask = get_scan_zone_mask(scan_zone_img, xp, yp, x0, y0, zone_theta, zone_radius)
            # scan_debug_img[scan_zone_mask] = 0

            if np.any(scan_zone_mask):
                pixel_value_mean = scan_zone_img[scan_zone_mask].mean()
            else:
                pixel_value_mean = 0
            pixel_res = L_res + (R_res - L_res) / (step * 10) * j

            if max_pixel_value_mean < pixel_value_mean:
                max_pixel_value_mean = pixel_value_mean
                output_res = pixel_res
                res_x, res_y = xp, yp
                res_mask = scan_zone_mask

        # if output_res is not None:
        #     scan_debug_img[res_mask] = 0

    print('finish')

    filename_with_extension = os.path.basename(test_path)
    output_result_path = os.path.join(output_path, filename_with_extension)

    if output_res is not None:
        scan_debug_img[res_mask] = 0
        print(output_res)
        print(f'res = {output_res:.{PRESICISION}f}{UNIT}')
        img2 = draw_text(img2, f'res = {output_res:.{PRESICISION}f}{UNIT}')
    else:
        print('failed')
        exit(0)
        img2 = draw_text(img2, 'failed')
    save_cv_image_with_plt(img2, output_result_path)


if __name__ == '__main__':

    source_path = 'source/1.3'
    output_path = 'output/'
    if os.path.exists(source_path):
        rmtree(output_path)
        os.mkdir(output_path)

    entries = os.listdir(os.path.join(source_path,'img'))


    for i, entry in enumerate(entries):
        full_path = os.path.join(source_path, 'img', entry)
        print(f'checking for {i}/{len(entries)}', full_path)
        start_time = time.time()
        process_image(source_path, full_path, output_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"程序运行时间: {elapsed_time} 秒")
