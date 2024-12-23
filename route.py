from flask import Blueprint, render_template, request, jsonify, send_file
from transformers import AutoModel, AutoTokenizer
import os
import json

# 创建 Blueprint
pointer_read_bp = Blueprint('pointer_read', __name__, template_folder='templates')

# 设置存储路径
IMG_PATH = "./pointer_read/test.png"  # 将上传的图片保存到这个路径

@pointer_read_bp.route('/')
def index():
    return render_template('pointer_read.html')


def load_and_save_data(file_path, data=None):
    """
    加载和保存数据的通用函数。
    如果传入数据，则保存数据到文件；如果不传入数据，则加载数据并返回。
    """
    if data is not None:  # 保存数据
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:  # 加载数据
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}



@pointer_read_bp.route('/get_task_data', methods=['GET'])
def get_task_data():
    TASK_DATA_JSON = os.path.join(os.getcwd(), "model_train/tasks.json")
    try:
        # 加载现有的任务数据
        if os.path.exists(TASK_DATA_JSON):
            multitask_data = load_and_save_data(TASK_DATA_JSON).get("pointer", {})
        else:
            multitask_data = {}

        return jsonify(multitask_data), 200

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


# 图片上传与处理接口
@pointer_read_bp.route('api/read-image', methods=['POST'])
def read_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    task_id = request.form["task_id"]

    # 保存图片到指定路径
    file.save(IMG_PATH)

    # 模拟处理图片并返回
    ocr_data = process_image_logic(IMG_PATH, task_id)


    return jsonify({"result": ocr_data})


def process_image_logic(input_path, task_id):
    """
    模拟图像处理逻辑
    """

    # 模板图处理
    # %%
    import cv2
    import os
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.optimize as so

    candidate_point = np.array([
        (700, 766),
        (624, 659), (603, 528), (647, 403), (745, 311), (851, 276),
        (981, 292), (1086, 361), (1152, 472), (1160, 601),
        (1107, 720)
    ], dtype=float)

    # 为了表盘刻度区域不影响指针扫描 需要找到一块较干净的椭圆环型区域进行扫描
    # zone_point定义为该环形区域中线上的一点 用于缩放椭圆
    zone_point = (1046, 679)
    zone_radius = 10  # 扫描区域半径
    zone_theta = 15  # 扫描区域扇形角度
    vaild_threshold = 70  # 要求扇形区域至少包含了vaild_threshold个像素点才能被统计
    START_POINT = -0.1  # 起始刻度值
    END_POINT = 0.9  # 终止刻度值
    PRESICISION = 2  # 精确位数
    UNIT = 'MPa'
    # %%
    # 任务编号与目标检测图片，需要从前端获取
    # 逻辑是获取任务编号后从文件夹里读取唯一对应基准图片，然后变换
    task_id = task_id  # 需要保证这一类任务下的文件夹images下只有一个图片文件，上传时需覆盖
    gray_img2 = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2GRAY)  # 需要识别的图片

    annotations_path = f"{os.getcwd()}/pointer_annotation/annotations/{task_id}/annotations.json"

    # 读取 JSON 文件
    with open(annotations_path, 'r') as f:
        annotations_data = json.load(f)

    candidate_point = []

    # 获取图像路径
    image_dir = f"{os.getcwd()}/pointer_annotation/annotations/{task_id}/images/"
    #print(f"Image directory: {image_dir}")

    # 获取目录中的文件列表
    image_files = next(os.walk(image_dir), (None, None, []))[2]  # 获取该目录下的所有文件
    if image_files:
        image_path = os.path.join(image_dir, image_files[0])  # 获取第一个图片文件
        # print(f"Image path: {image_path}")
        image = cv2.imread(image_path)
    else:
        # print("No image files found in the directory.")
        image_path = None  # 如果没有图片，设为 None

    # 遍历 annotations 数据进行替换
    for annotation in annotations_data[next(os.walk(image_dir))[2][0]]['annotations']:
        if annotation['label'] == '1':
            #print(annotation['x'], annotation['y'])
            # 将 x, y 添加到 candidate_point 列表
            candidate_point.append((annotation['x'], annotation['y']))
        elif annotation['label'] == '0':
            # 获取 zone_point 的坐标
            zone_point = (annotation['x'], annotation['y'])

    candidate_point = np.array(candidate_point)

    # 获取新的 START_POINT 和 END_POINT
    START_POINT = float(annotations_data[next(os.walk(image_dir))[2][0]]['annotations'][0]['minValue'])
    END_POINT = float(annotations_data[next(os.walk(image_dir))[2][0]]['annotations'][0]['maxValue'])

    # %%
    def show_img(image):
        # OpenCV 读取的图像是 BGR 格式，而 Matplotlib 显示的是 RGB 格式
        # 因此需要将图像从 BGR 转换为 RGB
        if image is None:
            pass
            #print("Error: Unable to load image.")
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 使用 Matplotlib 显示图像
            plt.figure(figsize=(15, 15))
            plt.imshow(image_rgb)
            plt.axis('off')  # 关闭坐标轴
            #plt.show()

    # %%
    def show_point(image, x_point, y_point, color=(0, 255, 255), reversed=True):
        image_temp = image.copy()
        for (x, y) in zip(x_point, y_point):
            if reversed:
                cv2.circle(image_temp, (int(x), int(image.shape[1] - y)), 4, color, -1)
            else:
                cv2.circle(image_temp, (int(x), int(y)), 4, color, -1)
        #show_img(image_temp)

    # %%
    #print(image.shape)
    #show_img(image)

    # 拟合函数：点到椭圆两焦点距离之和
    def my_fun(parameters, x_samples, y_samples):
        # Unpack parameters: two focus points and the target distance sum
        x_focus_1, y_focus_1, x_focus_2, y_focus_2, sum_of_target_distance = parameters

        # Calculate the actual distances from the points to the two foci
        sum_of_actual_distance = (
                np.sqrt((x_samples - x_focus_1) ** 2 + (y_samples - y_focus_1) ** 2) +
                np.sqrt((x_samples - x_focus_2) ** 2 + (y_samples - y_focus_2) ** 2)
        )

        # Return the variance of the difference between actual and target distances
        return np.sum(((sum_of_actual_distance - sum_of_target_distance) ** 2) / (len(x_samples) - 1))

    # %%
    def fit_ellipse(x_samples, y_samples):

        # Optimize to fit the ellipse using initial guesses for the parameters
        initial_guess = np.array([np.mean(x_samples), np.mean(y_samples), np.mean(x_samples), np.mean(y_samples),
                                  100])  # Initial focus points and target distance
        res_optimized = so.minimize(fun=my_fun, x0=initial_guess, args=(x_samples, y_samples))

        # 拟合结束后 通过焦点坐标得到 半长短轴、椭圆中心、长轴与x轴正半轴夹角
        if res_optimized.success:
            # Unpack optimized parameters
            x1_res, y1_res, x2_res, y2_res, l2_res = res_optimized.x

            # Calculate the angle of the ellipse based on the foci
            alpha_res = np.arctan2(y2_res - y1_res, x2_res - x1_res)

            # Calculate the distance between the foci
            l_ab = np.sqrt((y2_res - y1_res) ** 2 + (x2_res - x1_res) ** 2)

            # Calculate semi-major and semi-minor axes
            a_res = l2_res / 2  # Semi-major axis length
            b_res = np.sqrt((l2_res / 2) ** 2 - (l_ab / 2) ** 2)  # Semi-minor axis length

            return a_res, b_res, (x1_res + x2_res) / 2, (y1_res + y2_res) / 2, alpha_res
        else:

            #print('Fail to fit ellipse')
            return None

    # %%
    x_samples = candidate_point.T[0]
    # 拟合过程前转换为直角坐标系（图像坐标系下的y是反的）
    y_samples = image.shape[1] - candidate_point.T[1]

    show_point(image, x_samples, y_samples)
    # Fit the ellipse to the generated samples
    a_res, b_res, x0, y0, alpha_res = fit_ellipse(x_samples, y_samples)

    # %%

    def get_Point_in_ellipse(h, k, a, b, alpha, theta):
        x = h + a * np.cos(alpha) * np.cos(theta) - b * np.sin(alpha) * np.sin(theta)
        y = k + a * np.cos(alpha) * np.sin(theta) + b * np.sin(alpha) * np.cos(theta)
        return x, y

    # '''
    # 当椭圆不仅需要平移，而且其长轴和短轴不与坐标轴平行时\会稍微复杂一些。此时，除了平移之外，还需要考虑旋转。假设椭圆中心位于点 \((h, k)\)，并且椭圆绕其中心旋转了一个角度 \(\theta\)（逆时针方向），那么椭圆的参数方程可以表示为：

    # \[ x = h + a \cos(\theta) \cos(\alpha) - b \sin(\theta) \sin(\alpha) \]
    # \[ y = k + a \cos(\theta) \sin(\alpha) + b \sin(\theta) \cos(\alpha) \]

    # 这里：
    # - \(a\) 是椭圆的半长轴。
    # - \(b\) 是椭圆的半短轴。
    # - \(theta\) 是参数，通常取值范围是从 \(0\) 到 \(2\pi\)。
    # - \((h, k)\) 是椭圆中心的坐标。
    # - \(\alpha\) 是椭圆长轴与正x轴之间的夹角。


    theta_res = np.linspace(0, 2 * np.pi, 100)  # Angle values
    x_res, y_res = get_Point_in_ellipse(x0, y0, a_res, b_res, alpha_res, theta_res)

    # 预览 检查椭圆曲线拟合
    show_point(image, x_res, y_res)

    # %%
    zone_point_x = zone_point[0]
    zone_point_y = image.shape[1] - zone_point[1]

    # 求出方向向量在椭圆上对应的坐标点
    theta_zone_point_in_ellipse = np.arctan2(zone_point_y - y0, zone_point_x - x0)
    point_in_ellipse_x, point_in_ellipse_y = get_Point_in_ellipse(x0, y0, a_res, b_res, alpha_res,
                                                                  theta_zone_point_in_ellipse)

    show_point(image, [point_in_ellipse_x, zone_point_x], [point_in_ellipse_y, zone_point_y])
    # %%
    zone_vec1 = np.array([zone_point_x, zone_point_y]) - np.array([x0, y0])
    zone_vec2 = np.array([point_in_ellipse_x, point_in_ellipse_y]) - np.array([x0, y0])
    s = np.linalg.norm(zone_vec1) / np.linalg.norm(zone_vec2)  # 根据向量长度求出缩放因子
    a_zone = a_res * s
    b_zone = b_res * s
    # %%
    # 扫描区域 中线的椭圆
    theta_res = np.linspace(0, 2 * np.pi, 100)  # 分割成100份
    x_zone, y_zone = get_Point_in_ellipse(x0, y0, a_zone, b_zone, alpha_res, theta_res)
    show_point(image, x_zone, y_zone, (0, 0, 255))
    # %%
    # 扫描区域可视化
    # zone_radius是自定义的一个向量，可能会出问题
    s_l = (np.linalg.norm(zone_vec1) - zone_radius) / np.linalg.norm(zone_vec2)  # 根据向量长度求出缩放因子
    s_r = (np.linalg.norm(zone_vec1) + zone_radius) / np.linalg.norm(zone_vec2)  # 根据向量长度求出缩放因子
    # 画两个圈
    x_zone_l, y_zone_l = get_Point_in_ellipse(x0, y0, a_res * s_l, b_res * s_l, alpha_res, theta_res)
    x_zone_r, y_zone_r = get_Point_in_ellipse(x0, y0, a_res * s_r, b_res * s_r, alpha_res, theta_res)

    show_point(image, np.concatenate((x_zone_l, x_zone_r)), np.concatenate((y_zone_l, y_zone_r)), (0, 0, 255))



    # %% md
    ## 计算刻度点坐标
    # %%
    # np.arctan2(y,x) 使用arctan2求弧度制夹角
    # 第一象限 0 -> pi （逆时针）
    # 第二象限 pi -> 2pi
    # 第三象限 -2pi -> -pi
    # 第四象限 -pi -> 0
    # %%
    # 刻度s到刻度t间 切分成separate_num份
    # 返回左闭右开的点集序列
    def separate_point(s_point, t_point, separate_num, keep_last_point=False):
        s_point_x, s_point_y = s_point[0], s_point[1]
        t_point_x, t_point_y = t_point[0], t_point[1]

        # # 两个相邻标点不应该超过90度
        theta_s = np.arctan2(s_point_y - y0, s_point_x - x0)
        theta_t = np.arctan2(t_point_y - y0, t_point_x - x0)
        if theta_s < theta_t: theta_s += np.pi * 2  # 负角度

        # 切分出separate_num + 1个点
        theta_small_point = np.linspace(theta_s, theta_t, separate_num + 1)
        x_small, y_small = get_Point_in_ellipse(x0, y0, a_res, b_res, alpha_res, theta_small_point)

        if not keep_last_point:
            x_small = x_small[:-1]
            y_small = y_small[:-1]
        # 预览
        return x_small, y_small


    samples_num = len(x_samples)
    x_samples_sep1, y_samples_sep1 = [], []

    for i in range(1, samples_num):
        s_point = (x_samples[i - 1], y_samples[i - 1])
        t_point = (x_samples[i], y_samples[i])

        x_res, y_res = separate_point(s_point, t_point, 5, i == samples_num - 1)
        x_samples_sep1.extend(x_res)
        y_samples_sep1.extend(y_res)

    x_samples_sep1 = np.array(x_samples_sep1)
    y_samples_sep1 = np.array(y_samples_sep1)
    # %%
    x_samples, y_samples = x_samples_sep1, y_samples_sep1
    show_point(image, x_samples, y_samples)
    # %% md


    # %%
    # 起点方向点
    theta_s = np.arctan2(y_samples[0] - y0, x_samples[0] - x0)  # 第一个点
    theta_t = np.arctan2(y_samples[1] - y0, x_samples[1] - x0)  # 第二个点
    if theta_s < theta_t: theta_s += np.pi * 2  # 负角度
    theta_newL = 2 * theta_s - theta_t  # theta_s是中点夹角

    point_newx, point_newy = get_Point_in_ellipse(x0, y0, a_res, b_res, alpha_res, [theta_newL])
    # show_point( image, np.concatenate((point_newx,x_samples[:2])), np.concatenate((point_newy, y_samples[:2])))
    show_point(image, point_newx, point_newy)

    ## 将这个新点添加到最前面
    x_samples = np.concatenate((point_newx, x_samples))
    y_samples = np.concatenate((point_newy, y_samples))

    # 终点方向点也生成和前面一样的点插入到最后段
    theta_s = np.arctan2(y_samples[-2] - y0, x_samples[-2] - x0)  # 倒数第二个点
    theta_t = np.arctan2(y_samples[-1] - y0, x_samples[-1] - x0)  # 倒数第一个点
    if theta_s < theta_t: theta_s += np.pi * 2  # 负角度
    theta_newL = 2 * theta_t - theta_s  # theta_t是中点夹角

    point_newx, point_newy = get_Point_in_ellipse(x0, y0, a_res, b_res, alpha_res, [theta_newL])
    show_point(image, np.concatenate((point_newx, x_samples[-2:])), np.concatenate((point_newy, y_samples[-2:])))

    ## 添加到最后面
    x_samples = np.concatenate((x_samples, point_newx))
    y_samples = np.concatenate((y_samples, point_newy))
    # %%
    show_point(image, x_samples, y_samples)
    # %%
    # 切分估读区域
    samples_num = len(x_samples)
    x_samples_sep2, y_samples_sep2 = [], []

    for i in range(1, samples_num):
        s_point = (x_samples[i - 1], y_samples[i - 1])
        t_point = (x_samples[i], y_samples[i])

        # 小刻度切分成10片模拟曲线？
        x_res, y_res = separate_point(s_point, t_point, 10, keep_last_point=(i == samples_num - 1))
        x_samples_sep2.extend(x_res)
        y_samples_sep2.extend(y_res)

    x_samples_sep2 = np.array(x_samples_sep2)
    y_samples_sep2 = np.array(y_samples_sep2)
    # %%
    x_samples, y_samples = x_samples_sep2, y_samples_sep2
    show_point(image, x_samples, y_samples)
    # %% md

    # %%
    # 接下来把点集移动到扫描区域中线即可
    theta_samples = [
        np.arctan2(y_samples[i] - y0, x_samples[i] - x0)
        for i in range(len(x_samples))
    ]
    x_res, y_res = get_Point_in_ellipse(x0, y0, a_zone, b_zone, alpha_res, theta_samples)
    #print(f'总共刻度点数 = {len(x_res)}')
    # %%
    show_point(image, x_res, y_res)
    # 转换回图片坐标
    candidate_point = np.array([x_res, image.shape[1] - y_res]).T
    # %% md
    ## 单应矩阵
    # %%
    template_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转化为灰度图像
    # %%
    #show_img(template_img)
    #show_img(gray_img2)

    # %%
    # AKAZE求解单应矩阵
    def AKAZE4H(gray_img1, gray_img2):

        akaze = cv2.AKAZE_create()
        # 检测关键点和计算描述符
        keypoints1, descriptors1 = akaze.detectAndCompute(gray_img1, None)
        keypoints2, descriptors2 = akaze.detectAndCompute(gray_img2, None)

        # 使用 BFMatcher 匹配描述符
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # 按照距离将匹配结果排序
        matches = sorted(matches, key=lambda x: x.distance)

        # 只保留前300个匹配（可调整）
        good_matches = matches[:300]

        if len(good_matches) > 4:  # At least 4 points needed to compute homography
            # 获取匹配点的坐标
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 计算单应矩阵
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # 可视化特征匹配
            height, width = gray_img2.shape
            warped_img = cv2.warpPerspective(gray_img1, H, (width, height))
            # Draw only inliers (matches used in homography)
            matches_mask = mask.ravel().tolist()
            draw_params = dict(matchColor=(0, 255, 0),  # Green color for inliers
                               singlePointColor=None,
                               matchesMask=matches_mask,  # Only draw inliers
                               flags=2)
            result_img = cv2.drawMatches(gray_img1, keypoints1, gray_img2, keypoints2, good_matches, None,
                                         **draw_params)
            show_img(result_img)

            return H
        else:
            print("Not enough matches found to compute homography.")
            return None

    # %%
    def transform_candidate(candidate_point, gray_img1, gray_img2):

        # 求解单应
        H = AKAZE4H(gray_img1, gray_img2)
        if H is None: return None

        # 将模板点点集透视变化到目标图中
        candidate_projected_points = cv2.perspectiveTransform(candidate_point.reshape(-1, 1, 2), H)
        return candidate_projected_points.reshape(-1, 2)

    # %%
    candidate_projected_points = transform_candidate(candidate_point, template_img, gray_img2)
    # %%
    show_point(gray_img2, candidate_projected_points.T[0], candidate_projected_points.T[1], reversed=False)
    # %% md
    ## 指针扫描

    #len(candidate_projected_points)  # 前面10个辅助点 + 真实刻度点 + 后10个

    # %%
    # 在其邻近点的斜率基础上，估算出每个点的切线角度，再转为方向角
    def avg_alpha(points):

        points_num = len(points)
        alpha = [0 for i in range(points_num - 20)]

        for i in range(10, points_num - 10):

            # 累加左右10个点
            sum_k, cnt_empty = 0, 0
            for j in range(1, 10 + 1):

                delta_y = points[i + j][1] - points[i - j][1]
                delta_x = points[i + j][0] - points[i - j][0]

                # 记录斜率不存在的情况
                if np.fabs(delta_x) < 1e-6:
                    cnt_empty += 1
                else:
                    sum_k += delta_y / delta_x

            # 斜率不存在的情况较多 直接设为0
            if cnt_empty > 5:
                alpha[i - 10] = 0.0
            else:
                k_i = sum_k / (10 - cnt_empty)
                if np.fabs(k_i) < 1e-6:
                    alpha[i - 10] = 90.0
                else:
                    alpha[i - 10] = np.arctan(-1 / k_i) / np.pi * 180  # 弧度转角度值
        return np.radians(alpha)

    alpha_points = avg_alpha(candidate_projected_points)

    # %%
    # 绘制线段
    def draw_segment(gray_img, x1, y1, x2, y2):
        color_image = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        for (x_s, y_s, x_t, y_t) in zip(x1, y1, x2, y2):
            cv2.line(color_image, (x_s, y_s), (x_t, y_t), (0, 0, 255), thickness=2)

        #show_img(color_image)

    for i in range(len(alpha_points)):
        x, y = candidate_projected_points[i + 10]

    # 求出所有刻度点的刻度线线段
    x_start = [int(candidate_projected_points[i + 10][0] - zone_radius * np.cos(alpha_points[i])) for i in
               range(len(alpha_points))]
    y_start = [int(candidate_projected_points[i + 10][1] - zone_radius * np.sin(alpha_points[i])) for i in
               range(len(alpha_points))]
    x_end = [int(candidate_projected_points[i + 10][0] + zone_radius * np.cos(alpha_points[i])) for i in
             range(len(alpha_points))]
    y_end = [int(candidate_projected_points[i + 10][1] + zone_radius * np.sin(alpha_points[i])) for i in
             range(len(alpha_points))]
    # 展示一部分
    draw_segment(gray_img2, x_start[::10], y_start[::10], x_end[::10], y_end[::10])

    # %%
    res_img = gray_img2.copy()
    res_img = cv2.medianBlur(res_img, 9)

    res_img = cv2.adaptiveThreshold(res_img, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,
                                    11, 2)
    show_img(res_img)



    # 叉积
    def cross(x1, y1, x2, y2):
        return x1 * y2 - x2 * y1

    # 精度误差
    def sgn(x):
        return np.sign(x) if np.fabs(x) >= 1e-6 else 0


    # %%
    temp_img = res_img.copy()
    for i in range(0, len(alpha_points), 10):

        x = candidate_projected_points[i + 10][0]
        y = candidate_projected_points[i + 10][1]

        vec1_x = np.cos(alpha_points[i] + zone_theta)
        vec1_y = np.sin(alpha_points[i] + zone_theta)

        vec2_x = np.cos(alpha_points[i] - zone_theta)
        vec2_y = np.sin(alpha_points[i] - zone_theta)

        # 用矩形区域框出
        x_min, x_max, y_min, y_max = int(x - zone_radius), int(x + zone_radius), int(y - zone_radius), int(
            y + zone_radius)
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, res_img.shape[1])
        y_max = min(y_max, res_img.shape[0])

        # sum_pixel = 0
        # cnt_inside_pixel = 0

        for ty in range(y_min, y_max):
            for tx in range(x_min, x_max):
                vx, vy = tx - x, ty - y

                # 超出扇形半径
                if vx ** 2 + vy ** 2 > (zone_radius) ** 2: continue

                # 位于两个扇形内
                if sgn(cross(vec1_x, vec1_y, vx, vy)) * sgn(cross(vec2_x, vec2_y, vx, vy)) <= 0:
                    temp_img[ty, tx] = 0

    #show_img(temp_img)
    # %%
    res_i, min_pixel = 0, 1e9  # 答案刻度点及像素均值

    for i in range(0, len(alpha_points)):

        x = candidate_projected_points[i + 10][0]
        y = candidate_projected_points[i + 10][1]

        vec1_x = np.cos(alpha_points[i] + zone_theta)
        vec1_y = np.sin(alpha_points[i] + zone_theta)

        vec2_x = np.cos(alpha_points[i] - zone_theta)
        vec2_y = np.sin(alpha_points[i] - zone_theta)

        # 用矩形区域框出
        x_min, x_max, y_min, y_max = int(x - zone_radius), int(x + zone_radius), int(y - zone_radius), int(
            y + zone_radius)
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, res_img.shape[1])
        y_max = min(y_max, res_img.shape[0])

        sum_pixel = 0.0
        cnt_inside_pixel = 0

        for ty in range(y_min, y_max):
            for tx in range(x_min, x_max):
                vx, vy = tx - x, ty - y
                # 超出扇形半径
                if vx ** 2 + vy ** 2 > (zone_radius) ** 2: continue

                # 位于两个扇形内
                if sgn(cross(vec1_x, vec1_y, vx, vy)) * sgn(cross(vec2_x, vec2_y, vx, vy)) <= 0:
                    sum_pixel += res_img[ty, tx]
                    cnt_inside_pixel += 1

        if cnt_inside_pixel > vaild_threshold and sum_pixel / cnt_inside_pixel < min_pixel:
            min_pixel = sum_pixel / cnt_inside_pixel
            res_i = i



    draw_segment(gray_img2, [x_start[res_i]], [y_start[res_i]], [x_end[res_i]], [y_end[res_i]])

    P = (res_i / (len(alpha_points) - 1)) * (END_POINT - START_POINT) + START_POINT

    res = f'读数 = {round(P, PRESICISION)}'


    return res

