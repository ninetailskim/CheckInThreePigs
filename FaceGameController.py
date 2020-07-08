import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import paddlehub as hub
from paddlehub.common.logger import logger
import numpy as np
import math

class HeadPostEstimation(object):
    """
    头部姿态识别 upgrade from https://aistudio.baidu.com/aistudio/projectdetail/504933
    """
    NOD_ACTION = 1
    SHAKE_ACTION = 2
    def __init__(self, face_detector=None):
        self.module = hub.Module(name="face_landmark_localization", face_detector_module=face_detector)
        # 头部3D关键点坐标
        self.model_points = np.array([
            [6.825897, 6.760612, 4.402142],
            [1.330353, 7.122144, 6.903745],
            [-1.330353, 7.122144, 6.903745],
            [-6.825897, 6.760612, 4.402142],
            [5.311432, 5.485328, 3.987654],
            [1.789930, 5.393625, 4.413414],
            [-1.789930, 5.393625, 4.413414],
            [-5.311432, 5.485328, 3.987654],
            [2.005628, 1.409845, 6.165652],
            [-2.005628, 1.409845, 6.165652],
            [2.774015, -2.080775, 5.048531],
            [-2.774015, -2.080775, 5.048531],
            [0.000000, -3.116408, 6.097667],
            [0.000000, -7.415691, 4.070434],
            [-7.308957, 0.913869, 0.000000],
            [7.308957, 0.913869, 0.000000],
            [0.746313,0.348381,6.263227],
            [0.000000,0.000000,6.763430],
            [-0.746313,0.348381,6.263227],
            ], dtype='float')

        # 点头动作index是0， 摇头动作index是1
        # 当连续30帧上下点头动作幅度超过15度时，认为发生了点头动作
        # 当连续30帧上下点头动作幅度超过45度时，认为发生了摇头动作，由于摇头动作较为敏感，故所需幅度更大
        self._index_action = {0:'nod', 1:'shake'}
        self._frame_window_size = 15
        self._pose_threshold = {0: 15/180 * math.pi,
                                1: 45/180 * math.pi}
        # 头部3D投影点
        self.reprojectsrc = np.float32([
            [10.0, 10.0, 10.0],
            [10.0, 10.0, -10.0], 
            [10.0, -10.0, -10.0],
            [10.0, -10.0, 10.0], 
            [-10.0, 10.0, 10.0], 
            [-10.0, 10.0, -10.0], 
            [-10.0, -10.0, -10.0],
            [-10.0, -10.0, 10.0]])
        # 头部3D投影点连线
        self.line_pairs = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

    @property
    def frame_window_size(self):
        return self._frame_window_size
    
    @frame_window_size.setter
    def frame_window_size(self, value):
        assert isinstance(value, int)
        self._frame_window_size = value

    @property
    def pose_threshold(self):
        return self._pose_threshold
    
    @pose_threshold.setter
    def pose_threshold(self, dict_value):
        assert list(dict_value.keys()) == [0,1,2]
        self._pose_threshold = dict_value

    def get_face_landmark(self, image):
        """
        预测人脸的68个关键点坐标
        images(ndarray): 单张图片的像素数据
        """
        try:
            # 选择GPU运行，use_gpu=True，并且在运行整个教程代码之前设置CUDA_VISIBLE_DEVICES环境变量
            res = self.module.keypoint_detection(images=[image], use_gpu=False)
            return True, res[0]['data'][0]
        except Exception as e:
            logger.error("Get face landmark localization failed! Exception: %s " % e)
            return False, None
        
    def get_image_points_from_landmark(self, face_landmark):
        """
        从face_landmark_localization的检测结果抽取姿态估计需要的点坐标
        """
        image_points = np.array([
            face_landmark[17], face_landmark[21], 
            face_landmark[22], face_landmark[26], 
            face_landmark[36], face_landmark[39], 
            face_landmark[42], face_landmark[45], 
            face_landmark[31], face_landmark[35],
            face_landmark[48], face_landmark[54],
            face_landmark[57], face_landmark[8],
            face_landmark[14], face_landmark[2], 
            face_landmark[32], face_landmark[33],
            face_landmark[34], 
            ], dtype='float')
        return image_points
    
    def caculate_pose_vector(self, image_points):
        """
        获取旋转向量和平移向量
        """
        # 相机视角
        center = (self.img_size[1]/2, self.img_size[0]/2)
        focal_length = center[0] / np.tan(60/ 2 * np.pi / 180)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]],
            dtype = "float")
        # 假设没有畸变
        dist_coeffs = np.zeros((4,1))
        
        success, rotation_vector, translation_vector= cv2.solvePnP(self.model_points, 
                                                                   image_points,
                                                                   camera_matrix, 
                                                                   dist_coeffs)
                                                                   
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs, reprojectdst

    def caculate_euler_angle(self, rotation_vector, translation_vector):
        """
        将旋转向量转换为欧拉角
        """
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
        return pitch, yaw, roll

    
    def classify_pose_in_euler_angles(self, video, poses=3):
        """
        根据欧拉角分类头部姿态(点头nod/摇头shake)
        video 表示不断产生图片的生成器
        pose=1 表示识别点头动作
        pose=2 表示识别摇头动作
        pose=3 表示识别点头和摇头动作
        """
        frames_euler = []
        self.nod_time = self.totate_time = self.shake_time = time.time()
        self.action_time = 0
        index_action ={0:[self.NOD_ACTION], 1:[self.SHAKE_ACTION]}

        for index, img in enumerate(video(), start=1):
            self.img_size = img.shape

            success, face_landmark = self.get_face_landmark(img)

            for i, action in enumerate(index_action):
                if i == 0:
                    index_action[action].append((20, int(self.img_size[0]/2 + 110)))
                elif i == 1:
                    index_action[action].append((120, int(self.img_size[0]/2 + 110)))

            if not success:
                logger.info("Get face landmark localization failed! Please check your image!")
                continue

            image_points = self.get_image_points_from_landmark(face_landmark)
            success, rotation_vector, translation_vector, camera_matrix, dist_coeffs, reprojectdst = self.caculate_pose_vector(image_points)
            
            if not success:
                logger.info("Get rotation and translation vectors failed!")
                continue

            # 画出投影正方体
            alpha=0.3
            if not hasattr(self, 'before'):
                self.before = reprojectdst
            else:
                reprojectdst = alpha * self.before + (1-alpha)* reprojectdst
            reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
            for start, end in self.line_pairs:
                cv2.line(img, reprojectdst[start], reprojectdst[end], (0, 0, 255))

            # 计算头部欧拉角
            pitch, yaw, roll = self.caculate_euler_angle(rotation_vector, translation_vector)
            cv2.putText(img, "pitch: " + "{:7.2f}".format(pitch), (20, int(self.img_size[0]/2 -10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), thickness=2)
            cv2.putText(img, "yaw: " + "{:7.2f}".format(yaw), (20, int(self.img_size[0]/2 + 30) ), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), thickness=2)
            cv2.putText(img, "roll: " + "{:7.2f}".format(roll), (20, int(self.img_size[0]/2 +70)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), thickness=2)
            for index, action in enumerate(index_action):
                cv2.putText(img, "{}".format(self._index_action[action]), index_action[action][1], 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 50), thickness=2)
            frames_euler.append([index, img, pitch, yaw, roll])

            # 转换成摄像头可显示的格式
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            final_action = None
            if len(frames_euler) > self.frame_window_size:
                # 比较当前头部动作欧拉角与过去的欧拉角，只有动作幅度幅度超过阈值，则判定发生相应的动作
                # picth值用来判断点头动作
                # yaw值用来判断摇头动作
                current = [pitch, yaw, roll]
                tmp = [abs(pitch), abs(yaw)]
                max_index = tmp.index(max(tmp))
                max_probability_action = index_action[max_index][0]
                for start_idx, start_img, p, y, r in frames_euler[0:int(self.frame_window_size/2)]:
                    start = [p, y, r]
                    if poses & max_probability_action and abs(start[max_index]-current[max_index]) >= self.pose_threshold[max_index]:
                        frames_euler = []
                        final_action = max_index
                        self.action_time = time.time()
                        yield {self._index_action[max_index]: [(start_idx, start_img), (index, img)]}
                        break
                else:
                    # 丢弃过时的视频帧
                    frames_euler.pop(0)
            # 动作判定发生则高亮显示0.5s
            if self.action_time !=0  and time.time() - self.action_time < 0.5:
                cv2.putText(img_rgb, "{}".format(self._index_action[max_index]), index_action[max_index][1], 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
            
            # 本地显示预测视频框，AIStudio项目不支持显示视频框
            # cv2.imshow('Pose Estimation', img_rgb)
            # 写入预测结果
            video_writer.write(img_rgb)

class FaceGameController(object):
    def __init__(self, debug):
        self.debug = debug
        self.module = hub.Module(name="face_landmark_localization")
        self.capture = cv2.VideoCapture(0) 
        self.actiondim = 8
        self.actionset = ["left","right","up","down","mopen","mheri","lefteye","righteye"]

    def get_face_landmark(self, image, use_gpu=False):
        try:
            res = self.module.keypoint_detection(images=[image], use_gpu=use_gpu)
            return True, res[0]['data'][0]
        except Exception as e:
            logger.error("Get face landmark localization failed! Exception: %s " % e)
            return False, None

    def judgeaction(self, face_landmark):
        action = np.zeros([8], dtype=np.int32)
        left = face_landmark[33][0] * 3 - face_landmark[2][0] - face_landmark[3][0] - face_landmark[4][0]
        right = face_landmark[14][0] + face_landmark[13][0] + face_landmark[12][0] - face_landmark[33][0] * 3
        if left < right * 0.6:
            action[0] = 1
        if right < left * 0.6:
            action[1] = 2

        up = face_landmark[29][1] - face_landmark[27][1]
        baseline = (face_landmark[27][1] * 2 - face_landmark[21][1] - face_landmark[22][1])  / 2
        if up < baseline:
            action[2] = 4

        if face_landmark[4][1] < face_landmark[50][1] or face_landmark[12][1] < face_landmark[52][1]:
            action[3] = 8

        if face_landmark[66][1] - face_landmark[62][1] > 1.5 * (face_landmark[62][1] - face_landmark[51][1]):
            action[4] = 16
        return action
        # return np.sum(action)

    def judgeleftoright(self, face_landmark):
        left = face_landmark[33][0] * 3 - face_landmark[2][0] - face_landmark[3][0] - face_landmark[4][0]
        right = face_landmark[14][0] + face_landmark[13][0] + face_landmark[12][0] - face_landmark[33][0] * 3
        if left < right * 0.6:
            return 1
        if right < left * 0.6:
            return 2
        return 0

    def judgeup(self, face_landmark):
        up = face_landmark[29][1] - face_landmark[27][1]
        baseline = (face_landmark[27][1] * 2 - face_landmark[21][1] - face_landmark[22][1])  / 2

        if up < baseline:
            return 4
        return 0


    def caltwo(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def judgedown(self, face_landmark):
        if face_landmark[4][1] < face_landmark[49][1] or face_landmark[12][1] < face_landmark[53][1]:
            return 8
        return 0 
        

    def control(self, use_gpu=False):
        ret, frame_rgb = self.capture.read()
        frame_rgb = cv2.flip(frame_rgb,1)
        success, face_landmark = self.get_face_landmark(image=frame_rgb, use_gpu=use_gpu)
        # print(len(result))
        if not success:
            if self.debug:
                cv2.imshow("Debug", frame_rgb)
                cv2.waitKey(1)  
            # return    
        else:
            tmp_img = frame_rgb.copy()
            for _, point in enumerate(face_landmark):
	            cv2.circle(tmp_img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
            
            action = self.judgeaction(face_landmark)
            for ind, i in enumerate(action):
                if i != 0:
                    print(self.actionset[ind])        
            
            cv2.imshow("Debug", tmp_img)
            cv2.waitKey(1)

        # cv2.destroyAllWindows()
