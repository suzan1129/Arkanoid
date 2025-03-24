import pickle
import numpy as np

def predict_landing_point(scene_info, previous_ball_position):
    """
    預測球的落點 x 座標，考慮反彈與鏡像座標 (使用數學公式 - 新版本)
    """
    ball_x, ball_y = scene_info["ball"]
    platform_y = 400  # 平台 y 座標
    screen_width = 200  # 遊戲畫面寬度

    if not previous_ball_position:
        return None  # 無法預測

    prev_x, prev_y = previous_ball_position
    ball_dx = ball_x - prev_x  # x 方向變化
    ball_dy = ball_y - prev_y  # y 方向變化

    if ball_dy <= 0:
        return None  # 球向上移動時不預測落點

    if ball_dx == 0:
        return ball_x  # 垂直下落，落點不變

    # 計算斜率
    slope = ball_dy / ball_dx

    # 計算與平台交點的 x 座標
    distance_y = platform_y - ball_y
    predicted_x = ball_x + (distance_y / slope)

    # 計算反彈 (使用數學方式取代 while 迴圈)
    bounce_count = int(predicted_x // screen_width)  # 計算反彈次數
    if bounce_count % 2 == 1:
        predicted_x = screen_width - abs(predicted_x % screen_width)
    else:
        predicted_x = abs(predicted_x % screen_width)

    # 確保座標合法
    return max(0, min(int(predicted_x), screen_width))


class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        Constructor
        """
        print(ai_name)
        self.previous_ball_position = None  # 記錄上一幀球的位置
        self.model = self.load_model()  # 嘗試載入模型
        if self.model:
            print("機器學習模型載入成功！")
        else:
            print("模型載入失敗，將使用預設策略 (預測落點演算法)。")

    def update(self, scene_info, *args, **kwargs):
        """
        Generate the command according to the received `scene_info`.
        """
        command = "NONE"
        predicted_x = None

        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            return "RESET"

        if not scene_info["ball_served"]:
            command = "SERVE_TO_LEFT"  # 自動發球
        else:
            ball_x = scene_info["ball"][0]
            ball_y = scene_info["ball"][1]
            platform_x = scene_info["platform"][0]

            ball_dx = 0
            ball_dy = 0
            if self.previous_ball_position:
                ball_dx = ball_x - self.previous_ball_position[0]
                ball_dy = ball_y - self.previous_ball_position[1]

            # 計算 predicted_x
            predicted_x = predict_landing_point(scene_info, self.previous_ball_position)

            if predicted_x is None:  # 如果無法預測落點
                predicted_x = 69  # 隨便填一個數字
                # if ball_dy > 0:  # 球向下移動
                #     if ball_x > platform_x + 20:  # 球在平台右邊
                #         command = "MOVE_RIGHT"
                #     elif ball_x < platform_x + 20:  # 球在平台左邊
                #         command = "MOVE_LEFT"
                #     else:  # 球在平台中間
                #         command = "NONE"
                # else:  # 球向上移動，不需要急著移動
                #     command = "NONE"
            # else:
            if self.model:  # 如果模型載入成功，使用模型預測
                feature = np.array([ball_x, ball_y, platform_x, ball_dx, ball_dy, predicted_x]).reshape(1, -1)
                predicted_label = self.model.predict(feature)[0]

                if predicted_label == 0:
                    command = "MOVE_LEFT"
                elif predicted_label == 1:
                    command = "MOVE_RIGHT"
                elif predicted_label == 2:
                    command = "NONE"
                else:
                    command = "NONE"
            else:  # 模型載入失敗，回退到預測落點演算法
                # if predicted_x < scene_info["platform"][0] + 20:
                #     command = "MOVE_LEFT"
                # elif predicted_x > scene_info["platform"][0] + 20:
                #     command = "MOVE_RIGHT"
                # else:
                #     command = "NONE"
                print('模型載入失敗')

        self.previous_ball_position = scene_info["ball"]
        return command

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False
        self.previous_ball_position = None  # 重置上一幀球的位置

    def load_model(self, filename="arkanoid_model.pickle"):
        """
        載入訓練好的機器學習模型
        """
        try:
            with open(filename, "rb") as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            print(f"模型檔案 '{filename}' 未找到，將使用預設策略 (預測落點演算法)。")  # [ 修改：更明確的提示訊息 ]
            return None
        except Exception as e:
            print(f"模型載入失敗: {e}")
            print(f"將使用預設策略 (預測落點演算法)。")  # [ 新增：更明確的提示訊息 ]
            return None