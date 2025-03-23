import pickle
import numpy as np  # 導入 numpy


def predict_landing_point(scene_info, previous_ball_position):
    """
    預測球的落點 x 座標，考慮反彈與鏡像座標 (使用數學公式 - 新版本)

    Args:
        scene_info: 當前遊戲畫面資訊 (scene_info)
        previous_ball_position: 上一幀球的位置 (ball 座標 tuple)

    Returns:
        predicted_x: 預測的落點 x 座標 (整數)
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
        self.previous_ball_position = None #  記錄上一幀球的位置
        self.model = self.load_model() # [ 嘗試載入模型 ]
        if self.model:
            print("機器學習模型載入成功！")
        else:
            print("模型載入失敗，將使用預設策略 (預測落點演算法)。")


    def update(self, scene_info, *args, **kwargs):
        """
        Generate the command according to the received `scene_info`.
        """
        command = "NONE"  # [ 初始化 command 變數，避免 UnboundLocalError ]
        predicted_x = None  # [ 初始化 predicted_x 變數，程式碼更清晰 ]

        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            # 遊戲結束時，返回 "RESET" (模型測試版本 **不再儲存資料**)
            return "RESET"

        if not scene_info["ball_served"]:
            command = "SERVE_TO_LEFT"  # 自動發球
            predicted_x = None #  發球時不預測落點
        else:
            # [ 平台指令預測邏輯 - **模型預測 或 預測落點演算法 (預設策略)** ]
            if self.model: # [ 如果模型載入成功，使用 **模型預測** 指令 ]
                # [ 提取特徵 - **與 ml_model_trainer.py 的 preprocess_data 函式保持一致** ]
                ball_x = scene_info["ball"][0]
                ball_y = scene_info["ball"][1]
                platform_x = scene_info["platform"][0]

                ball_dx = 0  # 預設速度為 0
                ball_dy = 0
                if self.previous_ball_position: # 如果不是第一幀
                    ball_dx = ball_x - self.previous_ball_position[0] # 計算 x 軸速度
                    ball_dy = ball_y - self.previous_ball_position[1] # 計算 y 軸速度

                feature = np.array([ball_x, ball_y, platform_x, ball_dx, ball_dy]).reshape(1, -1) #  [ 特徵順序 **必須與訓練時一致** ]
                predicted_label = self.model.predict(feature)[0] # [ 模型預測標籤 (0, 1, 或 2) ]

                if predicted_label == 0: # [ 標籤 0: MOVE_LEFT ]
                    command = "MOVE_LEFT"
                elif predicted_label == 1: # [ 標籤 1: MOVE_RIGHT ]
                    command = "MOVE_RIGHT"
                elif predicted_label == 2: # [ 標籤 2: NONE ]
                    command = "NONE"
                else: # [ 預防模型預測出非預期的標籤 ]
                    command = "NONE"
                predicted_x = predict_landing_point(scene_info, self.previous_ball_position) # [ 仍然計算預測落點 (僅用於記錄) ]
                print(f"[Model Predict] Ball: {scene_info['ball']}, Platform: {scene_info['platform']}, Predicted_x: {predicted_x}, Command: {command}, Label: {predicted_label}") # [ 模型預測時的除錯訊息 ]


            else: # [ 模型載入失敗，回退到 **預測落點演算法** 作為預設策略 ]
                predicted_x = predict_landing_point(scene_info, self.previous_ball_position) # [ 預測落點 ]
                if predicted_x is not None: # [ 成功預測到落點 ]
                    if predicted_x < scene_info["platform"][0] + 20 : # [ 預測落點在平台左半邊 ]
                        command = "MOVE_LEFT"
                    elif predicted_x > scene_info["platform"][0] + 20: # [ 預測落點在平台右半邊 ]
                        command = "MOVE_RIGHT"
                    else: # [ 預測落點在平台中間 ]
                        command = "NONE"
                else: # [ 無法預測落點 (例如球向上移動)，暫時不移動平台 ]
                    command = "NONE"
                print(f"[Fallback Algo] Ball: {scene_info['ball']}, Platform: {scene_info['platform']}, Predicted_x: {predicted_x}, Command: {command}") # [ 預設演算法的除錯訊息 ]


        self.previous_ball_position = scene_info["ball"] # 更新上一幀球的位置
        return command

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False
        self.previous_ball_position = None #  重置上一幀球的位置


    def load_model(self, filename="arkanoid_model.pickle"):
        """
        載入訓練好的機器學習模型
        """
        try:
            with open(filename, "rb") as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            print(f"模型檔案 '{filename}' 未找到，將使用預設策略 (預測落點演算法)。") # [ 修改：更明確的提示訊息 ]
            return None
        except Exception as e:
            print(f"模型載入失敗: {e}")
            print(f"將使用預設策略 (預測落點演算法)。") # [ 新增：更明確的提示訊息 ]
            return None