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


def should_predict_landing(scene_info):
    """
    判斷是否應該進行落點預測 (範例：球的 y 座標低於畫面一半時預測)

    Args:
        scene_info: 當前遊戲畫面資訊 (scene_info)

    Returns:
        bool: True 表示應該進行預測，False 表示不應該
    """
    ball_y = scene_info["ball"][1]  # 取得球的 y 座標
    screen_height = 500  # 遊戲畫面高度

    # 判斷條件：當球的 y 座標低於畫面高度的一半時，就應該預測
    if ball_y > screen_height / 2:  # 你可以調整這個閾值 (畫面高度的一半)
        return True
    else:
        return False


class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        Constructor
        """
        print(ai_name)
        self.data_buffer = []  # 用於儲存遊戲資料
        self.previous_ball_position = None  # 記錄上一幀球的位置
        self.model = self.load_model()  # 嘗試載入訓練好的模型
        if self.model:
            print("機器學習模型載入成功！")
        else:
            print("模型載入失敗，將使用預設策略 (預測落點演算法)。") #  [ 修改：更明確的提示訊息 ]

    def update(self, scene_info, *args, **kwargs):
        """
        Generate the command according to the received `scene_info`.
        """
        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            if scene_info["status"] == "GAME_PASS":
                self.save_data_to_pickle() # 遊戲通關時儲存資料 (可選)
            return "RESET"

        if not scene_info["ball_served"]:
            command = "SERVE_TO_LEFT"  # 自動發球
            predicted_x = None  # 發球時不預測落點
        else:
            #  [ 平台指令預測策略 - 優先使用機器學習模型，模型載入失敗時退回預測落點演算法 ]
            if self.model:  #  [ 優先策略：使用機器學習模型 ]
                #  [ 提取特徵 -  **加入球的速度 (ball_dx, ball_dy)** ]
                ball_x = scene_info["ball"][0]
                ball_y = scene_info["ball"][1]
                platform_x = scene_info["platform"][0]

                ball_dx = 0  #  預設速度為 0
                ball_dy = 0
                if self.previous_ball_position:  #  如果不是第一幀
                    ball_dx = ball_x - self.previous_ball_position[0]  #  計算 x 軸速度
                    ball_dy = ball_y - self.previous_ball_position[1]  #  計算 y 軸速度


                feature = np.array([
                    ball_x,
                    ball_y,
                    platform_x,
                    ball_dx,      #  加入球的 x 軸速度作為特徵
                    ball_dy       #  加入球的 y 軸速度作為特徵
                ]).reshape(1, -1)  #  提取特徵，注意特徵順序要和訓練時一致
                predicted_label = self.model.predict(feature)[0]  # 模型預測標籤 (0, 1, 或 2)

                if predicted_label == 0:
                    command = "MOVE_LEFT"
                elif predicted_label == 1:
                    command = "MOVE_RIGHT"
                elif predicted_label == 2:
                    command = "NONE"
                else:
                    command = "NONE"  # 預防模型預測出非預期的標籤
                predicted_x = predict_landing_point(scene_info, self.previous_ball_position) # 仍然計算預測落點 (僅用於記錄)

            else:  #  [ 備用策略：預測落點演算法 - 模型載入失敗時啟用 ]  [ 修改：使用預測落點演算法作為備用策略 ]
                if should_predict_landing(scene_info):  # 判斷是否需要預測
                    predicted_x = predict_landing_point(scene_info, self.previous_ball_position) # 預測落點
                    if predicted_x is not None:
                        if predicted_x < scene_info["platform"][0] + 20:
                            command = "MOVE_LEFT"
                        elif predicted_x > scene_info["platform"][0] + 20:
                            command = "MOVE_RIGHT"
                        else:
                            command = "NONE"
                    else:
                        command = "NONE"
                else:
                    command = "NONE" #  (如果使用 should_predict_landing 函式，球在高處時不移動)
                    predicted_x = None # (如果使用 should_predict_landing 函式，球在高處時不預測)
                    command = "NONE" #  [ 備用策略 - 平台預設不移動，等待球下墜再根據預測落點移動 ]  [ 修改：平台預設不移動 ]


        if scene_info["ball_served"]:
            self.data_buffer.append({
                "scene_info": scene_info,
                "command": command,
                "predicted_x": predicted_x  # 記錄預測落點 (方便觀察)
            })

        self.previous_ball_position = scene_info["ball"]  # 更新上一幀球的位置
        return command

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False
        self.data_buffer = []  # 清空資料 buffer
        self.previous_ball_position = None  # 重置上一幀球的位置

    def save_data_to_pickle(self):
        """
        儲存遊戲資料到 pickle 檔案 
        """
        filename = "arkanoid_data.pickle"  # 預設資料檔名 (可自訂)
        try:
            with open(filename, "wb") as f:
                pickle.dump(self.data_buffer, f)
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def load_model(self, filename="arkanoid_model.pickle"):
        """
        載入訓練好的機器學習模型
        """
        try:
            with open(filename, "rb") as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            print(f"模型檔案 '{filename}' 未找到，將使用預設策略 (預測落點演算法)。") #  [ 修改：更明確的提示訊息 ]
            return None
        except Exception as e:
            print(f"模型載入失敗: {e}")
            print(f"將使用預設策略 (預測落點演算法)。") #  [ 修改：更明確的提示訊息 ]
            return None