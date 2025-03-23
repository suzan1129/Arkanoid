import pickle

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
        self.data_buffer = []  # 用來暫存蒐集到的資料
        self.previous_ball_position = None #  記錄上一幀球的位置
        self.previous_x_direction = 0 #  記錄上一幀球的水平移動方向 (移除 wall_bounce_count 相關程式碼)

    def update(self, scene_info, *args, **kwargs):
        """
        Generate the command according to the received `scene_info`.
        """
        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            # 遊戲結束時，儲存資料並返回 "RESET"
            if scene_info["status"] == "GAME_PASS": # 只記錄成功通關的資料 (可選)
                self.save_data_to_pickle() # 儲存資料
            return "RESET"

        if not scene_info["ball_served"]:
            command = "SERVE_TO_LEFT"  # 自動發球
            predicted_x = None #  發球時不預測落點
        else:
            #  更新牆壁反彈次數 (基於球的 x 座標變化方向判斷) (移除 wall_bounce_count 相關程式碼)


            #  使用預測落點策略控制平台 (傳入 wall_bounce_count) (移除 wall_bounce_count 參數)
            predicted_x = predict_landing_point(scene_info, self.previous_ball_position)


            if predicted_x is not None: #  成功預測到落點
                if predicted_x < scene_info["platform"][0] + 20 : # 預測落點在平台左半邊
                    command = "MOVE_LEFT"
                elif predicted_x > scene_info["platform"][0] + 20: # 預測落點在平台右半邊
                    command = "MOVE_RIGHT"
                else: # 預測落點在平台中間
                    command = "NONE"
            else: # 無法預測落點 (例如球向上移動)，暫時不移動平台
                command = "NONE"

        # **[ 數據蒐集程式碼 ]** (保持不變，移除 wall_bounce_count 記錄)
        if scene_info["ball_served"]: #  只在發球後才開始記錄資料 (可選)
            self.data_buffer.append({
                "scene_info": scene_info,
                "command": command,
                "predicted_x": predicted_x #  記錄預測的落點 x 座標 (方便分析)
            })

        self.previous_ball_position = scene_info["ball"] # 更新上一幀球的位置
        return command

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False
        self.data_buffer = [] # 清空資料 buffer
        self.previous_ball_position = None #  重置上一幀球的位置
        self.previous_x_direction = 0 # 重置球的水平移動方向 (移除 wall_bounce_count 相關程式碼)


    def save_data_to_pickle(self):
        """
        將資料儲存到 pickle 檔案
        """
        filename = "arkanoid_data.pickle" #  你可以自訂檔名
        try:
            with open(filename, "wb") as f:
                pickle.dump(self.data_buffer, f)
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")