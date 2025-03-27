import pickle
import datetime
import os
import random

def predict_landing_point(scene_info, previous_ball_position):
    """
    預測球的落點 x 座標，考慮反彈與鏡像座標 (使用數學公式)
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


    """這邊等等看看要不要刪掉♥"""
    # 加入隨機偏移量 (例如，左右偏移 -10 到 +10 像素)
    offset = random.randint(-10, 10)  
    predicted_x += offset

    # 確保座標合法
    return max(0, min(int(predicted_x), screen_width))

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        Constructor
        """
        print(ai_name)
        self.data_buffer = []  # 用來暫存蒐集到的資料
        self.previous_ball_position = None  # 記錄上一幀球的位置


    def update(self, scene_info, *args, **kwargs):
        """
        Generate the command according to the received `scene_info`.
        """
        command = "NONE"
        predicted_x = None

        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            if scene_info["status"] == "GAME_PASS":  # 只記錄成功通關的資料
                folder_name = "arkanoid_data_collection"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                filename = os.path.join(folder_name, f"arkanoid_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pickle")
                self.save_data_to_pickle(filename)
            return "RESET"

        if not scene_info["ball_served"]:
            # 隨機選擇發球方向
            command = "SERVE_TO_LEFT" if random.randint(0, 1) == 0 else "SERVE_TO_RIGHT"           
        else:
            ball_x = scene_info["ball"][0]
            ball_y = scene_info["ball"][1]
            platform_x = scene_info["platform"][0]

            ball_dx = 0
            ball_dy = 0
            if self.previous_ball_position:
                ball_dx = ball_x - self.previous_ball_position[0]
                ball_dy = ball_y - self.previous_ball_position[1]

            predicted_x = predict_landing_point(scene_info, self.previous_ball_position)


            if predicted_x is None:  # 如果無法預測落點
                # predicted_x = platform_x  # 將 predicted_x 設為平台當前位置
                predicted_x = 100   # 設為畫面中央
            # else:
            if predicted_x < scene_info["platform"][0] + 20: 
                command = "MOVE_LEFT"
            elif predicted_x > scene_info["platform"][0] + 20:
                command = "MOVE_RIGHT"
            else:
                command = "NONE"

        if scene_info["ball_served"]: # 只在發球後才開始記錄資料
            scene_info.pop("bricks")    # 移除無法序列化的資料
            scene_info.pop("hard_bricks")
            scene_info.pop("frame")
            scene_info.pop("ball_served")  
            # pop完後只保存scene_info的platform & ball位置
            self.data_buffer.append({
                "scene_info": scene_info,  
                "command": command,
                "predicted_x": predicted_x  # 保留預測落點
            })

        self.previous_ball_position = scene_info["ball"]
        return command

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False
        self.data_buffer = []  # 清空資料 buffer
        self.previous_ball_position = None  # 重置上一幀球的位置

    def save_data_to_pickle(self, filename):
        """
        將資料儲存到 pickle 檔案
        """
        removed_status_databuffer = []
        for data in self.data_buffer:
            data_copy = data.copy()
            data_copy["scene_info"].pop("status")
            print("data_copy", data_copy)
            removed_status_databuffer.append(data_copy)
        self.data_buffer = removed_status_databuffer


        try:
            with open(filename, "wb") as f:
                pickle.dump(self.data_buffer, f)
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")