"""
The template of the main script of the manual machine learning process
"""
import pygame
import pickle
import datetime
import os

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        Constructor
        """
        self.ball_served = False
        self.data_buffer = []  # 初始化資料 buffer

    def update(self, scene_info, keyboard=None, *args, **kwargs):
        """
        Generate the command according to the received `scene_info`.
        """
        if keyboard is None:
            keyboard = []
        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            if scene_info["status"] == "GAME_PASS":  # 只記錄成功通關的資料
                folder_name = "manual_arkanoid_data_collection"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                filename = os.path.join(folder_name, f"manual_arkanoid_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pickle")
                self.save_data_to_pickle(filename)
            return "RESET"

        if pygame.K_q in keyboard:
            command = "SERVE_TO_LEFT"
            self.ball_served = True
        elif pygame.K_e in keyboard:
            command = "SERVE_TO_RIGHT"
            self.ball_served = True
        elif pygame.K_LEFT in keyboard or pygame.K_a in keyboard:
            command = "MOVE_LEFT"
        elif pygame.K_RIGHT in keyboard or pygame.K_d in keyboard:
            command = "MOVE_RIGHT"
        else:
            command = "NONE"

        # 只在發球後才開始記錄資料
        if scene_info["ball_served"]:  # 移除無法序列化的資料
            scene_info.pop("bricks")
            scene_info.pop("hard_bricks")
            scene_info.pop("frame")
            scene_info.pop("ball_served")
            self.data_buffer.append({
                "scene_info": scene_info,
                "command": command,
                "predicted_x": -1  # 手動模式下沒有預測落點，設為 -1
            })

        return command

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False
        self.data_buffer = []  # 清空資料 buffer

    def save_data_to_pickle(self, filename):
        """
        將資料儲存到 pickle 檔案
        """
        removed_status_databuffer = []
        for data in self.data_buffer:
            data_copy = data.copy()
            data_copy["scene_info"].pop("status") 
            removed_status_databuffer.append(data_copy)
        self.data_buffer = removed_status_databuffer
        print(self.data_buffer)
        try:
            with open(filename, "wb") as f:
                pickle.dump(self.data_buffer, f)
            print(f"Manual data saved to {filename}")
        except Exception as e:
            print(f"Error saving manual data: {e}")