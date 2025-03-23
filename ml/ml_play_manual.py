"""
The template of the main script of the machine learning process (Manual Control with Data Collection)
"""
import pygame
import pickle  # 導入 pickle 模組

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
        command = "NONE"  #  [ **修正： 在函式一開始 **初始化** `command` 變數，預設值為 "NONE" ]
        # Make the caller to invoke `reset()` for the next round.
        if keyboard is None:
            keyboard = []
        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            # 遊戲結束時，儲存資料並返回 "RESET"
            if scene_info["status"] == "GAME_PASS":  # 可選：只記錄成功通關的資料
                self.save_data_to_pickle()  # 儲存資料
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

        # **[ 數據蒐集程式碼 ]**
        if scene_info["ball_served"]:  # 可選：只在發球後才開始記錄資料
            self.data_buffer.append({
                "scene_info": scene_info,
                "command": command
            })

        return command

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False
        self.data_buffer = []  # 清空資料 buffer

    def save_data_to_pickle(self):
        """
        將資料儲存到 pickle 檔案
        """
        filename = "manual_arkanoid_data.pickle"  #  手動操作資料的檔名 (可自訂)
        try:
            with open(filename, "wb") as f:
                pickle.dump(self.data_buffer, f)
            print(f"Manual data saved to {filename}")
        except Exception as e:
            print(f"Error saving manual data: {e}")