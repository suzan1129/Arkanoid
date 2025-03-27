"""
Script to train a machine learning model for Arkanoid game.
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import glob
import argparse

# from sklearn.ensemble import RandomForestClassifier  # 隨機森林

def preprocess_data(data):
    print("進入 preprocess_data 函式")
    features = []
    labels = []
    previous_ball_position = None

    for item in data:
        scene_info = item["scene_info"]
        command = item["command"]
        predicted_x = item.get("predicted_x", None)  # 獲取 predicted_x，如果不存在則為 None

        if scene_info and command:
            ball_x = scene_info["ball"][0]
            ball_y = scene_info["ball"][1]
            platform_x = scene_info["platform"][0]

            ball_dx = 0
            ball_dy = 0
            if previous_ball_position:
                ball_dx = ball_x - previous_ball_position[0]
                ball_dy = ball_y - previous_ball_position[1]

            # 特徵包含 6 個值：ball_x, ball_y, platform_x, ball_dx, ball_dy, predicted_x
            feature = [ball_x, ball_y, platform_x, ball_dx, ball_dy]

            # 如果 predicted_x 存在，加入特徵
            if predicted_x is not None:
                feature.append(predicted_x)
            else:
                feature.append(platform_x)  # 如果 predicted_x 為 None，使用平台當前位置

            if command == "MOVE_LEFT":
                label = 0
            elif command == "MOVE_RIGHT":
                label = 1
            elif command == "NONE":
                label = 2
            else:
                continue

            features.append(feature)
            labels.append(label)

        previous_ball_position = scene_info["ball"]

    print("preprocess_data 函式執行完成，返回特徵和標籤")
    return np.array(features), np.array(labels)

def train_model(features, labels):
    print("進入 train_model 函式")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    print(f"len of features: {len(features)}, len of labels: {len(labels)}")
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    print(f"Train set: {len(train_features)}, Test set: {len(test_features)}")


    model = KNeighborsClassifier(n_neighbors=20)  # 使用 KNN 演算法
    model.fit(train_features, train_labels)

    test_accuracy = accuracy_score(test_labels, model.predict(test_features))
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("train_model 函式執行完成，返回模型")
    print(model)
    return model

def save_model(model, filename="arkanoid_model.pickle"):
    print("進入 save_model 函式")
    try:
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print(f"模型儲存成功: {filename}")
    except Exception as e:
        print(f"模型儲存失敗: {e}")

    print("save_model 函式執行完成")

def load_data_from_pickle(folder_path):
    print("進入 load_data_from_pickle 函式")
    all_data = []
    file_pattern = os.path.join(folder_path, "*.pickle")
    filenames = glob.glob(file_pattern)

    if not filenames:
        print(f"Error: No pickle files found in folder '{folder_path}'.")
        print("load_data_from_pickle 函式執行完成 (No files found)，返回 None")
        return None

    print(f"Found {len(filenames)} pickle files in folder '{folder_path}'.")

    for filename in filenames:
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
            print(f"Data loaded from {filename}")
            all_data.extend(data)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error loading data from {filename}: {e}")

    print(f"Total data loaded: {len(all_data)} items")
    print("load_data_from_pickle 函式執行完成，返回資料")
    return all_data

def main():
    print("進入 main 函式")

    parser = argparse.ArgumentParser(description="Train Arkanoid ML model.")
    parser.add_argument("--data_folder", type=str, default=None,
                        help="Path to the folder containing game data pickle files.")
    args = parser.parse_args()

    data_folder_input = args.data_folder

    if args.data_folder:
        data_folders = [args.data_folder]
        print(f"Using data folder specified in command line: {args.data_folder}")
    elif data_folder_input == "all":
        data_folders = ["manual_arkanoid_data_collection", "arkanoid_data_collection"]
        print(f"Using data folder specified in command line: Loading data from ALL folders (manual and auto).")
    else:
        data_folders = ["manual_arkanoid_data_collection", "arkanoid_data_collection"]
        print(f"Using default data folders: {data_folders}")

    all_game_data = []

    if data_folders:
        for folder_path in data_folders:
            game_data = load_data_from_pickle(folder_path)
            if game_data:
                all_game_data.extend(game_data)
    else:
        print("Error: No data folder specified or default data folders not set.")
        print("main 函式提前結束")
        return

    if not all_game_data:
        print("Error: No game data loaded from any folder. main 函式提前結束")
        return

    print(f"Total game data loaded from all folders: {len(all_game_data)} items")
    features, labels = preprocess_data(all_game_data)
    if features.size == 0:
        print("features 為空，main 函式提前結束")
        print("Error: No features extracted from data. Please check your data and preprocessing function.")
        return

    model = train_model(features, labels)

    if model:
        save_model(model)


    print("main 函式執行完成")

if __name__ == '__main__':
    print("程式開始執行 (if __name__ == '__main__':)")
    main()
    print("程式結束執行 (if __name__ == '__main__':)")