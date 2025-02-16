# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import configparser
from utils import DataUtils

# 读取配置文件
config = configparser.ConfigParser()
config.read("config/config.ini")  # 确保路径正确

# 获取参数（如果环境变量存在，则优先使用环境变量）
DATA_PATH = os.getenv("DATA_PATH", config["PROCESS"]["DATA_PATH"])
SAVE_PATH = os.getenv("SAVE_PATH", config["PROCESS"]["SAVE_PATH"])
MODE = os.getenv("MODE", config["PROCESS"]["MODE"])  # "debug" / "offline" / "online"
# METRIC_RECALL = os.getenv("METRIC_RECALL", config["DEFAULT"]["METRIC_RECALL"]).lower() == "true"

# 确保输出目录存在
os.makedirs(SAVE_PATH, exist_ok=True)

# 获取当前日期
CURRENT_DATE = datetime.now().strftime("%Y%m%d")


# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 归一化函数
max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

def get_all_click_sample(data_path, sample_nums=50000):
    """ 从训练数据中采样部分数据用于调试 """
    try:
        all_click = pd.read_csv(os.path.join(data_path, "train_click_log.csv"))
        sample_user_ids = np.random.choice(all_click.user_id.unique(), size=sample_nums, replace=False)
        all_click = all_click[all_click["user_id"].isin(sample_user_ids)]
        all_click = all_click.drop_duplicates(["user_id", "click_article_id", "click_timestamp"])
        logging.info(f"Sampled {sample_nums} users from training data.")
        return all_click
    except Exception as e:
        logging.error(f"Error in get_all_click_sample: {e}")
        return None

def get_all_click_df(data_path, offline=True):
    """ 读取完整点击日志，offline 模式不包含测试集 """
    try:
        if offline:
            all_click = pd.read_csv(os.path.join(data_path, "train_click_log.csv"))
        else:
            trn_click = pd.read_csv(os.path.join(data_path, "train_click_log.csv"))
            tst_click = pd.read_csv(os.path.join(data_path, "testA_click_log.csv"))
            all_click = pd.concat([trn_click, tst_click], ignore_index=True)
        
        all_click = all_click.drop_duplicates(["user_id", "click_article_id", "click_timestamp"])
        logging.info(f"Loaded click data: {len(all_click)} rows")
        return all_click
    except Exception as e:
        logging.error(f"Error in get_all_click_df: {e}")
        return None

def get_item_info_df(data_path):
    """ 读取文章的基本信息 """
    try:
        item_info_df = pd.read_csv(os.path.join(data_path, "articles.csv"))
        item_info_df.rename(columns={"article_id": "click_article_id"}, inplace=True)
        logging.info(f"Loaded item info: {len(item_info_df)} articles")
        return item_info_df
    except Exception as e:
        logging.error(f"Error in get_item_info_df: {e}")
        return None

def get_item_emb_dict(data_path):
    """ 读取文章 Embedding 并归一化 """
    try:
        item_emb_df = pd.read_csv(os.path.join(data_path, "articles_emb.csv"))
        item_emb_cols = [col for col in item_emb_df.columns if "emb" in col]
        item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])
        item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

        item_emb_dict = dict(zip(item_emb_df["article_id"], item_emb_np))
        pickle.dump(item_emb_dict, open(os.path.join(SAVE_PATH, f"item_content_emb_{CURRENT_DATE}.pkl"), "wb"))
        logging.info("Saved article embeddings")
        return item_emb_dict
    except Exception as e:
        logging.error(f"Error in get_item_emb_dict: {e}")
        return None

def process_and_save():
    """ 处理数据并保存 """
    try:
        if MODE == "debug":
            all_click_df = get_all_click_sample(DATA_PATH)
            file_name = f"train_click_debug_{CURRENT_DATE}.csv"
        elif MODE == "offline":
            all_click_df = get_all_click_df(DATA_PATH, offline=True)
            file_name = f"train_click_offline_{CURRENT_DATE}.csv"
        else:
            all_click_df = get_all_click_df(DATA_PATH, offline=False)
            file_name = f"train_click_online_{CURRENT_DATE}.csv"

        if all_click_df is not None:
            all_click_df["click_timestamp"] = all_click_df[["click_timestamp"]].apply(max_min_scaler)
            all_click_df.to_csv(os.path.join(SAVE_PATH, file_name), index=False)
            logging.info(f"Saved click data: {file_name}")

        item_info_df = get_item_info_df(DATA_PATH)
        if item_info_df is not None:
            item_info_df.to_csv(os.path.join(SAVE_PATH, f"articles_info_{CURRENT_DATE}.csv"), index=False)
            logging.info("Saved article info")

        get_item_emb_dict(DATA_PATH)
        
         # 提取历史点击数据和最后一次点击数据（仅 offline/debug 模式）
        if MODE != "online":
            trn_hist_click_df, trn_last_click_df = DataUtils.get_hist_and_last_click(all_click_df)
            trn_hist_click_df.to_csv(os.path.join(SAVE_PATH, f"train_hist_click_{MODE}_{CURRENT_DATE}.csv"), index=False)
            trn_last_click_df.to_csv(os.path.join(SAVE_PATH, f"train_last_click_{MODE}_{CURRENT_DATE}.csv"), index=False)
            logging.info("Saved historical and last click data.")

        logging.info("Data preprocessing completed successfully.")
    except Exception as e:
        logging.error(f"Error in process_and_save: {e}")

# 运行数据处理
if __name__ == "__main__":
    process_and_save()
