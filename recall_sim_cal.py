# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import pickle
import configparser
import math
import collections
from tqdm import tqdm
from datetime import datetime
from utils import DataUtils
from collections import defaultdict
import faiss
import logging

# 日志设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 获取当前日期
CURRENT_DATE = datetime.now().strftime("%Y%m%d")

# 读取配置文件
config = configparser.ConfigParser()
config.read("config/config.ini")

# 获取参数
DATA_PATH = os.getenv("DATA_PATH", config["PROCESS"]["DATA_PATH"])
SAVE_PATH = os.getenv("SAVE_PATH", config["PROCESS"]["SAVE_PATH"])
MODE = os.getenv("MODE", config["PROCESS"]["MODE"])

# 基于物品的协同过滤相似度计算
def itemcf_sim(df, item_created_time_dict):
    logging.info("Starting ItemCF similarity calculation...")
    try:
        user_item_time_dict = DataUtils.get_user_item_time(df)
        i2i_sim = {}
        item_cnt = defaultdict(int)

        for user, item_time_list in tqdm(user_item_time_dict.items()):
            for loc1, (i, i_click_time) in enumerate(item_time_list):
                item_cnt[i] += 1
                i2i_sim.setdefault(i, {})
                for loc2, (j, j_click_time) in enumerate(item_time_list):
                    if i == j:
                        continue

                    loc_alpha = 1.0 if loc2 > loc1 else 0.7
                    loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                    click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                    created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))

                    i2i_sim[i].setdefault(j, 0)
                    i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(len(item_time_list) + 1)

        for i, related_items in i2i_sim.items():
            for j, wij in related_items.items():
                i2i_sim[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

        pickle.dump(i2i_sim, open(os.path.join(SAVE_PATH, f'itemcf_i2i_sim_{MODE}_{CURRENT_DATE}.pkl'), 'wb'))
        logging.info("ItemCF similarity calculation completed successfully.")
        return i2i_sim

    except Exception as e:
        logging.error(f"Error in itemcf_sim: {e}")
        return {}

# 基于Embedding的相似度计算
def embdding_sim(click_df, item_emb_df, save_path, topk=10):
    logging.info("Starting embedding similarity calculation...")
    try:
        item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))
        item_emb_cols = [col for col in item_emb_df.columns if 'emb' in col]
        item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
        item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

        item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
        item_index.add(item_emb_np)
        sim, idx = item_index.search(item_emb_np, topk)

        item_sim_dict = collections.defaultdict(dict)
        for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
            target_raw_id = item_idx_2_rawid_dict[target_idx]
            for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
                if rele_idx == -1:
                    continue
                rele_raw_id = item_idx_2_rawid_dict[rele_idx]
                item_sim_dict[target_raw_id][rele_raw_id] += sim_value

        pickle.dump(item_sim_dict, open(os.path.join(save_path, f'emb_i2i_sim_{MODE}_{CURRENT_DATE}.pkl'), 'wb'))
        logging.info("Embedding similarity calculation completed successfully.")
        return item_sim_dict

    except Exception as e:
        logging.error(f"Error in embdding_sim: {e}")
        return {}

if __name__ == "__main__":
    try:
        logging.info("Starting similarity matrix calculation...")
        all_click_df = pd.read_csv(os.path.join(SAVE_PATH, f"train_click_{MODE}_{CURRENT_DATE}.csv"))
        item_info_df = pd.read_csv(os.path.join(SAVE_PATH, f"articles_info_{CURRENT_DATE}.csv"))
        item_emb_df = pd.read_csv(os.path.join(DATA_PATH, 'articles_emb.csv'))

        item_type_dict, item_words_dict, item_created_time_dict = DataUtils.get_item_info_dict(item_info_df)

        i2i_sim = itemcf_sim(all_click_df, item_created_time_dict)
        emb_i2i_sim = embdding_sim(all_click_df, item_emb_df, SAVE_PATH, topk=10)

        logging.info("Similarity matrix calculation completed successfully.")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
