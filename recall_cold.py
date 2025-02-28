# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import pickle
import configparser
import collections
import logging
from tqdm import tqdm
from datetime import datetime
from utils import DataUtils

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
METRIC_RECALL = config.getboolean("PROCESS", "METRIC_RECALL")

# 冷启动召回参数
SIM_ITEM_TOPK = config.getint("COLD", "SIM_ITEM_TOPK")
RECALL_ITEM_NUM = config.getint("COLD", "RECALL_ITEM_NUM")
TOPK = config.getint("COLD", "TOPK")

# 读取数据
logging.info("Loading data...")
all_click_df = pd.read_csv(os.path.join(SAVE_PATH, f"train_click_{MODE}_{CURRENT_DATE}.csv"))
item_info_df = pd.read_csv(os.path.join(SAVE_PATH, f"articles_info_{CURRENT_DATE}.csv"))
item_type_dict, item_words_dict, item_created_time_dict = DataUtils.get_item_info_dict(item_info_df)
if METRIC_RECALL:
    trn_hist_click_df, trn_last_click_df = DataUtils.get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = DataUtils.get_user_item_time(trn_hist_click_df)
i2i_sim = pickle.load(open(os.path.join(SAVE_PATH, f'emb_i2i_sim_{MODE}_{CURRENT_DATE}.pkl'), 'rb'))

item_topk_click = DataUtils.get_item_topk_click(trn_hist_click_df, k=TOPK)

def item_based_recommend(user_id, user_item_time_dict, sim_matrix, sim_item_topk, recall_item_num, item_topk_click, item_created_time_dict, method):
    user_hist_items = sorted(user_item_time_dict[user_id], key=lambda x: x[1], reverse=True)[:2]
    user_hist_items_ = {item_id for item_id, _ in user_hist_items}

    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        if i not in sim_matrix:
            continue

        for j, wij in sorted(sim_matrix[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue

            created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict.get(i, 0) - item_created_time_dict.get(j, 0)))
            loc_weight = (0.9 ** (len(user_hist_items) - loc))
            content_weight = 1.0

            item_rank.setdefault(j, 0)
            item_rank[j] += created_time_weight * loc_weight * content_weight * wij

    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item not in item_rank:
                item_rank[item] = -i - 100
                if len(item_rank) == recall_item_num:
                    break

    return sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, SIM_ITEM_TOPK, 
                                                        RECALL_ITEM_NUM, item_topk_click, item_created_time_dict, method='embedding')

pickle.dump(user_recall_items_dict, open(os.path.join(SAVE_PATH, f'cold_start_items_raw_dict_{MODE}_{CURRENT_DATE}.pkl'), 'wb'))

def get_click_article_ids_set(all_click_df):
    return set(all_click_df.click_article_id.values)

def cold_start_items(user_recall_items_dict, user_hist_item_typs_dict, user_hist_item_words_dict,
                     user_last_item_created_time_dict, item_type_dict, item_words_dict,
                     item_created_time_dict, click_article_ids_set, recall_item_num):
    cold_start_user_items_dict = {}
    for user, item_list in tqdm(user_recall_items_dict.items()):
        cold_start_user_items_dict.setdefault(user, [])
        for item, score in item_list:
            hist_item_type_set = user_hist_item_typs_dict[user]
            hist_mean_words = user_hist_item_words_dict[user]
            hist_last_item_created_time = user_last_item_created_time_dict[user]
            hist_last_item_created_time = datetime.fromtimestamp(hist_last_item_created_time)

            curr_item_type = item_type_dict[item]
            curr_item_words = item_words_dict[item]
            curr_item_created_time = item_created_time_dict[item]
            curr_item_created_time = datetime.fromtimestamp(curr_item_created_time)

            if curr_item_type not in hist_item_type_set or \
               item in click_article_ids_set or \
               abs(curr_item_words - hist_mean_words) > 200 or \
               abs((curr_item_created_time - hist_last_item_created_time).days) > 90:
                continue

            cold_start_user_items_dict[user].append((item, score))

    cold_start_user_items_dict = {k: sorted(v, key=lambda x:x[1], reverse=True)[:recall_item_num] \
                                  for k, v in cold_start_user_items_dict.items()}

    pickle.dump(cold_start_user_items_dict, open(os.path.join(SAVE_PATH, f'cold_start_user_items_dict_{MODE}_{CURRENT_DATE}.pkl'), 'wb'))

    return cold_start_user_items_dict

user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict = DataUtils.get_user_hist_item_info_dict(all_click_df)
click_article_ids_set = get_click_article_ids_set(all_click_df)
cold_start_user_items_dict = cold_start_items(user_recall_items_dict, user_hist_item_typs_dict, user_hist_item_words_dict,
                                              user_last_item_created_time_dict, item_type_dict, item_words_dict,
                                              item_created_time_dict, click_article_ids_set, RECALL_ITEM_NUM)

if METRIC_RECALL:
    DataUtils.metrics_hit_mrr(cold_start_user_items_dict, trn_last_click_df, topk=5)
