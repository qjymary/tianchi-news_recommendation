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
SIM_ITEM_TOPK = config.getint("ITEMCF", "SIM_ITEM_TOPK")
RECALL_ITEM_NUM = config.getint("ITEMCF", "RECALL_ITEM_NUM")
CLICK_TOPK = config.getint("ITEMCF", "CLICK_TOPK")

# 读取数据
logging.info("Loading data...")
all_click_df = pd.read_csv(os.path.join(SAVE_PATH, f"train_click_{MODE}_{CURRENT_DATE}.csv"))
if METRIC_RECALL:
    trn_hist_click_df, trn_last_click_df = DataUtils.get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = DataUtils.get_user_item_time(trn_hist_click_df)

# 加载相似性矩阵
logging.info("Loading similarity matrices...")
i2i_sim = pickle.load(open(os.path.join(SAVE_PATH, f'itemcf_i2i_sim_{MODE}_{CURRENT_DATE}.pkl'), 'rb'))
emb_i2i_sim = pickle.load(open(os.path.join(SAVE_PATH, f'emb_i2i_sim_{MODE}_{CURRENT_DATE}.pkl'), 'rb'))

# 获取点击最高的文章
logging.info("Getting top clicked items...")
item_topk_click = DataUtils.get_item_topk_click(trn_hist_click_df, k=CLICK_TOPK)

# 读取文章信息
item_info_df = pd.read_csv(os.path.join(SAVE_PATH, f"articles_info_{CURRENT_DATE}.csv"))
item_type_dict, item_words_dict, item_created_time_dict = DataUtils.get_item_info_dict(item_info_df)

# 统一召回计算函数
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
            
            if method == 'embedding':
                content_weight += emb_i2i_sim.get(i, {}).get(j, 0) + emb_i2i_sim.get(j, {}).get(i, 0)

            item_rank.setdefault(j, 0)
            item_rank[j] += created_time_weight * loc_weight * content_weight * wij

    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item not in item_rank:
                item_rank[item] = -i - 100
                if len(item_rank) == recall_item_num:
                    break

    return sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

logging.info("Starting recall process...")
user_multi_recall_dict = {}

# ItemCF 召回
user_recall_items_dict = {}
for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, SIM_ITEM_TOPK, RECALL_ITEM_NUM, item_topk_click, item_created_time_dict, method='itemcf')

user_multi_recall_dict['itemcf'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['itemcf'], open(os.path.join(SAVE_PATH, f'itemcf2_recall_dict_{MODE}_{CURRENT_DATE}.pkl'), 'wb'))

# Embedding 召回
user_recall_items_dict = {}
for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, emb_i2i_sim, SIM_ITEM_TOPK, RECALL_ITEM_NUM, item_topk_click, item_created_time_dict, method='embedding')

user_multi_recall_dict['embedding'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['embedding'], open(os.path.join(SAVE_PATH, f'embedding_sim_item_recall_{MODE}_{CURRENT_DATE}.pkl'), 'wb'))

if METRIC_RECALL:
    logging.info("Evaluating recall performance...")
    DataUtils.metrics_hit_mrr(user_multi_recall_dict['itemcf'], trn_last_click_df, topk=RECALL_ITEM_NUM)
    DataUtils.metrics_hit_mrr(user_multi_recall_dict['embedding'], trn_last_click_df, topk=RECALL_ITEM_NUM)

logging.info("Recall process completed successfully.")
