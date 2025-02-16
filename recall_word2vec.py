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
from gensim.models import Word2Vec
from annoy import AnnoyIndex

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

# Word2Vec 参数
VECTOR_SIZE = config.getint("WORD2VEC", "VECTOR_SIZE")
WINDOW = config.getint("WORD2VEC", "WINDOW")
MIN_COUNT = config.getint("WORD2VEC", "MIN_COUNT")
WORKERS = config.getint("WORD2VEC", "WORKERS")
N_TREES = config.getint("WORD2VEC", "N_TREES")

# 读取数据
logging.info("Loading data...")
all_click_df = pd.read_csv(os.path.join(SAVE_PATH, f"train_click_{MODE}_{CURRENT_DATE}.csv"))
if METRIC_RECALL:
    trn_hist_click_df, trn_last_click_df = DataUtils.get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = DataUtils.get_user_item_time(trn_hist_click_df)

# 训练 Word2Vec 模型
def train_word2vec(click_df, f1, f2, save_path):
    sentences = click_df.groupby(f1)[f2].agg(list).tolist()
    model_path = os.path.join(save_path, f'w2v_model_{MODE}_{CURRENT_DATE}.pkl')
    
    if os.path.exists(model_path):
        model = Word2Vec.load(model_path)
    else:
        model = Word2Vec(sentences=sentences, vector_size=VECTOR_SIZE, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, sg=1, seed=42)
        model.save(model_path)
    
    return {int(word): model.wv[word] for word in model.wv.index_to_key}

# 构建 Annoy 索引
def build_annoy_index(article_vec_map):
    annoy_index = AnnoyIndex(VECTOR_SIZE, 'angular')
    for article_id, vector in article_vec_map.items():
        annoy_index.add_item(article_id, vector)
    annoy_index.build(N_TREES)
    return annoy_index

# 训练 Word2Vec 并构建索引
logging.info("Training Word2Vec model...")
article_vec_map = train_word2vec(all_click_df, 'user_id', 'click_article_id', SAVE_PATH)
logging.info("Building Annoy index...")
annoy_index = build_annoy_index(article_vec_map)

# 获取点击最高的文章
logging.info("Getting top clicked items...")
item_topk_click = DataUtils.get_item_topk_click(trn_hist_click_df, k=CLICK_TOPK)

# 读取文章信息
item_info_df = pd.read_csv(os.path.join(SAVE_PATH, f"articles_info_{CURRENT_DATE}.csv"))
item_type_dict, item_words_dict, item_created_time_dict = DataUtils.get_item_info_dict(item_info_df)

# Word2Vec 召回
def word2vec_recall(user_id, user_item_time_dict, annoy_index, article_vec_map, recall_item_num, item_topk_click):
    rank = collections.defaultdict(float)
    if user_id not in user_item_time_dict or not user_item_time_dict[user_id]:
        return []

    last_clicked_item = sorted(user_item_time_dict[user_id], key=lambda x: x[1], reverse=True)[0][0]
    if last_clicked_item not in article_vec_map:
        return []

    item_vector = article_vec_map[last_clicked_item]
    similar_items = annoy_index.get_nns_by_vector(item_vector, recall_item_num, include_distances=True)

    for related_item, distance in zip(similar_items[0], similar_items[1]):
        rank[related_item] += 1 / (1 + distance)

    if len(rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item not in rank:
                rank[item] = -i - 100
                if len(rank) == recall_item_num:
                    break

    return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

# 执行 Word2Vec 召回
logging.info("Starting Word2Vec recall...")
user_recall_items_dict = {}
for user_id in tqdm(all_click_df['user_id'].unique()):
    user_recall_items_dict[user_id] = word2vec_recall(user_id, user_item_time_dict, annoy_index, article_vec_map, RECALL_ITEM_NUM, item_topk_click)

pickle.dump(user_recall_items_dict, open(os.path.join(SAVE_PATH, f'w2v_recall_{MODE}_{CURRENT_DATE}.pkl'), 'wb'))

if METRIC_RECALL:
    logging.info("Evaluating recall performance...")
    DataUtils.metrics_hit_mrr(user_recall_items_dict, trn_last_click_df, topk=RECALL_ITEM_NUM)

logging.info("Word2Vec recall process completed successfully.")
