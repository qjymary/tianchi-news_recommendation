import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc, os
import time
from datetime import datetime
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


data_path = './data_new/'
save_path = './tmp_results/'
offline = False


# 重新读取数据的时候，发现click_article_id是一个浮点数，所以将其转换成int类型
trn_user_item_feats_df = pd.read_csv(data_path + 'trn_user_item_feats_df_online_20250206.csv')
trn_user_item_feats_df['click_article_id'] = trn_user_item_feats_df['click_article_id'].astype(int)

if offline:
    val_user_item_feats_df = pd.read_csv(data_path + 'val_user_item_feats_df_online_20250206.csv')
    val_user_item_feats_df['click_article_id'] = val_user_item_feats_df['click_article_id'].astype(int)
else:
    val_user_item_feats_df = None
    
tst_user_item_feats_df = pd.read_csv(data_path + 'tst_user_item_feats_df_online_20250206.csv')
tst_user_item_feats_df['click_article_id'] = tst_user_item_feats_df['click_article_id'].astype(int)

# 做特征的时候为了方便，给测试集也打上了一个无效的标签，这里直接删掉就行
del tst_user_item_feats_df['label']


# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)
    print(user_num)
    for k in range(5, topk+1, 5):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            # 获取前k个召回的结果
            # tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            tmp_recall_items = [x for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1
        
        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)

tst_last_click_df = pd.read_csv(data_path + 'new_test_click_last_clicks.csv')

# 评估召回结果
user_recall_items_dict = tst_user_item_feats_df.groupby('user_id')['click_article_id'].apply(lambda x : list(x)).reset_index()
user_recall_items_dict = dict(zip(user_recall_items_dict['user_id'], user_recall_items_dict['click_article_id']))
metrics_recall(user_recall_items_dict, tst_last_click_df, topk=50)

def submit(recall_df, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')
    
    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    assert tmp.min() >= topk
    
    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()
    
    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2', 
                                                  3: 'article_3', 4: 'article_4', 5: 'article_5'})
    
    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    submit.to_csv(save_name, index=False, header=True)


def metrics_rank(recall_df, topk=5, tst_last_click_df=tst_last_click_df, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')
    
    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    assert tmp.min() >= topk
    
    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()
    
    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2', 
                                                  3: 'article_3', 4: 'article_4', 5: 'article_5'})
    tst_last_click_df_dict = dict(zip(tst_last_click_df['user_id'], tst_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)
    print(user_num)
    click_num = 0
    score = 0
    for index, row in submit.iterrows():
        print(tst_last_click_df_dict[row['user_id']])
        click_id = tst_last_click_df_dict[row['user_id']]
        if row['article_1'] == click_id:
            score += 1
            click_num += 1
        elif row['article_2'] == click_id:
            score += 1/2
            click_num += 1
        elif row['article_3'] == click_id:
            score += 1/3
            click_num += 1
        elif row['article_4'] == click_id:
            score += 1/4
            click_num += 1
        elif row['article_5'] == click_id:
            score += 1/5
            click_num += 1
    print('click_num', click_num, 'click_score', click_num/user_num, 'score', score/user_num, 'user_num', user_num)
    


# 排序结果归一化
def norm_sim(sim_df, weight=0.0):
    # print(sim_df.head())
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))

    sim_df = sim_df.apply(lambda sim: sim + weight)  # plus one
    return sim_df


# 防止中间出错之后重新读取数据
trn_user_item_feats_df_rank_model = trn_user_item_feats_df.copy()

if offline:
    val_user_item_feats_df_rank_model = val_user_item_feats_df.copy()
    
tst_user_item_feats_df_rank_model = tst_user_item_feats_df.copy()


# 定义特征列
lgb_cols = ['sim0', 'time_diff0', 'word_diff0','sim_max', 'sim_min', 'sim_sum', 
            'sim_mean', 'score','click_size', 'time_diff_mean', 'active_level',
            'click_environment','click_deviceGroup', 'click_os', 'click_country', 
            'click_region','click_referrer_type', 'user_time_hob1', 'user_time_hob2',
            'words_hbo', 'category_id', 'created_at_ts','words_count']

# 排序模型分组
trn_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)
g_train = trn_user_item_feats_df_rank_model.groupby(['user_id'], as_index=False).count()["label"].values

if offline:
    val_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)
    g_val = val_user_item_feats_df_rank_model.groupby(['user_id'], as_index=False).count()["label"].values

# 模型及参数的定义
lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                            max_depth=-1, n_estimators=500, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                            learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs= 16, verbose=10)  


# 模型训练
if offline:
    lgb_Classfication.fit(trn_user_item_feats_df_rank_model[lgb_cols], trn_user_item_feats_df_rank_model['label'],
                    eval_set=[(val_user_item_feats_df_rank_model[lgb_cols], val_user_item_feats_df_rank_model['label'])], 
                    eval_metric=['auc', ])
else:
    lgb_Classfication.fit(trn_user_item_feats_df_rank_model[lgb_cols], trn_user_item_feats_df_rank_model['label'])


# 模型预测
tst_user_item_feats_df['pred_score'] = lgb_Classfication.predict_proba(tst_user_item_feats_df[lgb_cols])[:,1]

# 将这里的排序结果保存一份，用户后面的模型融合
tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']].to_csv(save_path + 'lgb_cls_score.csv', index=False)


# 预测结果重新排序, 及生成提交结果
rank_results = tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']]
rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
submit(rank_results, topk=5, model_name='lgb_cls')
# 评估命中结果
metrics_rank(rank_results, topk=5, tst_last_click_df=tst_last_click_df, model_name='lgb_cls')

# 五折交叉验证，这里的五折交叉是以用户为目标进行五折划分
#  这一部分与前面的单独训练和验证是分开的
def get_kfold_users(trn_df, n=5):
    user_ids = trn_df['user_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set

k_fold = 5
trn_df = trn_user_item_feats_df_rank_model
user_set = get_kfold_users(trn_df, n=k_fold)

score_list = []
score_df = trn_df[['user_id', 'click_article_id', 'label']]
# sub_preds = np.zeros(tst_user_item_feats_df_rank_model.shape[0])

# 五折交叉验证，并将中间结果保存用于staking
for n_fold, valid_user in enumerate(user_set):
    train_idx = trn_df[~trn_df['user_id'].isin(valid_user)] # add slide user
    valid_idx = trn_df[trn_df['user_id'].isin(valid_user)]
    
    # 模型及参数的定义
    lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                            max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                            learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs= 16, verbose=10)  
    # 训练模型
    lgb_Classfication.fit(train_idx[lgb_cols], train_idx['label'],eval_set=[(valid_idx[lgb_cols], valid_idx['label'])], 
                          eval_metric=['auc', ])
    
    # 预测验证集结果
    valid_idx['pred_score'] = lgb_Classfication.predict_proba(valid_idx[lgb_cols], 
                                                              num_iteration=lgb_Classfication.best_iteration_)[:,1]
    
    # 对输出结果进行归一化 分类模型输出的值本身就是一个概率值不需要进行归一化
    # valid_idx['pred_score'] = valid_idx[['pred_score']].transform(lambda x: norm_sim(x))
    
    valid_idx.sort_values(by=['user_id', 'pred_score'])
    valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')
    
    # 将验证集的预测结果放到一个列表中，后面进行拼接
    score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])
    
    # 如果是线上测试，需要计算每次交叉验证的结果相加，最后求平均
    if not offline:
        sub_preds += lgb_Classfication.predict_proba(tst_user_item_feats_df_rank_model[lgb_cols], 
                                                     num_iteration=lgb_Classfication.best_iteration_)[:,1]
    
score_df_ = pd.concat(score_list, axis=0)
score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])
# 保存训练集交叉验证产生的新特征
score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(save_path + 'trn_lgb_cls_feats.csv', index=False)
    
# 测试集的预测结果，多次交叉验证求平均,将预测的score和对应的rank特征保存，可以用于后面的staking，这里还可以构造其他更多的特征
tst_user_item_feats_df_rank_model['pred_score'] = sub_preds / k_fold
tst_user_item_feats_df_rank_model['pred_score'] = tst_user_item_feats_df_rank_model['pred_score'].transform(lambda x: norm_sim(x))
tst_user_item_feats_df_rank_model.sort_values(by=['user_id', 'pred_score'])
tst_user_item_feats_df_rank_model['pred_rank'] = tst_user_item_feats_df_rank_model.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

# 保存测试集交叉验证的新特征
tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(save_path + 'tst_lgb_cls_feats.csv', index=False)


# 预测结果重新排序, 及生成提交结果
rank_results = tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score']]
rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
submit(rank_results, topk=5, model_name='lgb_cls')
# 评估命中结果
metrics_rank(rank_results, topk=5, tst_last_click_df=tst_last_click_df, model_name='lgb_cls')
