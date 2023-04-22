import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # 引入余弦相似度计算公式
import json
from tqdm import *

# 读取文件内容
addition_1 = np.load('./model/addition_1.npy')
addition_2 = np.load('./model/addition_2.npy')
addition_3 = np.load('./model/addition_3.npy')
addition_4 = np.load('./model/addition_4.npy')
addition_5 = np.load('./model/addition_5.npy')
addition_6 = np.load('./model/addition_6.npy')
base = np.load('./model/base.npy')
tran_1 = np.load('./model/tran_1.npy')
tran_2 = np.load('./model/tran_2.npy')
tran_3 = np.load('./model/tran_3.npy')
tran_4 = np.load('./model/tran_4.npy')
tran_5 = np.load('./model/tran_5.npy')
tran_6 = np.load('./model/tran_6.npy')
with open('./model/index2word.json', 'r') as f:
    index2word = json.load(f)

final_model = dict()
final_model['base'] = base

final_model['tran'] = dict()
final_model['tran']['1'] = tran_1
final_model['tran']['2'] = tran_2
final_model['tran']['3'] = tran_3
final_model['tran']['4'] = tran_4
final_model['tran']['5'] = tran_5
final_model['tran']['6'] = tran_6
final_model['addition'] = dict()
final_model['addition']['1'] = addition_1
final_model['addition']['2'] = addition_2
final_model['addition']['3'] = addition_3
final_model['addition']['4'] = addition_4
final_model['addition']['5'] = addition_5
final_model['addition']['6'] = addition_6

final_model['index2word'] = index2word

# 计算得到最终的嵌入向量
print("Calculating Final Embedding")
embeddings = np.zeros(base.shape)
indexs = [x for x in range(len(index2word))]
weight = {'1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1}  # 每个网络的权重系数
for index in range(base.shape[0]):
    fake_index = indexs[index]
    v_list = []
    for layer_id in final_model['addition']:
        v = base[index] + weight[layer_id] *\
            np.dot(final_model['addition'][layer_id]
                   [index], final_model['tran'][layer_id])
        v_list.append(v)
    embeddings[fake_index] = np.mean(v_list, axis=0)
print("Embedding Completed")
print("Calculating Similarity Matrix")
Similarity = np.zeros([embeddings.shape[0], embeddings.shape[0]])  # 相似度矩阵
for i in tqdm(range(embeddings.shape[0])):
    for j in range(embeddings.shape[0]):
        if i != j:
            sim_AB = cosine_similarity(embeddings[i, :].reshape(
                1, -1), embeddings[j, :].reshape(1, -1))  # 计算矩阵向量间的相似度
            Similarity[i][j] = sim_AB
np.save("Similarity.npy", Similarity)
print("Similarity Matrix Completed!")
print("Network Propagation!")
def Random_Walkwith_Restart(Similarity, P0, N_max_iter=100, r_restart=0.1, Eps_min_change=1e-6):
    # !!!必须让Similarity的每一列相加都为1，后面的网络传播算法才是有效的
    normal_Similarity = Similarity/Similarity.sum(axis=1)
    N_max_iter = 100
    r_restart = 0.1
    P0 = np.array(list(gene_inipr.values()))
    Wadj = normal_Similarity
    Eps_min_change = 1e-6
    P0 = P0.reshape(len(index2word), -1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))  # 将数据归一到0到1，可以根据数据特点归一到-1到1
    P0 = scaler.fit_transform(P0)  # 归一化
    Pt = P0
    count = 0  # count用来监视循环次数
    for i in range(N_max_iter):
        count += 1
        Pt1 = (1 - r_restart)*(np.dot(Wadj, Pt)) + r_restart*P0
        print(sum(abs(Pt1-Pt)))
        if all(sum(abs(Pt1-Pt)) < Eps_min_change):
            break
        Pt = Pt1
    Score = {}  # 构建基因——分数字典
    for i in range(len(index2word)):
        Score[index2word[i]] = Pt[i][0]
    Score_sort = sorted(
        Score.items(), key=lambda x: x[1], reverse=True)  # 依据重要性排序
    return dict(Score_sort)

# 构造P0，也就是随机游走的初始值
with open("./model/index2word.json", 'r') as file:  
    index2word = json.load(file)
file.close()

with open("./allscore/all_score_GSE1456.json", 'r') as file:  
    gene_score = json.load(file)
file.close()

gene_inipr = {}  # 从初始打分中筛选出相应的基因
i = 0
for index in index2word:
    if index not in gene_score.keys():  
        score = 0
    else:
        score = gene_score[index]
    gene_inipr[index] = score
    i += 1
P0 = np.array(list(gene_inipr.values()))
print("Network Propagation!")
Sort_gene = Random_Walkwith_Restart(Similarity, P0)
print("Done!")
with open("./Biomarker/biomarker.json", "w") as f:
    f.write(json.dumps(Sort_gene, ensure_ascii=False,
            indent=4, separators=(',', ':')))
print("Biomarker Saved!")
