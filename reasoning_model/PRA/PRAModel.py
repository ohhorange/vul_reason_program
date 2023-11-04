from dataclasses_json import dataclass_json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score,recall_score
import pickle
from collections import defaultdict
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,precision_recall_curve
import matplotlib.pyplot as plt
# import torch
import warnings
from functools import lru_cache
warnings.filterwarnings("ignore")

def load_pickle_file(pickle_file):
    with open(pickle_file,'rb') as f:
        return pickle.load(f)

class PRAReasonModel:
    '''PRA 对比实验'''
    def __init__(self) -> None:
        '''使用相同的数据集'''
        self.kg_config_folder = 'pra_data/'
        self.kg_reasoning_folder = self.kg_config_folder + 'reasoning/' # 图谱三元组文件 这是图谱的基本数据
        self.split = [0.7,0.3]
        # assert sum(self.split)==1
        self.train_file = self.kg_reasoning_folder + 'pra_train_data.pickle' #_nosim
        '''加载训练和测试数据集'''
        train_test_data = load_pickle_file(self.train_file)
        split_point = int(len(train_test_data[0])*self.split[0])
        self.train_ent_pair_indices = train_test_data[0][:1800]
        self.test_ent_pair_indices = train_test_data[0][1800:]
        self.train_key_paths_indices = train_test_data[1][:1800]
        self.test_key_paths_indices = train_test_data[1][1800:]
        self.train_rel_val_indices = train_test_data[2][:1800]
        self.test_rel_val_indices = train_test_data[2][1800:]
        # 推理所需的实体邻居dict 其中不包括待推理关系形成的邻居
        self.ent_neighbors_file = self.kg_reasoning_folder + 'ent_neighbor_dict.pickle'
        self.ent_neighbors_dict = load_pickle_file(self.ent_neighbors_file) #相比于在三元组中逐个查找 用dict能提高效率

        self.pra_output_folder = self.kg_reasoning_folder + 'pra_results/'
        self.train_pra_path_features_file = self.pra_output_folder + 'train_path_features.txt'
        self.test_pra_path_features_file = self.pra_output_folder + 'test_path_features.txt'
        self.model_file = self.pra_output_folder + 'model.pkl'
        self.result_file = self.pra_output_folder + 'result.txt'

        self.kg_key_path2id = self.key_paths_assemble()
        self.train_data=defaultdict(list)
        self.test_data=defaultdict(list)

        self.feature_file = self.pra_output_folder + 'path_features.txt'
        self.features = []
        self.labels = []
        self.coef = []
        self.test_result=[]
        self.path_num = 0
        # self.data_preprocess()
        self.path_ids = [n for n in range(29)]

    def key_paths_assemble(self):
        '''
        路径汇总 
        汇总所有关键路径
        查看测试集和训练集是否所有的关键路径是否存在不同 
        '''
        kg_train_key_paths_set = set()
        kg_test_key_paths_set = set()
        for ent_pair_key_paths in self.train_key_paths_indices:
            for key_path in ent_pair_key_paths:
                kg_train_key_paths_set.add(key_path)
        for ent_pair_key_paths in self.test_key_paths_indices:
            for key_path in ent_pair_key_paths:
                kg_test_key_paths_set.add(key_path)
        # print(kg_test_key_paths_set)
        '''因为就一种关系 关键路径有限 大概率一样 不一样也暂且不知道怎么办'''
        # assert kg_train_key_paths_set == kg_test_key_paths_set 

        kg_train_key_paths_list = list(kg_train_key_paths_set)

        kg_key_path2id = {}
        i = 0
        for key_path in kg_train_key_paths_list:
            kg_key_path2id[key_path] = i
            i += 1
        print('对训练数据中的路径汇总得到所有的关键路径有',kg_key_path2id)
        return kg_key_path2id

    def data_preprocess(self,feature_file):
        features = []
        labels = []
        with open(feature_file,"r") as f:
            datas = f.readlines() # len train 2811
            for data in datas:
                data = eval(data.strip().split("\t")[1]) # 有些是int，有些是float
                for n,d in enumerate(data):
                    data[n] = float(d)
                labels.append(data[0])
                features.append(data[1:])
        
        return features,labels

    def train(self, stop_loss=0, max_iter=100000):
        # self.model = LogisticRegression(
        #                     C=10000,
        #                     random_state=0,
        #                     penalty="none",
        #                     class_weight="balanced",
        #                     solver="sag",
        #                     tol=stop_loss,
        #                     max_iter=max_iter,
        #                     dual=False,
        #                     verbose=1
        #                     ) # class_weight="balanced", solver="liblinear", random_state=0,
        # self.model = RandomForestClassifier()
        # self.model = SVC(C=10,kernel='rbf',probability=True) # sigmoid rbf linear poly
        self.model = KNeighborsClassifier(n_neighbors=2)
        X_train,y_train = self.data_preprocess(self.train_pra_path_features_file)
        # print('train',len(X_train),y_train)
        X_test,y_test = self.data_preprocess(self.test_pra_path_features_file)
        # print(X_test,y_test)
        # print('test',len(X_test),y_test)
        self.model.fit(X_train, y_train)

        y_predicted = self.model.predict(X_train)

        print(classification_report(y_train,y_predicted,digits=4))
        y_predict = self.model.predict(X_test)
        print(confusion_matrix(y_test,y_predict))
        print(classification_report(y_test,y_predict,digits=4))
        with open('result.txt','w') as f:
            f.write(str(confusion_matrix(y_test,y_predict))+'\n')
            f.write(str(classification_report(y_test,y_predict,digits=4)))
        y_predict = self.model.predict_proba(X_test) # 0  中的是为0的概率 1 中的是正例的概率
        # print(confusion_matrix(y_test,y_predict))
        # print(y_test)
        # print(y_predict)
        precision, recall, thresholds = precision_recall_curve(y_test, y_predict[:,1]) #
        with open(self.pra_output_folder+'PRA_PR_record.txt','w') as f:
            for i in range(len(recall)):
                f.write('{}\t{}\n'.format(recall[i],precision[i]))
        plt.plot(recall,precision)
        plt.savefig(self.pra_output_folder+'PRA_pr.png')


        # self.model.predict_proba([X_test[0]])
        # self.test_result = precision_recall_fscore_support(y_test, self.model.predict(X_test))
        # self.coef = self.model.coef_[0]
        return

    # def train_fc(self):
    #     optimizer = torch.optim.Adam(self.model_path_gat.parameters(),lr = 0.001,weight_decay=0.0000001)



    def save(self):
        with open(self.model_file,"wb") as f:
            pickle.dump(self.model,f)

        with open(self.result_file,"w") as f:
            for result in self.test_result:
                f.write(str(result)+"\n")
            f.write("\n\n\n")
            for n,c in enumerate(self.coef):
                f.write(str(self.path_ids[n])+"\t"+str(c)+"\n")
        return

    def path_selection(self,threshold=0.001):

        tem = []
        for n, c in enumerate(self.coef):
            if abs(c) > threshold:
                tem.append(self.path_ids[n])
            else:
                continue
        self.path_ids = tem
        del tem
        return

    def retrain(self,stop_loss=0.0001, max_iter=100000):
        for m,feature in enumerate(self.features):
            tem = []
            for n,v in enumerate(feature):
                if n in self.path_ids:
                    tem.append(v)
            self.features[m] = tem
            del tem
        self.train(stop_loss=stop_loss, max_iter=max_iter)

    # @lru_cache(maxsize=None)
    from collections import defaultdict

    def _prob(self, begin, end, relation_path): # 采取后向截断的动态规划
        '''已更新'''
        '''计算从begin 随机游走经由relation_path能到达end 的概率'''
        prob = 0
        length = len(relation_path)
        cur_rel = relation_path[0]
        available_next_count = 0
        for (rel, ent) in self.ent_neighbors_dict[begin]:
            if rel == cur_rel:
                available_next_count += 1
        if length == 1:
            if (relation_path[0],end) in self.ent_neighbors_dict[begin]:
                prob = 1/available_next_count
            else:
                prob = 0
            return prob
        else:
            if begin not in self.ent_neighbors_dict:
                return 0
            else:
                for (relation,entity) in self.ent_neighbors_dict[begin]: 
                    if relation == relation_path[0]:
                        prob += (1/available_next_count) * self._prob(entity, end, relation_path[1:]) # 这句代码下载时有错 注意
                return prob # 返回准确概率
    
    def get_probs(self,feature_data,ent_pairs,rel_vals,save_file,prob_flag="pcrw-exact",walker_num=50): # 完全按照随机游走的公式来计算路径概率
        
        for i in range(len(ent_pairs)):
            (node_1,node_2) = ent_pairs[i]
            flag = rel_vals[i]
            feature_data[(node_1,node_2)].append(flag)
            for path in self.kg_key_path2id.keys():
                if prob_flag == "pcrw-exact":
                    tem = self._prob(node_1,node_2,path)
                feature_data[(node_1, node_2)].append(tem)
            i+=1
            print("第{}个结束\n".format(i))
        print('写入',save_file)
        with open(save_file,"a") as f:
            for key in feature_data:
                f.write(str(key)+"\t"+str(feature_data[key])+"\n")
        return

class Walker:
    def __init__(self,name,begin):
        self.name = name
        self.walk_history = [begin]
        self.state = "walking"

    def onestep_walk(self,subnodes):
        if subnodes==[]:
            self.state = "stop"
            # print("walker %s stopped!"%self.name)
            return
        else:
            n = len(subnodes) # 
            m = np.random.randint(n,size=1)[0]
            self.walk_history.append(subnodes[m])
        return

if __name__ == "__main__":
    '''
    0:实体对 (ent1,ent2)
    1:实体对之间的路径[[(path1),(path2)],[(path1),(path2),...]]
    2:是否存在关系 flag[0,1,0,...]1表示存在,0表示不存在
    '''
    '''先生成路径的feature文件'''
    # pra = PRAReasonModel()
    # # pra.get_probs(pra.train_data,pra.train_ent_pair_indices,pra.train_rel_val_indices,pra.train_pra_path_features_file)
    # pra.get_probs(pra.test_data,pra.test_ent_pair_indices,pra.test_rel_val_indices,pra.test_pra_path_features_file)
    # del pra


    '''生成测试集'''


    '''训练'''
    pra = PRAReasonModel()
    pra.data_preprocess(pra.train_pra_path_features_file)
    pra.data_preprocess(pra.test_pra_path_features_file)
    pra.train()

