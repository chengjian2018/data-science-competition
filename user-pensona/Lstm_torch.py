import torch
import torch.nn as nn
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import os.path as osp
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score,recall_score
warnings.filterwarnings('ignore')

emb_size = 64
MAX_NB_WORDS = 230637
max_len = 128
batch_size = 512
epochs = 10
num_fold= 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "h:/cj/user_persona/data"
train = pd.read_csv(osp.join(path,'train.txt'), header=None, names=['pid', 'label', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'make', 'model'])
test = pd.read_csv(osp.join(path,'test.txt'), header=None, names=['pid', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'make', 'model'])
data = pd.concat([train, test])
import logging  # 引入logging模块
import os.path
import time
# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Log等级总开关
# 第二步，创建一个handler，用于写入日志文件
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = os.path.dirname(os.getcwd()) + '/'
log_name = log_path + rq + '.log'
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
# 第三步，定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
# 第四步，将logger添加到handler里面
logger.addHandler(fh)

data['tagid'] = data['tagid'].apply(lambda x: eval(x))
sentences = data.tagid.values.tolist()

for i in tqdm(range(len(sentences))):
    sentences[i] = [str(item) for item in sentences[i]]

model = Word2Vec(sentences, vector_size=emb_size, window=6, min_count=5, sg=0, hs=0, seed=2021)

for i in tqdm(range(len(sentences))):
    slen = len(sentences[i])
    if slen<max_len:
        sentences[i] = ['sep']*(max_len-slen) + sentences[i]
    else:
        sentences[i] = sentences[i][-max_len:]

sentences = np.array(sentences)

class Mydataset(Dataset):
    def __init__(self,status,sentences,label):
        super(Mydataset,self).__init__()
        self.corpors = sentences
        self.status = status
        self.label = label
    def __getitem__(self, idx):
        s = self.corpors[idx]
        embedding = np.zeros((max_len,emb_size))
        for i,w in enumerate(s):
            if w in model.wv.key_to_index:
                embedding[i,:] = model.wv[w]
        if self.status == 'train':
            label = self.label[idx]
            return embedding,label
        else:
            return embedding

    def __len__(self):
        return len(self.corpors)



class Mymodel(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(Mymodel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True,dropout=0.5)
        self.out = nn.Linear(self.hidden_size,2)


    def forward(self,x):
        # import pdb
        # pdb.set_trace()
        self.hidden = self.initHidden(x.size(0))
        self.hidden = [item.to(device) for item in self.hidden]
        x,_ = self.lstm(x,self.hidden)
        out = self.out(x[:,-1,:])
        return out


    def initHidden(self,batch_size):
        if self.lstm.bidirectional:
            return (torch.rand(self.num_layers*2,batch_size,self.hidden_size),torch.rand(self.num_layers*2,batch_size,self.hidden_size))
        else:
            return (torch.rand(self.num_layers,batch_size,self.hidden_size),torch.rand(self.num_layers,batch_size,self.hidden_size))

def get_acc(output,label):
    # import pdb
    # pdb.set_trace()
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct



if __name__ == '__main__':
    folds = StratifiedKFold(n_splits=num_fold,shuffle=True,random_state=2021)
    y_categorical = train.label.values
    test_dataset = Mydataset(status='test', sentences=sentences[train.shape[0]:], label=None)
    for fold_,(trn_idx,val_idx) in enumerate(folds.split(train,train['label'])):
        logger.info("fold {} \n".format(fold_ + 1))
        n = len(trn_idx)
        tra_dataset = Mydataset(status='train',sentences=sentences[trn_idx],label=y_categorical[trn_idx])
        val_dataset = Mydataset(status='valid',sentences=sentences[val_idx],label=None)
        tra_dataloader = DataLoader(tra_dataset,batch_size=batch_size,shuffle=True)
        val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
        mymodel = Mymodel(input_size=64,hidden_size=128,num_layers=3).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mymodel.parameters()),lr=1e-3,betas=(0.9, 0.99))
        # optimizer = torch.optim.Adam(mymodel.parameters(), lr=1e-3)
        for i in range(epochs):
            train_loss = 0
            acc_global = 0
            trainbar = tqdm(tra_dataloader)
            for x,y in trainbar:
                import pdb
                pdb.set_trace()
                x = x.to(torch.float32).to(device)
                y = y.to(device)
                predict = mymodel(x)
                loss = criterion(predict,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_acc = get_acc(predict,y)
                acc_global += num_acc
                trainbar.set_description("epoch:{} train_loss:{} train_acc:{}%".format(i,loss.item(),round(num_acc*100.0/batch_size,4)))

            logger.info("fold {} epoch {} precision: {} \n".format(fold_,i,acc_global*100.0/n))
            if (i+1)%3==0:
                #valid
                mymodel.eval()
                valbar = tqdm(val_dataloader)
                val_label = y_categorical[val_idx]
                pred = []
                for x in valbar:
                    x = x.to(torch.float32).to(device)
                    predict = mymodel(x)
                    pred_prob,_ = predict.max(1)
                    pred_prob_lst = pred_prob.cpu().detach().numpy().tolist()
                    pred += pred_prob_lst
                bound_37 = [1 if p>0.4 else 0 for p in pred]
                bound_45 = [1 if p>0.45 else 0 for p in pred]
                bound_50= [1 if p > 0.5 else 0 for p in pred]

                logger.info("{} fold; {} recall; {} recall; {} recall".format(fold_, recall_score(bound_37, val_label),
                                                                              recall_score(bound_45, val_label),
                                                                              recall_score(bound_50, val_label)))
                logger.info("{} fold; {} acc; {} acc; {} acc".format(fold_, accuracy_score(bound_37, val_label),
                                                                     accuracy_score(bound_45, val_label),
                                                                     accuracy_score(bound_50, val_label)))
                logger.info("{} fold; {} f1; {} f1; {} f1 \n".format(fold_, f1_score(bound_37, val_label),
                                                                     f1_score(bound_45, val_label),
                                                                     f1_score(bound_50, val_label)))
                mymodel.train()

        test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
        predicts = []
        for x_test in test_dataloader:
            x_test = x_test .to(torch.float32).to(device)
            pred_ = mymodel(x_test)
            pred_prob_test,_ = pred_.max(1)
            predicts += pred_prob_test.cpu().detach().numpy().tolist()
        test['fold_'+str(fold_)] = predicts
        del mymodel
        torch.cuda.empty_cache()
    test = test.rename(columns={"pid":"user_id"})
    # test.to_csv("submit/torch_lstm.csv")
    save_cols = ["user_id"]
    for i in range(num_fold):
        save_cols.append('fold_'+str(i))
    test[save_cols].to_csv('submit/lstm_torch.csv', index=False)

    # mydataset = Mydataset(status='train',sentences=sentences[:train.shape[0]],label=train.label.values)
    # traindataloader = DataLoader(mydataset,batch_size=batch_size,shuffle=True)
    # mymodel = Mymodel(input_size=64,hidden_size=128,num_layers=3).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(mymodel.parameters(),lr=1e-3)
    # n = train.shape[0]
    # for _ in range(epochs):
    #     print('begin {} epoch'.format(_))
    #     train_loss = 0
    #     train_acc = 0
    #     trainbar = tqdm(traindataloader)
    #     for x,y in trainbar:
    #         x = x.to(torch.float32).to(device)
    #         y = y.to(device)
    #         predict = mymodel(x)
    #         loss = criterion(predict,y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         train_loss += loss.item()
    #         num_acc = get_acc(predict,y)
    #         train_acc += num_acc
    #         trainbar.set_description("epoch:{} train_loss:{} train_acc:{}%".format(_,loss.item(),round(num_acc*100.0/batch_size,4)))
    #
    #     print("precision is {}".format(train_acc*100.0/n))
    #
    # testdataset = Mydataset(status='test',sentences=sentences[train.shape[0]:],label=None)
    # testdataloader = DataLoader(testdataset,batch_size=batch_size,shuffle=False)
    # mymodel.eval()
    # predicts = []
    # for x_test in testdataloader:
    #     x_test = x_test .to(torch.float32).to(device)
    #     y_ = mymodel(x_test)
    #     _, pred_label = y_.max(1)
    #     predicts += pred_label.cpu().numpy().tolist()
    # test['category_id'] = predicts
    # test = test.rename(columns={"pid":"user_id"})
    # # test.to_csv("submit/torch_lstm.csv")
    # test[['user_id', 'category_id']].to_csv('submit/lstm_torch.csv', index=False)