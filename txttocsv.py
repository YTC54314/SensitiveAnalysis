import pyprind
import pandas as pd
import os
import numpy as np

pbar = pyprind.ProgBar(50000)#進度條#總共有50000个文件
labels = {"pos":1,"neg":0} 
data = pd.DataFrame()
for s in ("test","train"):
    for l in ("pos","neg"):
        #影評的存放路徑
        path = "C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/aclImdb/%s/%s"%(s,l)
        #list路徑裡所有的檔案
        for file in os.listdir(path):
            with open(os.path.join(path,file),"r",encoding="utf-8") as infile:#打開路徑下的檔案 #join(path,file)路徑和檔案結合
                #讀檔案內容
                txt = infile.read()
            #將影評和情感標籤存入到DataFrame中
            data = data.append([[txt,labels[l]]],ignore_index=True)
            #更新進度條
            pbar.update()
#設置列名
data.columns = ["review","sentiment"]
#seed影響permutation打亂影評順序-->參數相同,打亂的順序會是一樣的
#np.random.seed(0)
#permutation-->打亂影評順序按照Seed的參數"隨機"打亂data的index排列(index-->每列的指標數)
#若要真正隨機,就不要用seed
#reindex-->照打亂的index再次重新給data對應的index
data = data.reindex(np.random.permutation(data.index))
#存檔
#index=False不顯示index
data.to_csv("C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/aclImdb.csv",index=False)
