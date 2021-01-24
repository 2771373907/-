import pandas as pd
import numpy as np
# 将所有文本和标题汇总成一个txt
def get_text(base_path, out_path):
#只要评论列
    data = pd.read_csv(base_path, usecols=[1], encoding='UTF-8')
    data.columns = ["comment"]
    # data = open(base_path, 'r', encoding='UTF-8')
    f2 = open(out_path, 'w', encoding='UTF-8')
    for i in range(len(data)):
        try:
            context=data["comment"][i]
            if context=="我是一只大大龙":
                pass
            f2.write(str(np.squeeze(data.iloc[i, [0]].values)) + ',')
            f2.write(context+ '\n')
        except:
            print(data.iloc[i,[0]].values)

    f2.close()

def excute():
    base_path = "./get_data/test.csv"
    out_path = "./get_data/data.txt"
    get_text(base_path, out_path)


