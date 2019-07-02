import pandas as pd
dataset =  pd.read_csv('datasets/Womens Clothing E-Commerce Reviews.csv')
dataset=dataset[['Review Text','Rating']]
dataset['Rating'].replace(to_replace=[1,2],value=0,inplace=True)
dataset['Rating'].replace(to_replace=[3],value=pd.NaT,inplace=True)
dataset['Rating'].replace(to_replace=[4,5],value=1,inplace=True)
        
f = open('datasets/wcer-lite.csv','w+')
f.write(dataset.to_csv(index=False))
f.close()