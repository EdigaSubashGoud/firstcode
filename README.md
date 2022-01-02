# firstcode
This is first code in git hub
#multiple classification

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
# to have a data set from sklearn load digits
# we need to predict the blured value in the image
d=load_digits()
dir(d)

d['images']

d.data[0]

plt.gray()
for i in range(0,5):
  plt.matshow(d.images[i])

d.target[0:5]

use data, target to train our model

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(d.data,d.target,test_size=0.2)

len(x_test)

create a model LogisticRegression

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)

model.score(x_test,y_test)

model.predict([d.data[6]])

