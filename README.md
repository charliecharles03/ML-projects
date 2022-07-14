# 

what you can learn from this 
mostly my experience:

this was fun to learn a thing or two about neural networks and also what a node is and how it works and 
the most beautiful this is that linear regression output is processed by logistic regression to form a node
which kind of made me see the big picture of learning this far it always had a purpose.
other than that the technicality is that I used google collabe to write this code which I found online which was difficult 
to understand in the beginning but slowly and steadily I can to when I was going with this project and also how dense neural networks
and what a hidden lays does in a neural network and not to forget how neural net algorithms make is own data set. 
it was not a superior level of understanding that I had in this topic but was sure worth going the distance to know the seriously dense beauty of these algorithms.



FYI:no need to download data set its already avaliable in sklearn
code:

from sklearn.datasets import load_digits
%matplotlib inline
import matplotlib.pyplot as plt
digits = load_digits()
import os

plt.gray()
for i in range(5):
  plt.matshow(digits.images[i])
  from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size = 0.2)
len(y_train)



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train,y_train)

model.coef_

model.intercept_

y_ans=model.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_ans,y_test)
