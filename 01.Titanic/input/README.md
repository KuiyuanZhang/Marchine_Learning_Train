### Overview

The data has been split into two groups:

- training set (train.csv)
- test set (test.csv)

**The training set **should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use [feature engineering ](https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/)to create new features.

**The test set **should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

We also include **gender_submission.csv**, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.

### Data Dictionary

| **Variable** | **Definition**                           | **Key**                                  |
| ------------ | ---------------------------------------- | ---------------------------------------- |
| survival     | Survival          是否生存                   | 0 = No, 1 = Yes                          |
| pclass       | Ticket class    票等级                      | 1 = 1st, 2 = 2nd, 3 = 3rd                |
| sex          | Sex                  性别                  |                                          |
| Age          | Age in years   年龄                        |                                          |
| sibsp        | of siblings / spouses aboard the Titanic        船上兄弟姐妹或者配偶的数量。 |                                          |
| parch        | of parents / children aboard the Titanic       船上父母或者孩子的数量。 |                                          |
| ticket       | Ticket number          票的数目              |                                          |
| fare         | Passenger fare         票价                |                                          |
| cabin        | Cabin number          船舱号                |                                          |
| embarked     | Port of Embarkation        登船地点          | C = Cherbourg, Q = Queenstown, S = Southampton |

### Variable Notes

**pclass**: 社会经济地位的指标(SES)
1st = 上
2nd = 中
3rd = 下

**age**: 年龄小于1怎是分数.如果年龄是被估计的, is it in the form of xx.5
**sibsp**: 数据集以这种方式定义家庭关系...
Sibling = 兄弟，姐妹，继兄弟，继父
Spouse = 丈夫，妻子（情妇和未婚妻被忽略）

**parch**: 数据集以这种方式定义家庭关系...
Parent = 母亲，父亲
Child = 女儿，儿子，继女，继子
有些孩子只是和保姆旅行，所以parch = 0。



应该使用测试集来看看你的模型在看不见的数据上表现如何。 对于测试集，我们不提供每个乘客的基本事实。 预测这些结果是你的工作。 对于测试集中的每个乘客，使用您训练的模型来预测他们是否在泰坦尼克号的沉没中幸免于难。

我们还包括gender_submission.csv，作为提交文件应该是什么样子的一个例子，假定所有的和唯一的女性乘客生存。





