import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import  Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import  PCA
import graphviz
import pydotplus


# sl:satisfaction_level-----Fale:MiniMaxScaler;True:StandeardScaler
# ke:last_evaluation--------Fale:MiniMaxScaler;True:StandeardScaler
# npr:number_project--------Fale:MiniMaxScaler;True:StandeardScaler
# amh:average_montly_hours--------Fale:MiniMaxScaler;True:StandeardScaler
# tsc:time_spend_company--------Fale:MiniMaxScaler;True:StandeardScaler
# wa:Work_accident--------Fale:MiniMaxScaler;True:StandeardScaler
# pl5:promotion_last_5years--------Fale:MiniMaxScaler;True:StandeardScaler
# dp:Department-------False:LabelEncoder;True:OneHotEncoder
# slr:salary----------False:LabelEncoder;True:OneHotEncoder
def hr_preprocessing(sl=False, le=False, npr=False, amh=False, tsc=False, wa=False, pl5=False, dp=False, slr=False,
                     lower_d=False, ld_n=1):
    df = pd.read_csv("./HR.csv")

    # 1、清洗数据
    df = df.dropna(subset=["satisfaction_level", "last_evaluation"])
    df = df[df["satisfaction_level"] <= 1][df["salary"] != "nam"]

    # 2、得到标注
    label = df["left"]
    df = df.drop("left", axis=1)  # 以列为索引进行删除

    # 3、特征处理
    #离散化
    scaler_lst = [sl, le, npr, amh, tsc, wa, pl5]
    column_lst = ["satisfaction_level", "last_evaluation", "number_project", \
                  "average_montly_hours", "time_spend_company", "Work_accident", \
                  "promotion_last_5years"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
#            print(df[column_lst[i]].values.reshape(-1, 1))

            df[column_lst[i]] = \
                MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            df[column_lst[i]] = \
                StandardScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
    #数值化
    scaler_lst = [slr, dp]
    column_lst = ["salary", "Department"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            if column_lst[i] == "salary":
                df[column_lst[i]] = [map_salary(s) for s in df["salary"].values]
            else:
                df[column_lst[i]] = LabelEncoder().fit_transform(df[column_lst[i]])
        else:
            df = pd.get_dummies(df, columns=[column_lst[i]])
    if lower_d:
        #        return LinearDiscriminantAnalysis(n_components=ld_n)
        return PCA(n_components=ld_n).fit_transform(df.values), label

    return df, label

d = dict([("low",0),("midium",1),("high",2)])
def map_salary(s):
    return d.get(s, 0)


def hr_modeling(features,labels):
    from sklearn.model_selection import  train_test_split   #----切分数据
    f_v  = features.values
    f_names = features.columns.values
    l_v = labels
    X_tt,X_validation,Y_tt,Y_validation = train_test_split(f_v,l_v,test_size=0.2)
    X_train,X_test,Y_train,Y_test = train_test_split(X_tt,Y_tt,test_size=0.25)
#    print(len(X_train),len(X_validation),len(X_test))


    #------------------model------------------
    from sklearn.metrics import  accuracy_score,recall_score,f1_score   #衡量指标
    from sklearn.neighbors import  NearestCentroid,KNeighborsClassifier #KNN
    from sklearn.naive_bayes import GaussianNB,BernoulliNB              #贝叶斯
    from sklearn.tree import  DecisionTreeClassifier,export_graphviz    #决策树
    from sklearn.svm import SVC                                         #SVM
    from sklearn.ensemble import RandomForestClassifier                 #随机森林
    from sklearn.ensemble import  AdaBoostClassifier                    #AdaBoost
    from sklearn.linear_model import  LogisticRegression                #逻辑回归
    from keras.models import Sequential                                 
    from keras.layers.core import Dense,Activation
    from keras.optimizers import SGD
    from sklearn.ensemble import GradientBoostingClassifier             #GBDT

    # mdl = Sequential()
    # mdl.add(Dense(50,input_dim=len(f_v[0])))
    # mdl.add(Activation("sigmoid"))
    # mdl.add(Dense(2))
    # mdl.add(Activation("softmax"))
    # sgd = SGD(lr=0.01)
    # mdl.compile(loss="mean_squared_error",optimizer="adam")
    # mdl.fit(X_train,np.array([[0,1] if i==1 else [1,0] for i in Y_train]), nb_epoch=10,batch_size=8999)
    #
    # xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
    # for i in range(len(xy_lst)):
    #     X_part = xy_lst[i][0]
    #     Y_part = xy_lst[i][1]
    #     Y_pred = mdl.predict_classes(X_part)
    #     print(i)
    #     print("NN", "-ACC:", accuracy_score(Y_part, Y_pred))
    #     print("NN", "-REC:", recall_score(Y_part, Y_pred))
    #     print("NN", "-F1:", f1_score(Y_part, Y_pred))
    # return

    models = []
    models.append(("KNN",KNeighborsClassifier(n_neighbors=3)))
    models.append(("GaussianNB",GaussianNB()))
    models.append(("BernoulliNB",BernoulliNB()))
    models.append(("DecisionTreeGini:",DecisionTreeClassifier()))
    models.append(("DecisionTreeEntropy",DecisionTreeClassifier(criterion="entropy")))
    models.append(("SVM Classifier",SVC(C=1000)))
    models.append(("RandomForestClassifier",RandomForestClassifier(n_estimators=11)))
    models.append(("AdaBoostClassifier",AdaBoostClassifier(n_estimators=1000)))
    models.append(("LogisticRegression",LogisticRegression()))
    models.append(("GBDT",GradientBoostingClassifier(max_depth=6,n_estimators=100)))

    for clf_name, clf in models:
        clf.fit(X_train,Y_train)
        xy_lst = [(X_train,Y_train),(X_validation,Y_validation),(X_test,Y_test)]
        for i in range(len(xy_lst)):                                 #0 为训练、1为验证、2为测试
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = clf.predict(X_part)
            print(i)
            print(clf_name,"-ACC:",accuracy_score(Y_part,Y_pred))     # 准确率
            print(clf_name,"-REC:",recall_score(Y_part,Y_pred))       # 召回率
            print(clf_name,"-F1:",f1_score(Y_part,Y_pred))            # 综合率


            #  决策树画图
            # dot_data = export_graphviz(clf,out_file = None,        
            #                            feature_names=f_names,
            #                            class_names=["NL", "L"],
            #                            filled=True,
            #                            rounded=True,
            #                            special_characters=True)
            # graph = pydotplus.graph_from_dot_data((dot_data))
            # graph.write_pdf("dt_tree.pdf")

    
    #模型存储
    # from sklearn.externals import joblib
    # joblib.dump(knn_clf,"knn_clf")   #-----存储
    # knn_clf = joblib.load("knn_clf")  #-----使用



def regr_test(features,label):    #线性回归
    print("X",features)
    print("Y",label)
    from sklearn.linear_model import  LinearRegression,Ridge,Lasso
    #regr = LinearRegression()
    #regr = Ridge(alpha = 0.8)    岭回归
    regr = Lasso(alpha=0.002)
    regr.fit(features.values,label.values)
    Y_pred = regr.predict(features.values)
    print("Coef:",regr.coef_)
    from sklearn.metrics import mean_squared_error         #------评估
    print("MSE:",mean_squared_error(Y_pred,label.values))


def main():
    features,label = hr_preprocessing(slr=True )
   # regr_test(features[["number_project","average_montly_hours"]],features["last_evaluation"])
    hr_modeling(features,label)

if __name__ == "__main__":
    main()