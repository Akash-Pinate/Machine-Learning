'''
    Case Study : Wine Predictor
        -Wine quality(class) is dicided on its contents
        -Calculate the accuracy using - Knn model

'''

#########################################################


# Author         : Akash Pinate
# Date           : 25-March-2022

# ML Type        : Supervised Learning 
# Classifier     : Knn (KNeighborsClassifier)
# Dataset        : WinePredictor.csv
# Features       : Alcohol,Malic acid,Ash,Alcalinity of ash,Magnesium,Total phenols,Flavanoids,Nonflavanoid phenols,Proanthocyanins,Color intensity,Hue,OD280/OD315 of diluted wines,Proline
# Label          : Class



#########################################################


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd


def Wine_predictor():
    #---------------------------------------------------------------------------
    # Step 1: get the features and lable from data
    # 1.1: Read CSV
    wine_data=pd.read_csv("WinePredictor.csv")

    #print(wine_data.head(5))

    data=wine_data[[ 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
           'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
           'Proanthocyanins', 'Color intensity', 'Hue',
           'OD280/OD315 of diluted wines', 'Proline']]

    label=wine_data["Class"]
    
    Accuracy=Accuracy_Result(data,label)

    return Accuracy

     
def Accuracy_Result(data,label):

    x_train, x_test, y_train, y_test=train_test_split(data,label,test_size=0.3)
 
    a=[]
    for n in range(1,16):

        # Algorithm
        Classifier=KNeighborsClassifier(n_neighbors=n)

        # Model
        Model= Classifier.fit(x_train, y_train)
    
        # Prediction
        Prediction=Model.predict(x_test)

        # accuracy_score
        Accuracy=accuracy_score(Prediction,y_test)
        a.append(Accuracy)

    return max(a)
        

# main Entry Function
def main():
    print("\n----- Wine Class Predictor by Akash Pinate : -----")
    print("----- ML Type : Supervised Learning : -----")
    print("----- Algorithm : Knn (KNeighborsClassifier) : -----\n") 

    # Note : Train_test_split Method ->by Default Shuffle the data . so every time Accuracy is Different  
    for i in range(5):
        Accuracy_Result= Wine_predictor()
        print(f"Accuracy Result :{Accuracy_Result*100}")

# starter
if __name__=="__main__":
    main()

#Accuracy Range -> 70-85%






