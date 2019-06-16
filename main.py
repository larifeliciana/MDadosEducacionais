import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

##RECEBE O NOME DO CSV E OS DADOS E AS CLASSES
def carregar_dados(csv):

    data = pd.read_csv(csv)
    classe = data.keys()[-1] #Pega o nome da coluna das classes

    y = data[classe] #y Recebe as classes de todas as instâncias da base de dados

    X = data.copy()
    del X[classe] #Recebe os demais atributos

    return data, X, y

def matriz_correlacao(data): #RETORNA A MATRIZ DE CORRELAÇÃO
    cor = data.corr()
    return cor

def seleciona_features(data, n): #Retorna os n mais relevantes features de acordo com a correlação deles com a classe
    cor = matriz_correlacao(data) #Matriz de correlação
    classe = data.keys()[-1]
    cor_target = abs(cor[classe]) #Correlação com a classe
    relevant_features = cor_target.sort_values(ascending=False) #Ordena
    return relevant_features.keys()[1:n] #Pega as n maiores correlações com a classe

def classificador(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    classificador = RandomForestClassifier()

    classificador.fit(X_train, y_train)

    y_pred = classificador.predict(X_test)
    avaliacao(y_pred,y_test)
    return y_pred

def avaliacao(y_pred, y_test):
    print("Acurácia: " + str(accuracy_score(y_test,y_pred)))
    print("Matriz de Confusão: ")
    print(confusion_matrix(y_test,y_pred))

csv = "cognitivePortuguese.csv"
#csv = "cognitiveEnglish.csv"

data,x,y = carregar_dados(csv)
features = seleciona_features(data, 10)
z = classificador(x,y)

#print(features)
#print(z)