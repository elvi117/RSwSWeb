from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy

class DataProvider():


    def triplesCreator(self):
        values = [4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 31, 33, 34, 35, 37, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63]
        tableX = []
        tableY = []
        with open('binary_output_for.txt') as f:
            singleObject = ""
            i = 0
            for k in f:
                if k == "--------------------------\n":         #koniec jednej krotki
                    tableX.append(str(singleObject))
                    singleObject = ""
                    i = 0
                    continue
                if i in values:
                    if len(k) > 1:
                        w = k.replace("[", "").replace("'", "").replace("(", "").replace(")", "").replace("]", "").replace("\n", "").replace(" ", "")
                        singleObject += str(w)
                elif i == 0:
                    w = k.replace("[", "").replace("'", "").replace("(", "").replace(")", "").replace("]", "").replace("\n", "").replace(" ", "")
                    if w == "relevant,":
                        tableY.append(1)
                    else:
                        tableY.append(0)
                i += 1


        return (tableX, tableY)

    def split(self, ratio, x, y ):
        # podział na konkretne zbiory - ratio jest tu rozumiane jako rozmiar testSet'u a nie trainingSetu - takie jest założenie sklearn
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=ratio)


provider = DataProvider()
w = provider.triplesCreator()
#print(w[1])

count_vect = CountVectorizer(binary=True)
data = count_vect.fit_transform(w[0]).toarray()
#print(data)
#print(count_vect.get_feature_names())

p1 = []
p2 = []
p3 = []
tableOfSplits = [0.75, 0.5, 0.25]
for k in tableOfSplits:
    p11 = []
    p22 = []
    p33 = []
    for t in range(0,10):
        provider.split(k,data,w[1])
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB().fit(provider.X_train, provider.y_train)
        predicted = clf.predict(provider.X_test)
        p11.append( np.mean(predicted == provider.y_test))

        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier().fit(provider.X_train, provider.y_train)
        predicted = clf.predict(provider.X_test)
        p22.append( np.mean(predicted == provider.y_test))

        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(provider.X_train, provider.y_train)
        predicted = clf.predict(provider.X_test)
        p33.append( np.mean(predicted == provider.y_test))
    p1.append(p11)
    p2.append(p22)
    p3.append(p33)



print(scipy.stats.friedmanchisquare(p1,p2,p3))

print("Bayes")
print(p1)
print("**********")
print("SGD")
print(p2)
print("**********")
print("NN")
print(p3)


import matplotlib.pyplot as plt


i = 0.75
for t in p1:
    label = "Bayes " + str(i)
    plt.plot(np.array([1,2,3,4,5,6,7,8,9,10]), np.array(t), label=label)
    i = i-0.25

i = 0.75
for t in p2:
    label = "SGD " + str(i)
    plt.plot(np.array([1,2,3,4,5,6,7,8,9,10]), np.array(t), label=label)
    i = i - 0.25

i = 0.75
for t in p3:
    label = "NN " + str(i)
    plt.plot(np.array([1,2,3,4,5,6,7,8,9,10]), np.array(t), label=label)
    i = i - 0.25



plt.legend(bbox_to_anchor=(0.8, 0.9), loc=2, borderaxespad=0.)
plt.show()






