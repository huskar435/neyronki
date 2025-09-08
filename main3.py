from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

text = [
    "Фильм потрясающий,смотрел на одном дыхании",#положительный
    "Очень скучно и длинно",#отрицательный
    "Неплохо,но второй раз бы не пошел",#отрицательный
    "Шикарная игра актеров!",#положительный
]
lable = [1,0,0,1] # 1-положительный, 0-отрицательный

text_train, text_test, y_train, y_test = train_test_split(text, lable, test_size=0.33, random_state=20)

print(text_train)
print(text_test)
print(y_train)
print(y_test)



pipe =  make_pipeline(
    CountVectorizer(),
    MultinomialNB()

)
pipe.fit(text_train, y_train)
y_pred = pipe.predict(text_test)
print(f"Accuracy: {accuracy_score(y_test,y_pred)}")

print("New:", pipe.predict(["Потрясающий фильм, но второй раз бы не пошел "])[0])
print("New:", pipe.predict(["Очень скучно,но игра актеров шикарная "])[0])             

             