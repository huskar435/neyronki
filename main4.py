from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

text = [
    "Лает,охраняет дом",#собака
    "Мурлычет,любит спать на подоконнике",#кот
    "Какает в тапки,любит рыбку на завтрак",#кот
    "Скулит, грызет косточку ",#собака

    
  
]
lable = [1,0,0,1] # 1-собака, 0-кот

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

print("New:", pipe.predict(["Мяукает и какает в тапки,а еще обожает спать на диване "])[0])
print("New:", pipe.predict(["ноет, играет с мячиком"])[0])             

             