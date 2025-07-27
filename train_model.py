import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('data\college_student_placement_dataset.csv')
search_df = pd.read_csv('data\college_student_placement_dataset.csv')
#print(df.head())

df.Placement = (df.Placement=='Yes').astype(int)
df.Internship_Experience = (df.Internship_Experience=='Yes').astype(int)
y = df.Placement
cols = df.columns[1:-1]
x = df[cols]
print(y)
print(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8)

knn_model = KNeighborsClassifier(n_neighbors = 3)

knn_model.fit(x_train,y_train)
ypred = knn_model.predict(x_test)
print(classification_report(ypred,y_test))

joblib.dump(knn_model, "model.pkl")

unique_df = df.drop_duplicates(subset="College_ID", keep="first")

# Convert to dictionary
student_dict = unique_df.set_index("College_ID").to_dict(orient="index")

student_array_dict = {k: np.array(list(v.values())) for k, v in student_dict.items()}


joblib.dump(student_array_dict, "student_dict.pkl")
