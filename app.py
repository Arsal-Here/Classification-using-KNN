
import streamlit as st
import joblib
import numpy as np

knn_model = joblib.load('Train\model.pkl')
student_array_dict = joblib.load('Train\student_dict.pkl')

val = st.text_input("Enter Roll num")

val_info = student_array_dict.get(val)
query = val_info[:-1]
print(query)
query = query.reshape(1,-1)

predicted_value = knn_model.predict(query)

print(predicted_value)
if predicted_value[0] == 1:  # assuming 1 = Placed, 0 = Not Placed
    st.write("✅ Student has been Placed")
else:
    st.write("❌ Student has not been Placed")

