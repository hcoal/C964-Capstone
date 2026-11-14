import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import create_model

titanic_set = pd.read_csv("data/Titanic-Dataset.csv")
titanic_set.dropna(subset="Age", inplace=True)
st.title("Titanic Survival Predictor")
st.header("Enter your information and find out if you're unsinkable")

with st.form(key="user_input"):
    age = st.number_input("Age", min_value=0, step=1)
    st.divider()
    sex_input_note = st.text("If you are female, please put a 1 in the female field and a 0 in the male field. Likewise, if you are male, mark a 0 in the female field, and a 1 in the male field.")
    sex_female = st.number_input("Female", min_value=0, max_value=1 )
    sex_male = st.number_input("Male", min_value=0, max_value=1 )
    st.divider()
    ticket_class = st.selectbox("Please select your ticket class", [1, 2, 3])
    ticket_class_note = st.caption("* A note regarding ticket class: ticket classes are based on economic status. "
                                   "1st class was mainly for the wealthy and aristocrats. 2nd class was middle class - "
                                   "teachers, musicians and the like. 3rd class was for those coming to America or Canada and often times "
                                   "started a new life.")


    predict_button = st.form_submit_button(label="Predict")


    if predict_button:
        if (sex_female == 0 and sex_male == 0) or (sex_female == 1 and sex_male == 1):
            st.warning("Please mark a 1 or 0 based on your sex. Both cannot be the same value.")
        else:
            user_input = pd.DataFrame([{"Sex Female": sex_female, "Sex Male": sex_male, "Ticket_Class": ticket_class, "Age": age}])
            prediction = create_model.model.predict(user_input)
            st.subheader("And the results are:")
            if prediction == 1:
                st.success("You are unsinkable! :)")
            elif prediction == 0:
                st.error("You went down with the ship :(")


st.subheader("Here are some fun data visualizations about the Titanic:")
st.caption("There are 3 here, remember to scroll down!")

# visualizations:
fig, ax = plt.subplots(figsize = (5, 5))
ax.set_title("How many people survived the Titanic")
ax.set_xlabel("Survived")
ax.set_ylabel("# of People")
ax = titanic_set["Survived"].value_counts().plot( kind="bar")
first_leg = ax.legend(["0-Did not survive"], loc="upper right")
ax.add_artist(first_leg)
ax.legend(["1-Survived"], loc="center right")
st.pyplot(fig)

fig, ax = plt.subplots(figsize = (5, 5))
ax.set_title("How many men and women on the Titanic")
ax.set_ylabel("# of Men and Women")
ax = titanic_set["Sex"].value_counts().plot(kind="bar")
st.pyplot(fig)

fig, ax = plt.subplots(figsize = (6, 8))
ax.set_title("Age and Ticket Class Survivals")
ax.set_ylabel("Age")
ax.set_xlabel("Ticket Class")
scatter_plot = ax.scatter(x=titanic_set["Pclass"], y=titanic_set["Age"], c=titanic_set["Survived"])
ax.legend(*scatter_plot.legend_elements(), loc="upper right")
st.pyplot(fig)

