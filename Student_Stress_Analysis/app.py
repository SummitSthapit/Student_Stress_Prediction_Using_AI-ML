import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = 'D:\\bootcamp\\AI_ML Bootcamp\\Student_Stress_Analysis\\ML_Model\\knn_model.pkl'
model = joblib.load(model_path)

# Extract feature names from the model if they are available
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")

def main():
    # Set the title of the web app
    st.title('Student Stress Prediction')

    # Add a description
    st.write('Enter student information to predict stressed or not stressed.')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Student Information')

        # Add input fields for features
        student_name = st.text_input('Student Name')
        anxiety_level = st.slider("Student's Anxiety Level",0,30,15)
        self_esteem = st.slider("Student's Self Esteem", 0, 30, 15)
        mental_health_history = st.slider("Mental Health History",0,1,1)
        depression = st.slider("Student's Dression Level",0,30,15)
        headache = st.slider("Student's Number of  Headaches",0,5,3)
        blood_pressure = st.slider("Student's Blood Pressure Level",1,3,2)
        breathing_problem = st.slider("Student's Breathing Problem Level", 0,5,3)
        sleep_quality = st.slider("Student's Sleep Quality Level", 0,5,3)
        noise_level = st.slider("Student's Surrounding Noise Level",0,5,3)
        living_conditions = st.slider("Student's Living Conditions",0,5,3)
        safety = st.slider("Student's safety", 0,5,3)
        basic_needs = st.slider("Student's Basic Needs Level",0,5,3)
        academic_performance = st.slider("Student's Recent Academic Performance",0,5,3)
        study_load = st.slider("Student's Study Load", 0,5,3)
        teacher_student_relationship = st.slider("Student's Relationship with Teacher",0,5,3)
        future_career_concernss = st.slider("Student's Future Career Concerns",0,5,3)
        social_support = st.slider("Student's Social Support",0,3,1)
        peer_pressure = st.slider("Student's Peer Pressure",0,5,2)
        extracurricular_activities = st.slider("Student's Extracurricular Activities",0,5,2)
        bullying = st.slider("Level of Bullying Faced",0,5,1)
        stress_level = st.slider("Self Estimated Stress Level",0,2,1)
        STRESS = st.slider("Estimated Stress Level(Equal to Above)",0,2,1)
        untrained_column = st.text_input('Additional Information (not used in prediction)')

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'anxiety_level': [anxiety_level],
        'self_esteem': [self_esteem],
        'mental_health_history': [mental_health_history],
        'depression': [depression],
        'headache': [headache],
        'blood_pressure': [blood_pressure],
        'sleep_quality': [sleep_quality],
        'breathing_problem': [breathing_problem],
        'noise_level': [noise_level],
        'living_conditions': [living_conditions],
        'safety': [safety],
        'basic_needs': [basic_needs],
        'academic_performance': [academic_performance],
        'study_load': [study_load],
        'teacher_student_relationship': [teacher_student_relationship],
        'future_career_concerns': [future_career_concernss],
        'social_support': [social_support],
        'peer_pressure': [peer_pressure],
        'extracurricular_activities': [extracurricular_activities],
        'bullying': [bullying],
        'stress_level': [stress_level],
        'STRESS': [STRESS]
    })

    # Ensure columns are in the same order as during model training
    input_data = input_data[expected_columns]

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]
            
            st.write(f'Prediction for {student_name}: {"Stressed" if prediction[0] == 1 else "Not Stressed"}')
            if prediction[0] == 1:
                st.success(f"{student_name} is stressed. Consider improving living condtions, realtions, and seek professional help")
            else:
                st.error(f"{student_name} is not stressed.")

if __name__ == '__main__':
    main()
