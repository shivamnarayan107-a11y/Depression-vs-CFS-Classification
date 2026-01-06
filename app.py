import pandas as pd 
import numpy as np
import sklearn
import pickle
import streamlit as st


lr = pickle.load(open('lr1.pkl','rb'))
dt = pickle.load(open('dt1.pkl','rb'))
rf = pickle.load(open('rf1.pkl','rb'))

model = st.sidebar.selectbox('Select the ML Model',
                             ['LogReg','Decision Tree','Random Forest'])



st.title('ME/CFS, Depression Prediction')

st.write('Fill the details below for diagnosis')

col1, col2 = st.columns(2)

with col1:
  age = st.number_input('Age',18,70,25) # min=18,max=70,default = 25
  gender = st.selectbox('Gender',('Male','Female'))
  sq_index = st.number_input('Sleep Quality Index',1.0,10.0,5.7)
  bf_level = st.number_input('Brain Fog Level',0.0,10.0,2.0)
  pps_score = st.number_input('Physical Pain Score',0.0,10.0,4.2)
  stress_level = st.number_input('Stress Level',0.0,10.0,5.0)
  dep_phq9 = st.number_input('Depression phq9 Score',0,27,19)

with col2:
  fs_scale = st.number_input('Fatigue Severity Scale Score',0.0,10.0,5.6)
  pem_dur = st.number_input('PEM Duration',0,47,16)
  sleep_hrs = st.number_input('Sleep Hrs',3.0,10.0,5.5)
  pem_present = st.selectbox('Is PEM Present',('Yes','No'))
  med = st.selectbox('Meditation/Mindfulness Present',('Yes','No'))
  work_status = st.selectbox('Work Status',('Partially working','Working','Not working'))
  social_level = st.selectbox('Social Activity Level',('Very low','High','Low','Very high','Medium'))
  ex_freq = st.selectbox('Exercise Frequency',('Rarely','Sometimes','Never','Often','Daily'))

if gender=="Male":
  gen_m = 1
  gen_f = 0
else:
  gen_m = 0
  gen_f = 1

if pem_present=="Yes":
  pem_y = 1
  pem_n = 0
else:
  pem_y = 0
  pem_n = 1

if med == "Yes":
  med_y = 1
  med_n = 0
else:
  med_y = 0
  med_n = 1

if work_status == "Partially working":
  ws_pw = 1
  ws_w = 0
  ws_nw = 0
elif work_status == "Working":
  ws_pw = 0
  ws_w = 1
  ws_nw = 0
else:
  ws_pw = 0
  ws_w = 0
  ws_nw = 1

if social_level == "Very low":
  sl_vl = 1
  sl_h = 0
  sl_l = 0
  sl_vh = 0
  sl_m = 0
elif social_level == "High":
  sl_vl = 0
  sl_h = 1
  sl_l = 0
  sl_vh = 0
  sl_m = 0
elif social_level == "Low":
  sl_vl = 0
  sl_h = 0
  sl_l = 1
  sl_vh = 0
  sl_m = 0
elif social_level == "Very high":
  sl_vl = 0
  sl_h = 0
  sl_l = 0
  sl_vh = 1
  sl_m = 0
else:
  sl_vl = 0
  sl_h = 0
  sl_l = 0
  sl_vh = 0
  sl_m = 1


if ex_freq == "Rarely":
  ex_r = 1
  ex_s = 0
  ex_n = 0
  ex_of = 0
  ex_da = 0
elif ex_freq == "Sometimes":
  ex_r = 0
  ex_s = 1
  ex_n = 0
  ex_of = 0
  ex_da = 0
elif ex_freq == "Never":
  ex_r = 0
  ex_s = 0
  ex_n = 1
  ex_of = 0
  ex_da = 0
elif ex_freq == "Often":
  ex_r = 0
  ex_s = 0
  ex_n = 0
  ex_of = 1
  ex_da = 0
else:
  ex_r = 0
  ex_s = 0
  ex_n = 0
  ex_of = 0
  ex_da = 1



test_input = np.array([age,gen_m,sq_index,bf_level,pps_score,stress_level,dep_phq9,fs_scale,pem_dur,sleep_hrs,pem_y,med_y,ws_pw,ws_w,sl_l,
              sl_m,sl_vh,sl_vl,ex_n,ex_of,ex_r,ex_s]).reshape(1,-1)


df_columns = ['age', 'gender', 'sleep_quality_index', 'brain_fog_level',
       'physical_pain_score', 'stress_level', 'depression_phq9_score',
       'fatigue_severity_scale_score', 'pem_duration_hours',
       'hours_of_sleep_per_night', 'pem_present', 'meditation_or_mindfulness',
       'work_status_Partially working', 'work_status_Working',
       'social_activity_level_Low', 'social_activity_level_Medium',
       'social_activity_level_Very high', 'social_activity_level_Very low',
       'exercise_frequency_Never', 'exercise_frequency_Often',
       'exercise_frequency_Rarely', 'exercise_frequency_Sometimes']

test_df = pd.DataFrame(test_input,columns=df_columns)
st.write('Test Data')
st.write(test_df)

if st.button('Predict'):
    if model == "LogReg":
      pred = lr.predict(test_df)
      st.success(pred[0])
    elif model == "Decision Tree":
      pred = dt.predict(test_df)
      st.success(pred[0])
    else:
      pred = rf.predict(test_df)
      st.success(pred[0])

