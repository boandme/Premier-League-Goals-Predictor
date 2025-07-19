
### All rights reserved by Boandme; sole owner; ###3
import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib as plt
import numpy as np
percent_sum = 0
## This program uses machine learning to predict the amount of goals a player scores in a soccer season(1 yr) based on their total shots, shots on target, and conversion percent. It is trained on a English Premier League stats database

dataFrame = pd.read_csv('epl_player_stats_24_25.csv')
dataFrame['Conversion %'] = dataFrame['Conversion %'].str.replace('%', '')
dataFrame["Conversion %"] = dataFrame["Conversion %"].astype(int)
### Training DATA ####

## Filter the first half of the dataframe to use as training data
midpoint = len(dataFrame)//2
first_half = dataFrame.iloc[:midpoint]




x = first_half[["Shots", "Shots On Target", "Conversion %"]]
y = first_half["Goals"]
model = linear_model.LinearRegression()
model.fit(x,y)
goals = model.predict([[39,12, 5]])
if goals < 0:
    goals = 0



### Testing Data ####
second_half = dataFrame.iloc[midpoint:]

### Use the other half of the dataframe for testing data and report back accuracy

value = dataFrame.iloc[284, dataFrame.columns.get_loc('Shots')]
amount = 280
print(f"Running goalPredictor on {amount} rows of data....")

for i in range(0,amount):
    actual_goals = dataFrame.iloc[[(282+i)], dataFrame.columns.get_loc('Goals')]
    shots = dataFrame.iloc[[(282+i)], dataFrame.columns.get_loc('Shots')]
    shots_on_target = dataFrame.iloc[[(282+i)], dataFrame.columns.get_loc('Shots On Target')]
    conversion_rate = dataFrame.iloc[[(282+i)], dataFrame.columns.get_loc('Conversion %')]
    predicted_goals = model.predict([[int(shots.iloc[0]), int(shots_on_target.iloc[0]), int(conversion_rate.iloc[0])]])
    print(f"Actual:  {int(actual_goals.iloc[0])}")
    print(f"Predicted: {int(predicted_goals[0])}")
    ## Percent accuracy: Predicted/ Actual * 100
    if int(predicted_goals[0]) == int(actual_goals.iloc[0]):
        percent_sum += 100
    elif int(actual_goals.iloc[0]) == 0:
        percent_sum += 0
    else:
        accuracy = (int(predicted_goals[0])/int(actual_goals.iloc[0]))*100
        percent_sum += accuracy
    print(percent_sum)
percent_accuracy = percent_sum/amount
print()
print()
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("SUMMARY: ")
print(f" In {amount} trials of testing on the testing data of the EPL dataset, goalPredictor averaged {percent_accuracy}% success")




##print(f" The amount of goals scored by this player in a soccer season according to GoalPredictor:  {int(goals[0])}")
