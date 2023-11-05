#!/usr/bin/env python
# coding: utf-8

# # Q1 :

# In[3]:


# import the sys module 
import sys

# function to calculate fuel level after driving a certain number of miles
def drive(miles, fuel_efficiency, fuel_level):
    # maximum distance that can be driven with the current fuel level
    max_distance = fuel_efficiency * fuel_level
    # if the number of miles requested to drive is more than the maximum distance, the car cannot drive that far
    if miles > max_distance:
        # print a message indicating how far the car can drive with the current fuel level
        print(f"You don't have enough fuel to drive {miles} miles. You can drive another {max_distance} miles on this gas.")
        # return the current fuel level as it is
        return fuel_level
 # otherwise, the car can drive the requested distance
    else:
        # calculate the fuel used to drive the requested distance
        fuel_used = miles / fuel_efficiency
        # update the fuel level accordingly
        fuel_level -= fuel_used
        # print a message indicating the distance driven and the remaining fuel level
        print(f"You drove {miles} miles. You have {fuel_level:.2f} gallons of gas left.")
        # return the updated fuel level
        return fuel_level
    
    

# function to add gas to the car's tank
def add_gas(gallons, tank_size, fuel_level):
    # if the number of gallons to add is not positive, it is an invalid input
    if gallons <= 0:
        print("You must enter a positive number of gallons.")
    # otherwise, add the requested number of gallons to the fuel level    
    else:
        fuel_level += gallons
        # if the fuel level exceeds the tank capacity, fill the tank and print a message
        if fuel_level > tank_size:
            fuel_level = tank_size
            print(f"You added {gallons:.2f} gallons, but your tank is full. You have {fuel_level:.2f} gallons in your tank.")
             # otherwise, print a message indicating the number of gallons added and the current fuel level
        else:
            print(f"You added {gallons:.2f} gallons. You have {fuel_level:.2f} gallons in your tank.")
    # return the updated fuel level
    return fuel_level




# function to show the current fuel level
def show_fuel_level(fuel_level):
    # print a message indicating the current fuel level
    print(f"You have {fuel_level:.2f} gallons of gas left.")

    
    
# function to log the action performed and its result in a file
def log_action(action, result):
    # open a file named "LogFuel.txt" in append mode and assign it to the variable "logfile"
    with open("Desktop\LogFuel.txt", "a") as logfile:
        # write a line in the file indicating the action and its result
        logfile.write(f"{action}: {result}\n")

        
        
# main function that runs the fuel management program
def main():
    # ask the user to enter the fuel efficiency of the car in miles per gallon
    fuel_efficiency = float(input("Please enter the car's fuel efficiency (miles/gallon):"))
    # ask the user to enter the tank size of the car in gallons
    tank_size = float(input("Please enter the size of the fuel tank (in gallons): "))
    # initialize the current fuel level to zero
    fuel_level = 0.0 
 


    # start a loop that runs until the user chooses to exit
    while True:
        # display the available
        # Print the menu options
        print("What would you like to do:")
        print("1. See current fuel level")
        print("2. Drive")
        print("3. Add gas")
        print("4. Exit")
        # Get the user's choice
        choice = input()
        
        # If the user chooses to see the current fuel level, call the show_fuel_level function and log the action
        if choice == "1":
            show_fuel_level(fuel_level)
            log_action("Current Fuel Level", fuel_level)
        
# If the user chooses to drive, the drive function is called and the user is prompted to enter
# the number of miles they want to drive. The fuel level is updated based on the miles driven,
# and the log_action function is called to record this action in a log file.
        elif choice == "2":
            miles_to_drive = float(input("How many miles do you want to drive? "))
            fuel_level = drive(miles_to_drive, fuel_efficiency, fuel_level)
            log_action(f"Drive {miles_to_drive} miles", fuel_level)

# If the user chooses to add gas, the add_gas function is called and the user is prompted to enter
# the number of gallons they want to add. The fuel level is updated based on the amount added,
# and the log_action function is called to record this action in a log file.
        elif choice == "3":
            gallons_to_add = float(input("How much gas do you want to add? "))
            fuel_level = add_gas(gallons_to_add, tank_size, fuel_level)
            log_action(f"Add {gallons_to_add} gallons", fuel_level)

# If the user chooses to exit, break fuction is called terminate the program.
        elif choice == "4":
           break

# If the user enters an invalid choice, a message is displayed and they are prompted to try again.
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")
    print("GoodBye")

# The main function is called to start the program.    
if __name__ == "__main__":
    main()



# In[ ]:





# # Q2 :

# Q1 : There may be duplicate records in the data. Remove them. How many records do you have now?

# In[4]:


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_csv('Desktop\AluminiumSheetPricesData.csv')

#Length of records before removing duplicate records
ActualLength = len(data)

# Remove duplicate records
data = data.drop_duplicates()

# Print the number of records
print("Number of records before removing duplicates:", ActualLength)
print("Number of records after removing duplicates:", len(data))


# In[ ]:





# Q2 : Draw a Histogram of the Price variable. Is it a bell curve? If not, what is it?

# In[5]:


import matplotlib.pyplot as plt

# Draw a histogram of the Price variable
plt.hist(data['price'], bins=30)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Price')
plt.show()


# ##  The histogram of the Price variable reveals that the distribution is positively right-skewed with a long tail to the right. This means that there are some very high-priced aluminum sheets in the dataset.

# In[ ]:





# Q3 : Do some basic data exploration (e.g. using commands as head( ), info( ), describe( ), nunique( ), etc). 
# Which variables will you NOT select?

# In[6]:


# Printing the first few records
print(data.head())

# Printing the column information
print(data.info())

# Printing the statistical summary of the data
print(data.describe())

# Printing the number of unique values in each column
print(data.nunique())


# # Answer :
# Based on the data exploration, the variables that could potentially be excluded from the model are:
# 
# From the output, we can see that the grade and thickness variables are strings(objects) and have a large number of unique values compared to the other variables.
# 
# Thickness: it has only 8 unique values, and it's not clear how it relates to the other variables.
# Grade: it has only 7 unique values, and it's not clear how it relates to the other variables.
# We will choose not to select these variables for our analysis.

# In[ ]:





# Q4 : Are there any outliers in the data? What about missing values? If any of either, treat them.

# In[7]:


import seaborn as sns

# Detect outliers in the Price variable
sns.boxplot(x=data['price'])

# Replace outliers with the median value
median = data['price'].median()
data.loc[data['price'] > 10000, 'price'] = median

# Remove rows with missing values
data = data.dropna()

# Print the number of records after outlier and missing value treatment
print("Number of records after treatment:", len(data))


# In[8]:


#Check if we have any null, duplicate, Nan values are left
data.isnull().sum()


# In[9]:


#Converting string into '0' and '1' / binary 

df = pd.get_dummies(data, columns=['cut','thickness', 'grade'])
df.columns


# In[10]:


#Checking df values
df.head()


# In[ ]:





# Q5 : Partition the data into a training set (with 70% of the observations), and testing set (with 30% of the observations) using the random state of 12345 for cross validation.

# In[11]:


from sklearn.model_selection import train_test_split

# To Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X = df.drop(['price'],axis=1) # x consistents of all independent variable
y = df['price'] # y is target variable

#test_size=0.3 indicates that the testing set will have 30% of the total observations and random_state=12345 ensures that the same random splits willbe made each time the code is run.

#Training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

# print the shape of the sets
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# In[ ]:





# Q6 : On the partitioned data, build the best KNN model. Show the accuracy numbers. (Hint: What is the best value of k? How do you decide the ‘best k’?)

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import r2_score

# Define a range of k values to test
k_values = list(range(1, 51))

# Train and test KNN models for each k value
best_k = None
best_accuracy = 0

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
# Print the best value of k and its accuracy
print('Best k:', best_k)
print('Accuracy:', best_accuracy)



# In[41]:


#Cross checking if the 'k' value is correct as the k value predicted in graph
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# create an empty list to store the mean squared errors for different k values
mse_values = []

# try different k values from 1 to 50
for k in range(1, 51):
    # create a KNN regressor with k neighbors
    knn = KNeighborsRegressor(n_neighbors=k)
    
    # fit the model on the training data
    knn.fit(X_train, y_train)
    
    # make predictions on the test data
    y_pred = knn.predict(X_test)
    
    # calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    
    # add the mse value to the list
    mse_values.append(mse)
    
# plot the mse values against different k values
plt.plot(range(1, 51), mse_values, marker='o')
plt.xlabel('K')
plt.ylabel('Mean Squared Error')
plt.show()


# In[13]:


#creating empty lists to hold results
training_accuracy = []
test_accuracy = []
# This is a range. This range basically selects the different values of “k”.
neighbors_settings = range(1,51)


# In[14]:


#This step returns an empty model, which is then fit/ run according to the X and y we selected
#above as the training set. For each value of k (1,2,3...9) the corresponding model fit scores are
#added to the two lists we created
for n_neigh in neighbors_settings:
    modelfit = KNeighborsClassifier(n_neighbors=n_neigh)
    modelfit.fit(X_train, y_train)
    training_accuracy.append(modelfit.score(X_train, y_train))
    test_accuracy.append(modelfit.score(X_test, y_test))


# In[15]:


#The two lists (for model training and testing) are then plotted using the code below
plt.plot(neighbors_settings, training_accuracy, label = "Training Accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "Test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("No of KNN")
plt.legend()


# In[16]:


# The best value of k is 4
#Fit the model
modelfit = KNeighborsClassifier(n_neighbors=1)
modelfit.fit(X_train, y_train)
print("Train set accuracy: ", modelfit.score(X_train, y_train))
print("Test set accuracy: ", modelfit.score(X_test, y_test))


# In[ ]:





# Q7 : On the partitioned data, build the best logistic regression model. Show the accuracy numbers

# In[20]:


from sklearn.linear_model import LogisticRegression
import statsmodels as sm

#Instantiate the model
logreg = LogisticRegression(max_iter=10)


# In[21]:


logreg.fit(X_train, y_train)


# In[22]:


print("Train set accuracy: ", logreg.score(X_train, y_train))
print("Test set accuracy: ", logreg.score(X_test, y_test))


# In[ ]:





# Q8 : Based on the results of k-nearest neighbor, and logistic regression, what is the best model to classify 
# the data? Provide explanation to support your argument.

# Answer : Based on the results, KNN model performs significantly better than logistic regression. The accuracy of the KNN model on the training set is very high, at 0.995, indicating that it is overfitting the data. However, on the test set, it still performs better than logistic regression with an accuracy of 0.142.
# 
# On the other hand, the logistic regression model performs very poorly on both the training and test sets, with an accuracy of only 0.098.
# 
# Therefore, based on these results, the KNN model is the better choice for classifying the data. However, it is important to note that further analysis and fine-tuning of the models may be necessary to achieve even better results.

# In[ ]:




