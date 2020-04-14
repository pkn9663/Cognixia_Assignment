"""
******************************************************************************
*        Cogninxia project:                                                  *
*                                                                            *
*            AUTHOR : PRAVEEN KUMAR N                                        *
*            DATE: 08-April-2020                                             *
*                                                                            *
******************************************************************************
"""
import pandas as pd
import numpy as np

df = pd.read_excel("D:/COGNIXIA_Machine_Learning/Spyder/2. Project/marks.xlsx", 
                   sheet_name = "Marks")
df.head()
type(df)

# 2. percentage column to data table

df2 = ((df.maths+df.science+df.english)/300)*100

df = pd.concat([df , df2] , axis = 1)  

df.rename(columns={0: "percentage"} , inplace = True)
df
    
# 1. class1 & class2 student data
g = df.groupby("class")
g

for class_ , df_ in g:
    print(class_)
    print(df_)
    
class1 = g.get_group(1)
class1
class2 = g.get_group(2)
class2

# 3. Topper of class 1 in math
print(class1[class1.maths == max(class1.maths)])

# 4. Topper of class 2 in english
print(class2[class2.english == max(class2.english)])

# 5. Topper in both classes
print(df[df.percentage == max(df.percentage)])

# 6. Names of the students which starts with ‘A’ along with their class

df.get(df["Name"].str.startswith("A"))

"""

    7. Names of all students who failed in class 1 in any subject 
        (less than 40% mark is a fail in any subject)
    8. Names of students in class 1 who failed in all subjects

"""
print(class1[class1.percentage < 40] ) #& (class1.science) < 40 & (class1.english) < 40])







