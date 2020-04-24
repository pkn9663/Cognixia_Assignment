"""
******************************************************************************
*        Cogninxia project:                                                  *
*                                                                            *
*            AUTHOR : PRAVEEN KUMAR N                                        *
*            DATE: 08-April-2020                                             *
*
*                                                                            *
******************************************************************************
        1. Find the net salary after 30% deduction of tax
        2. remove all rows having null value
        3. filter the salaries of people where  name starts with 'M'
        4. Create another col Int_Sal where salary is stored as integer type
        5. Find the name of the employee having maximum salary
        6. Filter the salaries of the employees having 'm' in the last name
        7. count the number of salaries where null values are not there
        8. Sort the data based on name
        9. Filter the names of employees where salary in between 400000 to 600000
        10.find the mean salary from salary col
        
"""

import pandas as pd
import numpy as np

data = pd.read_excel("D:/COGNIXIA_Machine_Learning/Spyder/3. Project/empsal.xlsx",
                     sheet_name = "salary")
data.count() # --we have 14 Records

# 2. handelling missing DATA to clean the file

data = data.dropna()
data.count() # -- we have 12 records

data.columns
data = data.rename({" Salary": "salary" , "Name": "firstname", " Lastname":"lastname"} ,
                   axis = 1)

data = data.reset_index()
data = data.drop("index" , axis =1)

data
# 1. net salary after 30% dedduction from salary

data.salary = data.salary- (data.salary)*.3

data

# 3. filter the salaries of people where  name starts with 'M'
data
data.get(data["firstname"].str.startswith("M"))

# 4. Create another col Int_Sal where salary is stored as integer type

data["int_sal"] = data.salary

type(data.int_sal[0]) # now it's dtype: int64

data.int_sal = np.int32(data.salary)

type(data.int_sal[0]) # now it shows numpy.int32

# 5. Find the name of the employee having maximum salary

print(data[data.salary == max(data.salary)])

# 6. Filter the salaries of the employees having 'm' in the last name

data.get(data["lastname"].str.contains("m"))

# 7. count the number of salaries where null values are not there

data.salary.count() # it shows 12 records

# 8. Sort the data based on name

data.sort_values("firstname" ,ignore_index = True)

# 9. Filter the names of employees where salary in between 20000 to 30000

data[data.salary.between(20000,30000)]

# 10.find the mean salary from salary col

data.salary.mean() # data["salary"].mean()


# saving salary_new.xlsx

data.to_excel("D:/COGNIXIA_Machine_Learning/Spyder/3. Project/salary_new.xlsx" ,
              sheet_name ="salary" ,index = False)







