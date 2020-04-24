# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:50:35 2020

@author: Praveen Kumar

******************************************************************************
*        Cogninxia project:                                                  *
*                                                                            *
*            AUTHOR : PRAVEEN KUMAR N                                        *
*            DATE: 08-April-2020                                             *
*                                                                            *
*        You have been supplied with two lists of data: monthly Sales and    *
*        monthly profit for the financial year.                              *
*        Your tasks is to calculate the following metrics:                   *
*        • Profit ratio for each month                                       *
*        • Calculate mean sales and mean profit                              *
*        • Good months - Find the month with above mean profit               *
*        • Bad months - Find the month with below mean profit                *
*        • Best month – Find the best month where profit is maximum          *
*        • Worst month – Find the worst month where profit is minimum        *
*        • Worst month – Profit after tax was the min for the year           *
*                                                                            *
******************************************************************************

"""
import pandas as pd
import numpy as np


#Data 
sales = [14434.65, 21222.61, 16554.34, 15445.32, 16054.52, 19005.23, 22222.22, 17466.29, 11345.21, 14333.43, 14444.45, 21222.10]
profit = [1222, -500, 1343, 2222, 2122, 3122, 1000, 5330, 2123, 4332, 2221, 3213]

type(sales)

# 1 . profit ration for each month

profit_ratio = []
for i in range(0 , len(profit)):
    profit_ratio.append(sales[i]%profit[i])
print(profit_ratio)

# 2. Calculate the mean sales & mean profit

mean_sales = np.mean(sales)
mean_profit = np.mean(profit)

# 3. Good & Bad months which has sale above its mean profit

good_sales = []
for i in range( 0 , len(sales)):
    good_sales.append(profit[i] > mean_profit)
print(good_sales)

bad_months = []
for i in range (0, len(sales)):
    bad_months.append(profit < mean_profit)
bad_months

# 4. best month which has max profit

best_month = []
for i in range(0 , len(sales)):
    best_month.append(profit[i] == max(profit))
best_month[i]

best_month = max(profit)
best_month