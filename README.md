## Investment Analysis in Python

To day i was asked to analyse a apartment purchase proposal. 

The owner wanted to know if buying this apartment was a good idea back in 2013 and how mutch money he would be actually losing or earning by accepting the amount offered.

My strategy to solve this problem was to bring the amounts spent and received through out the years (2013 - 2022) to **Present Value** and compare them with government bonds that track IPCA idexes (Índice Nacional de Preços ao Consumidor Amplo). 

here is all the information i know about the proposal:

1. The apartment cost R$ 400.000,00 at sight.
2. It was bought in jan, 2013.
3. Since jan, 2016 the owner monthly received RS 1.400,00 for rent.
4. The purchase proposal amount was R$ 500.000,00.

To bring all this amounts to present value i'll use accumulated IPCA and savings accounts proftability. 

LET'S GO FOR IT. 

``` ruby
#### Import libraries #### 

## for data
import pandas as pd
import numpy as np
import datetime 

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import missingno as msno

## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm

## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
import xgboost
```

I downloaded IPCA historical series at IBGE's website: https://sidra.ibge.gov.br/tabela/1737

``` ruby
# read data
dtf = pd.read_excel("IPCA_acumulado_2013-2022.xlsx")
dtf = dtf[3:114]
dtf.rename(columns = {'Tabela 1737 - IPCA - Série histórica com número-índice, variação mensal e variações acumuladas em 3 meses, em 6 meses, no ano e em 12 meses (a partir de dezembro/1979)' : 'Month', 'Unnamed: 1' : 'IPCA_var'},
          inplace = True)
display(dtf)
```

<p align="center">
  <img src=images/data_display.png?raw=true />
</p>

Dates were written in portuguese, so i'll replace them into numbers and change their dtypes and set "Month" columns as index (for me it facilitates time series visualization)

``` ruby
# change months names to numbers
dtf["Month"] = dtf["Month"].str.replace("janeiro", "01")
dtf["Month"] = dtf["Month"].str.replace("fevereiro", "02")
dtf["Month"] = dtf["Month"].str.replace("março", "03")
dtf["Month"] = dtf["Month"].str.replace("abril", "04")
dtf["Month"] = dtf["Month"].str.replace("maio", "05")
dtf["Month"] = dtf["Month"].str.replace("junho", "06")
dtf["Month"] = dtf["Month"].str.replace("julho", "07")
dtf["Month"] = dtf["Month"].str.replace("agosto", "08")
dtf["Month"] = dtf["Month"].str.replace("setembro", "09")
dtf["Month"] = dtf["Month"].str.replace("outubro", "10")
dtf["Month"] = dtf["Month"].str.replace("novembro", "11")
dtf["Month"] = dtf["Month"].str.replace("dezembro", "12")
dtf["Month"] = dtf["Month"].str.replace(" ", "/")
dtf
```

<p align="center">
  <img src=images/data_display2.png?raw=true />
</p>

``` ruby
# devide IPCA_var per 100 to get percentage of variation
dtf["IPCA_var"] = dtf["IPCA_var"]/100

# set date column as a datetime format
dtf["Month"] = pd.to_datetime(dtf["Month"], format = "%m/%Y")

# set "Month" columns as index
dtf = dtf.set_index("Month")
```

Firts i want to visualize IPCA variation since 2013. I like to use Plotly for visualizations becouse of it's interability.

``` ruby
# Plot IPCA monthly variation (Manualy customized)
fig = px.line(dtf, x = dtf.index, y = dtf["IPCA_var"], markers = True)
fig['data'][0]['line']['color'] = 'RebeccaPurple'
fig['data'][0]['line']['width'] = 3

fig.update_layout(title = 'IPCA variation (2013 - 2022)',
                  title_x = 0.1,
                  title_font = dict(color = "white", size = 25),
                  plot_bgcolor = 'black',
                  paper_bgcolor = 'black',
                  font_family = "Arial")

fig.update_xaxes(title = '',
                 title_font=dict(size=18, color = "white"),
                 nticks = 10,
                 tickfont = dict(color = "RebeccaPurple"),
                 #linewidth = 2, linecolor = 'white',
                showgrid = False)

fig.update_yaxes(title = '', 
                 title_font=dict(size=18, color = "white"),
                 tickfont = dict(color = 'RebeccaPurple'),
                 gridcolor = 'black', zerolinecolor = "red", zerolinewidth = 2,
                 showgrid = False)

fig.add_shape(type = "line", line_color = "purple", line_width = 3, opacity = 1, line_dash = "dashdot",
              x0 = 0, x1 = 1, xref="paper", y0 = dtf["IPCA_var"].mean(), y1 = dtf["IPCA_var"].mean(), yref="y")

fig.add_shape(type = "line", line_color = "salmon", line_width = 3, opacity = 1, line_dash = "dot",
              x0 = dtf[dtf["IPCA_var"] == dtf["IPCA_var"].max()].index[0], 
              x1 = dtf[dtf["IPCA_var"] == dtf["IPCA_var"].max()].index[0], 
              xref="x", 
              y0 = 0, 
              y1 = dtf["IPCA_var"].max(), 
              yref="y")

# mean
fig.add_annotation(x = '2022-11-01 00:00:00', y = dtf["IPCA_var"].mean(),
                   text = "Mean (%)",
                   showarrow = True,                   
                   font = dict(color = "white"),
                   align="center",
                   arrowhead=2,
                   arrowsize=1,
                   arrowwidth=2,
                   arrowcolor="RebeccaPurple",
                   ax=20,
                   ay=-30,
                   bordercolor="Purple",
                   borderwidth=2,
                   borderpad=4,
                   bgcolor="RebeccaPurple",
                   opacity=0.8)

# max
fig.add_annotation(x = dtf[dtf["IPCA_var"] == dtf["IPCA_var"].max()].index[0], 
                   y = dtf["IPCA_var"].max(),
                   text = "Maximum",
                   showarrow = True,                   
                   font = dict(color = "white"),
                   align="center",
                   arrowhead=2,
                   arrowsize=1,
                   arrowwidth=2,
                   arrowcolor="RebeccaPurple",
                   ax=20,
                   ay=-30,
                   bordercolor="Purple",
                   borderwidth=2,
                   borderpad=4,
                   bgcolor="RebeccaPurple",
                   opacity=0.8)




fig.show()
``` 

<p align="center">
  <img src=images/IPCA_var_2013.png?raw=true />
</p>

``` ruby
# duplicate dataset, filtering the first since 2013 and the second since 2016
dtf_since2013 = dtf.copy()
dtf = dtf[dtf.index >= datetime.datetime(2016,1,1)]
```

To be able to bring the amount spent on buying the apartment at 2013, i'll calculate the accumulated inflation (2013 - 2022) and the accumulated savings accounts rate (2013 - 2022).

it can be calculated as follows:



(1+𝑖1)(1+𝑖2)...(1+𝑖𝑛)−1



i = variation rate for each month
n = number of periods
    
``` ruby
# add factor column (2013)
dtf_since2013["factor"] = dtf_since2013["IPCA_var"] + 1

# calculate accumulated inflation since 2013
product_since2013 = 1

for i in dtf_since2013["factor"]:
    product_since2013 = product_since2013 * i

product_since2013 = product_since2013 - 1    
print('Accumulated inflation', round((product_since2013)*100, 2), '% since 2013')
```
output: 

          Accumulated inflation 75.32 % since 2013
        
``` ruby
# Calculate accumulated savings accounts rate since 2013
print("Accumulated savings accounts rate of ",round(((1.0637*1.0716*1.0815*1.0830*1.0661*1.0462*1.0426*1.0211*1.0294) - 1)*100, 2), " % since 2013")

savings_acc_rate = (1.0637*1.0716*1.0815*1.0830*1.0661*1.0462*1.0426*1.0211*1.0294) - 1
```
output:

          Accumulated savings accounts rate of  63.19  % since 2013
        
Now i'll repeat the process considering 2016 to 2022

``` ruby
# add factor column since 2016
dtf["factor"] = dtf["IPCA_var"] + 1

# calculate accumulated inflation since 2016
product = 1

for i in dtf["factor"]:
    product = product * i

product = product - 1    
print('Accumulated inflation', round((product)*100, 2), '%')
```
output:

          Accumulated inflation 40.57 %
        
With rates calculated, to bring the investment to Present Value i just have to follow the equation:  

(1+𝑅2013)𝑉 

R2013: accumulated inflation since 2013
V: initial amount
``` ruby
# calculate Present Value of investment (R$ 400.000,00)
inv_PV = (1 + product_since2013)*400000
```

To calculate IPCA mean by month since 2016:
                                                                 
(1+𝑅2016)1/𝑛−1
 
R2016 = accumulated inflation since 2016
n = number of periods

``` ruby
# calculte everage IPCA rate since 2016
IPCA_mean = ((product + 1)**(1/len(dtf)))-1
print(round(IPCA_mean*100, 4), "% ao mês")
```
output:

          R$  124807.46 reais brought to PV
        
Let's create a accumulated rent column withs values in PV to visualize total rent evolution

``` ruby
# create accumulated IPCA column
ac_IPCA = []
for i in list(range(0, len(dtf))):
    rent_product = 1

    for r in dtf.iloc[0:(i+1), 1]:
        rent_product = rent_product * r
         
    rent_product = rent_product - 1
    ac_IPCA.append(rent_product)  
    
 dtf["IPCA_accumulated"] = ac_IPCA
 
 # create a corretion PV indice
ind_PV = []

for i in list(range(0, len(dtf))):
    rent_product = 1

    for r in dtf.iloc[i:len(dtf), 1]:
        rent_product = rent_product * r
         
    rent_product = rent_product - 1
    ind_PV.append(rent_product)     

dtf["ind_PV"] = ind_PV

# Calculate Present Value for every month since 01/2013
rent = []

for i in list(range(0, len(dtf))):
    x = 1400*(1+dtf.loc[:, "ind_PV"][i])
    rent.append(x)    
    
dtf["rent_PV"] = rent

# create accumulated rent in present value
acc_rent = []

for i in list(range(0, len(dtf))):
    acc = dtf.loc[:, 'rent_PV'][0:(i + 1)].sum()
        
    acc_rent.append(acc)
    
dtf["acc_rent"] = acc_rent

dtf["amount"] = dtf["acc_rent"] + 500000
```

All set to plot to see the amount with IPCA correction X savings account correction X proposal plus rents whith IPCA correction

``` ruby
# Plot IPCA monthly variation (using templates)
fig = px.line(dtf, x = dtf.index, y = "amount", markers = False, template = "plotly_dark")
#fig['data'][0]['line']['color'] = 'salmon'
#fig['data'][0]['line']['width'] = 3
fig['data'][0]['line']['color'] = 'darkseagreen'
fig['data'][0]['line']['width'] = 3

# investment PV value annotation
fig.add_annotation(x = dtf.index.max(), 
                   y = inv_PV,
                   text = ("Investment at PV Value: R$ " + str(round(inv_PV, 2))),
                   showarrow = True,                   
                   font = dict(color = "white"),
                   align="center",
                   arrowhead=2,
                   arrowsize=1,
                   arrowwidth=2,
                   arrowcolor="RebeccaPurple",
                   ax= -135,
                   ay= 50,
                   bordercolor="Purple",
                   borderwidth=2,
                   borderpad=4,
                   bgcolor="RebeccaPurple",
                   opacity=0.8)

# Selling value annotation
fig.add_annotation(x = dtf[dtf["amount"] == dtf["amount"].max()].index[0], 
                   y = dtf["amount"].max(),
                   text = ('Selling Value (Price + PV rents): R$ ' + str(round(dtf.loc[dtf["amount"] == dtf["amount"].max(), "amount"][0], 2))),
                   showarrow = True,                   
                   font = dict(color = "white"),
                   align="center",
                   arrowhead=2,
                   arrowsize=1,
                   arrowwidth=2,
                   arrowcolor="RebeccaPurple",
                   ax= -104,
                   ay= 100,
                   bordercolor="Purple",
                   borderwidth=2,
                   borderpad=4,
                   bgcolor="RebeccaPurple",
                   opacity=0.8)

# Saving account annotation
fig.add_annotation(x = dtf.index.mean(), 
                   y = (savings_acc_rate + 1)*400000,
                   text = ('PV value based on savings accounts profitability: ' + str(round((savings_acc_rate + 1)*400000))),
                   showarrow = True,                   
                   font = dict(color = "white"),
                   align="center",
                   arrowhead=2,
                   arrowsize=1,
                   arrowwidth=2,
                   arrowcolor="RebeccaPurple",
                   ax= -104,
                   ay= 100,
                   bordercolor="Purple",
                   borderwidth=2,
                   borderpad=4,
                   bgcolor="RebeccaPurple",
                   opacity=0.8)

# IPCA line
fig.add_shape(type = "line", line_color = "lightblue", line_width = 3, opacity = 1, line_dash = "dot",
              x0 = dtf.index.min(), 
              x1 = dtf.index.max(), 
              xref="x", 
              y0 = inv_PV, 
              y1 = inv_PV, 
              yref="y")

# savings line
fig.add_shape(type = "line", line_color = "red", line_width = 3, opacity = 1, line_dash = "dot",
              x0 = dtf.index.min(), 
              x1 = dtf.index.max(), 
              xref="x", 
              y0 = (savings_acc_rate + 1)*400000, 
              y1 = (savings_acc_rate + 1)*400000, 
              yref="y")


fig.update_layout(title = "Apartment porftability VS inflation and savings accounts",
                  title_x = 0.1,
                  title_font = dict(size = 18),
                  font_family = "arial")

fig.update_xaxes(title = "",
                 showgrid = False)

fig.update_yaxes(title = "(R$) at Present Value",
                 showgrid = False)
fig.show()

```

<p align="center">
  <img src=images/Final_plot.png?raw=true />
</p>

#### Chart explanation and conclusion

Blue line: R$ 400.000,00 brought to present value whith IPCA (2013 - 2022). I used this indicator to simulate the same amount invested in some governmet bond that tracks IPCA since 2013.
    
Red line: 400.000,00 brought to present value whith savings accounts proftability (2013 - 2022). Another category of investment to compare with.
    
Green line: represents the purchase proposal (500.000,00) added by rents montly received since 2016 and corrected by IPCA to Present Value. 

*Obs: The apartment value at 2016 already starts at R$ 500.000,00 assuming that it is it's Present Value at 2022 and it grew monthly with the receipt of rents.*
    

I used two options of low risk investment to compare with the apartment bought in 2013 because, in my opinion, it is also a low risk investiment. 
 
As you can see, considering the three options, buying the apartment was the worst one. It didn't even reach savings accounts proftability. A selling price that matches IPCA correction today would be R$ 572.519,59, but this is way too far from today's real state market price. 

Considering that high inflation rates are expected in Brazil for the next years, despite the money loss, i recommended selling the apartment and investing the amount in another option. 




