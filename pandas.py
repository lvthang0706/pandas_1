import pandas as pd
import numpy as np
import math

url = 'https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv'
data = pd.read_csv(url)

x = data.iloc[:, 1]
y = data.iloc[:, 4]

cov_matrix = np.cov(x, y)
cov_value = cov_matrix[0][1]

var_x = np.var(x, ddof=1)
var_y = np.var(y, ddof=1)

person_corr = np.corrcoef(x,y)[0,1]

theta_radians = math.acos(person_corr)
theta_degrees = math.degrees(theta_radians)

result = {
    "Covariance": cov_value,
    "Variance TV": var_x,
    "Variance Sales": var_y,
    "Correlation": person_corr,
    "Theta_radians": theta_radians,
    "Theta (Degrees)": theta_degrees,
}
print(result)