from utils import clean_math_output
import re

text = r'''Y_i(t+\Delta t)=Y_i(t)\exp\Bigl(-\frac{\Delta t}{\tau_a}\Bigr) +\psi\bigl(S(X_i(t))\bigr)\Bigl \[1-\exp\!\Bigl(-\frac{\Delta t}{\tau_a}\Bigr)\Bigr\] + ...'''
print("Original:")
print(repr(text))
print("Cleaned:")
print(repr(clean_math_output(text)))

text2 = r'''p_i(t)=\operatorname{ReLU}\Bigl \[ \lambda_0+\lambda_0\frac{1+2\tau_a}{\tau_a}\bigl(Y_i(t)-\psi(S(X_i(t)))\bigr)\Bigr \] \Delta t'''
print("Cleaned 2:")
print(repr(clean_math_output(text2)))
