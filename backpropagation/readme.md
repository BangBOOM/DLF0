# Chain Rule for One Element

## The function

$$ \hat{y} = \sigma(\omega_2 * \sigma(\omega_1*x + b_1) + b_2)$$

$$ z_1 = \omega_1*x + b_1$$

$$ a_1 = \sigma(z_1)$$

$$ z_2 = \omega_2*z_1 + b_1 $$

$$ a_2 = \sigma(z_2)$$

$$ e = (y - \hat{y})^2 = (y - a_2)^2 $$

## First layer

$$\frac{\partial e}{\partial w_2} = \frac{\partial e}{\partial a_2}
\cdot \frac{\partial a_2}{\partial w_2}$$

$$\frac{\partial e}{\partial w_2} = \frac{\partial e}{\partial a_2}
\cdot \frac{\partial a_2}{\partial z_2}
\cdot \frac{\partial z_2}{\partial w_2} $$

$$\frac{\partial e}{\partial a_1} = \frac{\partial e}{\partial a_2}
\cdot \frac{\partial a_2}{\partial z_2}
\cdot \frac{\partial z_2}{\partial a_1} $$

## Second layer

$$\frac{\partial e}{\partial w_1} = \frac{\partial e}{\partial a_1} 
\cdot \frac{\partial a_1}{\partial w_1}$$

$$\frac{\partial a_1}{\partial w_1} = \frac{\partial a_1}{\partial z_1} 
\cdot \frac{\partial z_1}{\partial w_1}$$
