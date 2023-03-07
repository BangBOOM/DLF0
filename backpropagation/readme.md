# Chain Rule for One Element

## first layer

$$\frac{\partial e}{\partial w_2} = \frac{\partial e}{\partial a_2}
\cdot \frac{\partial a_2}{\partial w_2}$$

$$\frac{\partial e}{\partial w_2} = \frac{\partial e}{\partial a_2}
\cdot \frac{\partial a_2}{\partial z_2}
\cdot \frac{\partial z_2}{\partial w_2} $$

$$\frac{\partial e}{\partial a_1} = \frac{\partial e}{\partial a_2}
\cdot \frac{\partial a_2}{\partial z_2}
\cdot \frac{\partial z_2}{\partial a_1} $$

## second layer

$$\frac{\partial e}{\partial w_1} = \frac{\partial e}{\partial a_1} 
\cdot \frac{\partial a_1}{\partial w_1}$$

$$\frac{\partial a_1}{\partial w_1} = \frac{\partial a_1}{\partial z_1} 
\cdot \frac{\partial z_1}{\partial w_1}$$