这几个验证指标都是深度估计里常用的，下面我用简单直白的话总结一下每个怎么算：

1. **d1、d2（阈值准确率）**

   * 看预测深度和真实深度的比值，要求它们差得不能太多。
   * 定义：算比例

     $$
     \delta = \max\left(\frac{\hat{d}}{d}, \frac{d}{\hat{d}}\right)
     $$
   * d1 = 预测中有多少比例满足 δ < 1.25
   * d2 = 预测中有多少比例满足 δ < 1.25²

   （所以 d1、d2 越高越好）

2. **AbsRel（绝对相对误差）**

   * 算预测和真实的差值占真实值的比例，取平均。

   $$
   \text{AbsRel} = \frac{1}{N}\sum \frac{|\hat{d} - d|}{d}
   $$

   （越小越好）

3. **RMSE（均方根误差）**

   * 先算预测和真实的差值平方，再取平均，最后开根号。

   $$
   \text{RMSE} = \sqrt{\frac{1}{N}\sum (\hat{d} - d)^2}
   $$

   （越小越好）

4. **MAE（平均绝对误差）**

   * 就是差值的绝对值取平均。

   $$
   \text{MAE} = \frac{1}{N}\sum |\hat{d} - d|
   $$

   （比 RMSE 更直观）

5. **SILog（尺度不变对数误差）**

   * 先把预测和真实的比值取 log，然后看方差。

   $
   \text{SILog} = \sqrt{\frac{1}{N}\sum \delta_i^2 - \left(\frac{1}{N}\sum \delta_i\right)^2}
   $

   其中 $\delta_i = \log \hat{d}_i - \log d_i$。
   （强调相对误差，越小越好）

