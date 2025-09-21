1. **CUSTOM TOTAL LOSS:**

&nbsp;	total\_loss = (0.5 \* bb\_loss) + (0.1 \* cls\_loss) + (0.1 \* ob\_loss)



**2. LEARNING RATE - POLYNOMIAL DECAY:**

&nbsp;	learning\_rate = PolynomialDecay(

&nbsp;   		initial\_learning\_rate= 1e-3,

&nbsp;   		decay\_steps= 20000,

&nbsp;  		end\_learning\_rate= 1e-6,

&nbsp;   		power=2,

&nbsp;   		name= "learning\_rate"

&nbsp;	)

