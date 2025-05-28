from matplotlib import pyplot as plt
from sympy import symbols, Abs, init_printing, simplify
from sympy.plotting import plot_implicit
init_printing(use_latex=True)
plt.close('all')
x,y = symbols("x y")
def ir(x,y):
 return (x+y-Abs(x-y))/2
w1=y
w2=x
w3=1-x**2-y**2
w12=ir(w1,w2)
w123=ir(w12,w3)
plot_implicit(w123>0,(x,-0.1,1.1),(y,-0.1,1.1),n=200,
 aspect=(1,1),border_color="k",color='gray',grid=False)