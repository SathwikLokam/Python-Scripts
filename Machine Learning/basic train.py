x=[2,3,5,8]
y=[40,50,70,100]
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()
def mul(x,y):
    li=[]
    for i in range(len(x)):
        li.append(x[i]*y[i])
    return li

def squ(x):
    li=[]
    for i in x:
        li.append(i*i)
    return li        

b=(len(x)*(sum(mul(x,y)))-((sum(x))*(sum(y))))/((len(x)*(sum(squ(x))))-(sum(x)*sum(x)))
a=(sum(y)-(b*sum(x)))/(len(x))
print(a,b)

print(a+(b*int(input("Enter the number to guess : "))))