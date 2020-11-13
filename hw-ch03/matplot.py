import pandas as pd
import matplotlib.pyplot as plt
import random

#------plot------

df = pd.read_csv("Coronadata.csv", index_col=0)

rand1 = random.randint(0, 186)
rand2 = rand1
if rand1 < 5:
    rand1 += 5
    df = df[rand2:rand1]
else:
    rand1 -= 5
    df = df[rand1:rand2]

print(df)

plot = df.plot()

plot.set_xlabel("Country")
plot.set_ylabel("Corona Confirmed Number")
plt.title("#Corona Virus Data")
plt.show()


#---------bar charts---------

xs = [i+0.1 for i, _ in enumerate(df.index)]
plt.bar(xs, df['3/31/20'])
plt.ylabel("#Corona Confirmed Number")
plt.xlabel('Country')
plt.title("Corona Virus Data")

plt.xticks([i+0.1 for i, _ in enumerate(df.index)], df.index)

plt.show()

#------Scatterplot------


plt.scatter(df.index, df['3/31/20'])
plt.xlabel('Country')
plt.ylabel('Corona Confirmed Number')
plt.title('#Corona Confirmed Data')

plt.show()