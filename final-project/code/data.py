import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

def load_data():
    data = "kc_house_data.csv"
    return pd.read_csv(data, header=0)
def describe_data(df):
    print df.describe()

def plot_data(x, y, filename):
    plt.scatter(x=x, y=y)
    plt.savefig("img/" + filename)
    plt.clf()