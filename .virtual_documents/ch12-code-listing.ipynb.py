get_ipython().run_line_magic("matplotlib", " inline")
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import pandas as pd


import matplotlib as mpl
mpl.style.use('ggplot')


s = pd.Series([909976, 8615246, 2872086, 2273305, 1200000, 450000])
s


s.dtype


s.index


s.values


s.index = ["Stockholm", "London", "Rome", "Paris", 'La Paz', 'Palma']


s.name = "Population"
s


s = pd.Series([909976, 8615246, 2872086, 2273305, 1200000, 450000], 
              index=["Stockholm", "London", "Rome", "Paris", 'La Paz', 'Palma'], name="Population")


s["London"]


s.Stockholm


s[["Paris", "Rome"]]


s.median(), s.mean(), s.std()


s.min(), s.max()


s.quantile(q=0.25), s.quantile(q=0.5), s.quantile(q=0.75)


s.describe()


fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

s.plot(ax=axes[0], kind='line', title="line")
s.plot(ax=axes[1], kind='bar', title="bar")
s.plot(ax=axes[2], kind='box', title="box")
s.plot(ax=axes[3], kind='pie', title="pie")

fig.tight_layout()
# fig.savefig("ch12-series-plot.pdf")
# fig.savefig("ch12-series-plot.png")


df = pd.DataFrame([[909976, 8615246, 2872086, 2273305, 4600000, 100000],
                   ["Sweden", "United kingdom", "Italy", "France", 'Spain', 'Bolivia']])
df


df = pd.DataFrame([[909976, "Sweden"],
                   [8615246, "United kingdom"], 
                   [2872086, "Italy"],
                   [2273305, "France"]])
df


df.index = ['Population','Country']


df.columns = ["Sweden", "United kingdom", "Italy", "France", 'Spain', 'Bolivia']


df


df = pd.DataFrame([[909976, "Sweden"],
                   [8615246, "United kingdom"], 
                   [2872086, "Italy"],
                   [2273305, "France"]],
                  index=["Stockholm", "London", "Rome", "Paris"],
                  columns=["Population", "State"])


df = pd.DataFrame({"Population": [909976, 8615246, 2872086, 2273305],
                   "State": ["Sweden", "United kingdom", "Italy", "France"]},
                  index=["Stockholm", "London", "Rome", "Paris"])


df


df.index


df.columns


df.values


df.Population


type(df.Population)


df.Population.Stockholm


type(df.index)


df.loc["Stockholm"]


type(df.loc["Stockholm"])


df.loc[["Paris", "Rome"]]


df.loc[["Paris", "Rome"], "Population"]


df.loc["Paris", "Population"]


df.Population.mean()


df.info()


df.dtypes


df.head()


s.head()


fig, axes = plt.subplots(1, 4, figsize=(12, 3))
s.plot(ax=axes[0], kind='line', title='line')
s.plot(ax=axes[1], kind='bar', title='bar')
s.plot(ax=axes[2], kind='box', title='box')
s.plot(ax=axes[3], kind='pie', title='pie')


fig, axes = plt.subplots(1, 3, figsize=(12, 3))
df.plot(ax=axes[0], kind='line', title='line')
df.plot(ax=axes[1], kind='bar', title='bar')
df.plot(ax=axes[2], kind='box', title='box')
# df.plot(ax=axes[3], kind='pie', title='pie')


get_ipython().getoutput("head -n5 /home/rob/datasets/european_cities.csv")


df_pop = pd.read_csv("data/european.csv")


df_pop.head()


df_pop = pd.read_csv("data/european.csv", delimiter=",", encoding="utf-8", header=0)


df_pop.shape


df_pop.info()


df_pop.head()


df_pop["Population"] = df_pop.Population.apply(lambda x: int(x.replace(",", "")))


df_pop["State"].values[:3]


df_pop["State"] = df_pop["State"].apply(lambda x: x.strip())


df_pop.head()


df_pop.dtypes


df_pop2 = df_pop.set_index("City")


df_pop2 = df_pop2.sort_index()


df_pop2.head()


df_pop3 = df_pop.set_index(["State", "City"]).sort_index(level=0)


df_pop3.head(7)


df_pop3.loc["Sweden"]


df_pop3.loc[("Sweden", "Gothenburg")]


df_pop.set_index("City").sort_values(["State", "Population"], ascending=[True, True]).head()


city_counts = df_pop.State.value_counts()


city_counts.name = "# cities in top 105"
city_counts


df_pop3 = df_pop[["State", "City", "Population"]].set_index(["State", "City"])


df_pop4 = df_pop3.groupby("State").sum().sort_values("Population", ascending=False)


df_pop4.head()


df_pop5 = (df_pop.drop("Rank", axis=1)
                 .groupby("State").sum()
                 .sort_values("Population", ascending=False))


df_pop5.head()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

city_counts.plot(kind='barh', ax=ax1)
ax1.set_xlabel("# cities in top 105")

df_pop5.Population.plot(kind='barh', ax=ax2)
ax2.set_xlabel("Total pop. in top 105 cities")

fig.tight_layout()
# fig.savefig("ch12-state-city-counts-sum.pdf")


import datetime


pd.date_range("2015-1-1", periods=31)


pd.date_range(datetime.datetime(2015, 1, 1), periods=31)


pd.date_range("2015-1-1 00:00", "2015-1-1 23:00", freq="H")


ts1 = pd.Series(np.arange(31), index=pd.date_range("2015-1-1", periods=31))


ts1.head()


ts1["2015-1-3"]


ts1.index[2]


ts1.index[2].year, ts1.index[2].month, ts1.index[2].day


ts1.index[2].nanosecond


ts1.index[2].to_pydatetime()


ts2 = pd.Series(np.random.rand(31), 
                index=pd.date_range("2015-8-1", periods=31))


ts2


periods = pd.PeriodIndex([pd.Period('2015-01'), pd.Period('2015-02'), pd.Period('2015-03')])


ts3 = pd.Series(np.random.rand(3), periods)


ts3


ts3.index


ts2.to_period('M')


pd.date_range("2015-1-1", periods=12, freq="M").to_period()


# get_ipython().getoutput("head -n 5 temperature_outdoor_2014.tsv")


df1 = pd.read_csv('data/out_temp.tsv', delimiter="\t", names=["time", "outdoor"])


df2 = pd.read_csv('data/int_temp.tsv', delimiter="\t", names=["time", "indoor"])


df1.head()


df2.head()


df1.time = (pd.to_datetime(df1.time.values, unit="s")
              .tz_localize('UTC').tz_convert('Europe/Stockholm'))


df1 = df1.set_index("time")


df2.time = (pd.to_datetime(df2.time.values, unit="s")
              .tz_localize('UTC').tz_convert('Europe/Madrid'))


df2 = df2.set_index("time")


df1.head()


df2.head()


df1.index[0]


fig, ax = plt.subplots(1, 1, figsize=(12, 4))
df1.plot(ax=ax)
df2.plot(ax=ax)

fig.tight_layout()
# fig.savefig("ch12-timeseries-temperature-2014.pdf")


# select january data


df1.info()


df1_jan = df1[(df1.index > "2014-8-1") & (df1.index < "2014-10-1")]


df1.index < "2014-10-1"


df1_jan.info()


df2_jan = df2["2014-8-1":"2014-9-30"]


fig, ax = plt.subplots(1, 1, figsize=(12, 4))

df1_jan.plot(ax=ax)
df2_jan.plot(ax=ax)

fig.tight_layout()
# fig.savefig("ch12-timeseries-selected-month.pdf")


# group by month


df1_month = df1.reset_index()


df1_month["month"] = df1_month.time.apply(lambda x: x.month)


df1_month.head()


df1_month = df1_month.groupby("month").aggregate(np.mean)


df2_month = df2.reset_index()


df2_month["month"] = df2_month.time.apply(lambda x: x.month)


df2_month = df2_month.groupby("month").aggregate(np.mean)


df_month = df1_month.join(df2_month)


df_month.head(3)


df_month = pd.concat([df.to_period("M").groupby(level=0).mean() for df in [df1, df2]], axis=1)


df_month.head(3)


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df_month.plot(kind='bar', ax=axes[0])
df_month.plot(kind='box', ax=axes[1])

fig.tight_layout()
fig.savefig("ch12-grouped-by-month.pdf")


df_month


# resampling


df1_hour = df1.resample("H").mean()


df1_hour.columns = ["outdoor (hourly avg.)"]


df1_day = df1.resample("D").mean()


df1_day.columns = ["outdoor (daily avg.)"]


df1_week = df1.resample("7D").mean()


df1_week.columns = ["outdoor (weekly avg.)"]


df1_month = df1.resample("M").mean()


df1_month.columns = ["outdoor (monthly avg.)"]


# df1.resample("D")


df_diff = (df1.resample("D").mean().outdoor - df2.resample("D").mean().indoor)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

df1_hour.plot(ax=ax1, alpha=0.25)
df1_day.plot(ax=ax1)
df1_week.plot(ax=ax1)
df1_month.plot(ax=ax1)

df_diff.plot(ax=ax2)
ax2.set_title("temperature difference between outdoor and indoor")

fig.tight_layout()
fig.savefig("ch12-timeseries-resampled.pdf")


pd.concat([df1.resample("5min").mean().rename(columns={"outdoor": 'None'}),
           df1.resample("5min").ffill().rename(columns={"outdoor": 'ffill'}),
           df1.resample("5min").bfill().rename(columns={"outdoor": 'bfill'})], axis=1).head()


df1_dec25 = df1[(df1.index < "2014-9-1") & (df1.index >= "2014-8-1")].resample("D")


df1_dec25 = df1.loc["2014-12-25"]


df1_dec25.head(5)


df2_dec25 = df2.loc["2014-12-25"]


df2_dec25.head(5)


df1_dec25.describe().T


fig, ax = plt.subplots(1, 1, figsize=(12, 4))

df1_dec25.plot(ax=ax)

fig.savefig("ch12-timeseries-selected-month.pdf")


df1.index


sns.set(style="darkgrid")


#sns.set(style="whitegrid")


df1 = pd.read_csv('temperature_outdoor_2014.tsv', delimiter="\t", names=["time", "outdoor"])
df1.time = pd.to_datetime(df1.time.values, unit="s").tz_localize('UTC').tz_convert('Europe/Stockholm')

df1 = df1.set_index("time").resample("10min").mean()
df2 = pd.read_csv('temperature_indoor_2014.tsv', delimiter="\t", names=["time", "indoor"])
df2.time = pd.to_datetime(df2.time.values, unit="s").tz_localize('UTC').tz_convert('Europe/Stockholm')
df2 = df2.set_index("time").resample("10min").mean()
df_temp = pd.concat([df1, df2], axis=1)


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
df_temp.resample("D").mean().plot(y=["outdoor", "indoor"], ax=ax)
fig.tight_layout()
fig.savefig("ch12-seaborn-plot.pdf")


#sns.kdeplot(df_temp["outdoor"].dropna().values, shade=True, cumulative=True);


sns.distplot(df_temp.to_period("M")["outdoor"]["2014-04"].dropna().values, bins=50);
sns.distplot(df_temp.to_period("M")["indoor"]["2014-04"].dropna().values, bins=50);

plt.savefig("ch12-seaborn-distplot.pdf")


with sns.axes_style("white"):
    sns.jointplot(df_temp.resample("H").mean()["outdoor"].values,
                  df_temp.resample("H").mean()["indoor"].values, kind="hex");
    
plt.savefig("ch12-seaborn-jointplot.pdf")


sns.kdeplot(df_temp.resample("H").mean()["outdoor"].dropna().values,
            df_temp.resample("H").mean()["indoor"].dropna().values, shade=False);

plt.savefig("ch12-seaborn-kdeplot.pdf")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

sns.boxplot(df_temp.dropna(), ax=ax1, palette="pastel")
sns.violinplot(df_temp.dropna(), ax=ax2, palette="pastel")

fig.tight_layout()
fig.savefig("ch12-seaborn-boxplot-violinplot.pdf")


sns.violinplot(x=df_temp.dropna().index.month, y=df_temp.dropna().outdoor, color="skyblue");

plt.savefig("ch12-seaborn-violinplot.pdf")


df_temp["month"] = df_temp.index.month
df_temp["hour"] = df_temp.index.hour


df_temp.head()


table = pd.pivot_table(df_temp, values='outdoor', index=['month'], columns=['hour'], aggfunc=np.mean)


table


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.heatmap(table, ax=ax);

fig.tight_layout()
fig.savefig("ch12-seaborn-heatmap.pdf")


get_ipython().run_line_magic("reload_ext", " version_information")


get_ipython().run_line_magic("version_information", " numpy, matplotlib, pandas, seaborn")






