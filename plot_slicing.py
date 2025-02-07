import pandas as pd
import matplotlib.pyplot as plt

filepath = 'autosave/slicing.dat'
experiment_name = '124r'

#sigma/ro, ErrMin, Contact angle, Contact diameter, Apex radius, Point count, Surface area, Volume, Iteration,
#IsPendant, File created, File processed, Process outcome, File path, Optional info, Optional param
df = pd.read_csv(filepath, delimiter=', ', encoding='ISO-8859-1', parse_dates=['File created','File processed'], dayfirst=True, engine='python')

expfilter = df['File path'].str.contains(experiment_name)
df = df.where(expfilter)
df = df[df['Optional info'] == 'slice%']
df = df[df['Iteration'] > 2]
df = df[df['sigma/ro'] < 200]
df = df[df['sigma/ro'] > -1500]
df = df[df['Optional param'] < 70]
#df['n'] = df['File path'].str.split('r')[-1]
print(df)

files = list(df['File path'].drop_duplicates())
fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained', figsize=(12, 9))
ax1.set_xlabel('slice%')
ax1.set_ylabel('Sigma/ro', color='blue')
ax2.set_ylabel('Error', color='red')
for file in files:
    df_part = df[df['File path'] == file]
    df_part.sort_values(by='Optional param', inplace=True)
    ax1.plot(df_part['Optional param'], -df_part['sigma/ro'], lw=1.5)
    ax2.plot(df_part['Optional param'], df_part['ErrMin'], lw=1.5)
ax1.legend(files)
ax1.grid(True)
ax2.grid(True)
plt.show()
