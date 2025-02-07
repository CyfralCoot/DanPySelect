import pandas as pd
import matplotlib.pyplot as plt

filepath = 'autosave/autosave.dat'
experiment_name = '125r'

#sigma/ro, ErrMin, Contact angle, Contact diameter, Apex radius, Point count, Surface area, Volume, Iteration, IsPendant, File creation time, File processing time, File path
df = pd.read_csv(filepath, delimiter=', ', encoding='ISO-8859-1', parse_dates=['File created','File processed'], dayfirst=True, engine='python')

df = df.sort_values('File created')
#09:01:36 27.07.2024 Sat

expfilter = df['File path'].str.contains(experiment_name)
df = df.where(expfilter)
df = df[df['Iteration'] > 2]
df = df[abs(df['sigma/ro']) < 2000]
df = df[df['ErrMin'] < 60]
#df['n'] = df['File path'].str.split('r')[-1]
print(df)

fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained', figsize=(12, 9))
ax1.set_xlabel('Recording time')
ax1.set_ylabel('Sigma/ro', color='blue')
ax1.plot(df['File created'], -df['sigma/ro'], c='blue', lw=1)
if df['IsPendant'].iloc[0] == False:
    ax1.set_ylabel('Contact angle', color='green')
    ax1.plot(df['File created'], df['Contact angle'], c='green', lw=1)

ax2.set_ylabel('Error', color='red')
ax2.plot(df['File created'], df['ErrMin'], c='red', lw=1)
plt.show()
