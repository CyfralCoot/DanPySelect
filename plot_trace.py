import pandas as pd
import numpy as np
import matplotlib
default_backend = matplotlib.get_backend()
import matplotlib.pyplot as plt
import glob
import os
from io import StringIO

def get_normalized_stringio(filepath, del_iters=4):
    """Reads file and outputs string io to feed into pandas"""
    with open(filepath, 'r') as file:
        mystring = file.read()
    for i in range(del_iters):#delete spaces up to 16 long by default
        mystring = mystring.replace('  ', ' ')
    return StringIO(mystring)

def read_file(filepath, names=['x','y','type']):
    data = get_normalized_stringio(filepath)
    df = pd.read_csv(data, delimiter=' ')
    df.dropna(inplace=True,axis=1,how='all')
    df.rename(inplace=True,columns={df.columns[0]:names[0],df.columns[1]:names[1],df.columns[2]:names[2]})
    return df

def read_final_file(filepath, names=['x','y']):
    """read trace.dat"""
    data = get_normalized_stringio(filepath, 3)
    
    df = pd.read_csv(data, delimiter=' ', skipfooter=5)
    df.dropna(inplace=True,axis=1,how='all')
    df.rename(inplace=True,columns={df.columns[0]:names[0],df.columns[1]:names[1]})
    return df

def plot(ooo_path='C://ООО//Select', pendant=True, dif_types=True, save=False, metadata=None):
    trace_file_paths = glob.glob(os.path.join(ooo_path, 'Trace*.dat'))
    trace_file_paths.sort(key=lambda x: os.path.getmtime(x))
    if 'trace.dat' in trace_file_paths[-1]:#If the latest file is trace.dat
        final_file_path = trace_file_paths[-1]
        trace_file_path = trace_file_paths[-2]
    else:
        final_file_path = None
        trace_file_path = trace_file_paths[-1]
    
    dot_file_path = trace_file_path.replace('Trace', '')
    print(trace_file_path)
    print(dot_file_path)

    dot_df = read_file(dot_file_path)
    trace_df = read_file(trace_file_path)

    n_points = trace_df.shape[0]#using trace_df because dot_df may contain 1 more line
    
    #if trace_df['y'].isna().sum() > n_points/2: #Fix occasional bug
    #    trace_df = read_file(trace_file_path,['x','dsfsg','y','type'])

    if pendant: #invert Y if pendant
        dot_df['y'] *= -1
        trace_df['y'] *= -1
    else: #Find line if sitting
        line_slice1 = dot_df[dot_df['x'] == 0]
        line_slice = line_slice1[line_slice1['type'] == 0]
        line_y = line_slice['y'].iloc[0]
        print(f'Line: {line_y}')

    #print(dot_df)
    #print(trace_df)

    if final_file_path is not None:
        try:
            final_df = read_final_file(final_file_path)
            if pendant: #invert Y and drop some points if pendant
                final_df['y'] *= -1
                final_df = final_df[final_df['y'] < 500]
            print(final_df)
            if final_df['x'].isna().sum() > final_df.shape[0]/2:
                print('Reading error. Reverting to trace_df')
            else:
                trace_df = final_df #swapping
        except:
            print('Failed to read trace.dat')

    if dif_types:
        dot_df['color'] = dot_df['type'].map(lambda x: 'blue' if x == 1 else 'gray')
    else:
        dot_df['color'] = 'blue'

    if save:
        matplotlib.use('Agg')#Disable interactive backend to fix memory leak
    else:
        matplotlib.use(default_backend)

    plt.figure(figsize=(12,9))
    plt.scatter(x=dot_df['x'], y=dot_df['y'], s=1, c=dot_df['color'])
    plt.plot(np.array(trace_df['x']), np.array(trace_df['y']), c='red', lw=1)
    
    if metadata:
        plt.suptitle(metadata)
    else:
        plt.suptitle(f'Total points: {n_points}')
    
    if not pendant:
        plt.axhline(line_y, c='lime', lw=1)
    
    if save:
        plt.savefig(save)
    else:
        plt.show()

if __name__ == '__main__':
    plot('C://ООО//Select', True)
