# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:05:12 2021

@author: sosulinal
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from neo import io

# Define file paths:
file_path = "Data/Igor_1/"
    
""" file_path is the main root path.     
    The definition above is a so-called relative file path -    
    relative from the folder, that contains the script you     
    are currently executing.    
    
    A so-called absolute path (and therefore indepenent from    
    your script's folder) can be defined as follows:    
    file_path = "/Users/Fabrizio/Python/Kurs/Data/Pandas_1/"   
    or    
    file_path = "C:/Users/Fabrizio/Python/Kurs/Data/Pandas_1/"    
    (adjust this to the absolute path on YOUR machine!)
"""
    
file_name = [f for f in os.listdir(file_path) if f.endswith('.ibw')]
#print(file_name[5])
file_name = sorted(file_name)

print(f"file list(sorted):{file_name}")


for i, single_file in enumerate(file_name): # single_file is one xls file inside the folder
    print(single_file)

#for current_file in file_name:

    file_name_1=single_file
    
    file_1=os.path.join(file_path, file_name_1) 
    #  actually it does: file_1 = file_path + file_name_1
    #file_2=os.path.join(file_path, file_name_2)
   
    current_igor_read = io.IgorIO(file_1).read_analogsignal(file_name_1)
    #  actually it does: file_1 = file_path + file_name_1
    #file_2=os.path.join(file_path, file_name_2)
    sampling_rate = 25000
    #times = len(current_igor_read) / sampling_rate
 
    times = np.arange(current_igor_read.shape[0]) / sampling_rate 
    
    """ The os.path.join() command just sticks the different     
        file-path components together. You can also just write:    
            
        file_1 = file_path + file_name_1    
        file_2 = file_path + file_name_2    
        Never forget to put the requiered '/' at the end of     
        the file_path definition::       
            file_path = "Data/Pandas_1/"  ⟵ okay       
            file_path = "Data/Pandas_1"   ⟵ not okay
    """
    
       
    plt.figure(1, figsize=(10, 4))
    #plt.clf()
    
    plt.plot(times, current_igor_read, label=file_name_1) #plot single sweep with legend
    
    #for key in i_results_df.keys():
    #    plt.plot(time, i_results_df[key])
       
       
    plt.title("Firing pattern ") #printing title and file name (i)
    plt.legend()
          
    plt.xlabel("time [s]")
    plt.ylabel("mV")
         
    #df_igor_all = pd.DataFrame()
    #df_igor_all[i] = array_1d
    
    
    if i==0:
    
        y = len(file_name)
        x = len(current_igor_read)
        all_igor_reads = np.zeros(shape=(x, y))
    """creating new empty array using len to define dimentions
    """
    array_1d = current_igor_read.flatten() #redusing dimensions to 1
    
    all_igor_reads[:, i] = array_1d
    
plt.plot(times, all_igor_reads.mean(axis=1), 'k', lw=5, label="avrg")        
plt.legend(loc="best", title="Legend:") 

#%%
# Kick-out traces without signifcant action potential spikings

thresholed_igor_reads = all_igor_reads 
kick_out_threshold = -40


thresholed_igor_reads_df = pd.DataFrame(thresholed_igor_reads, columns=list(file_name)) #we create panda dataframe from array
fig = plt.figure(2, figsize=(10, 4))
plt.clf()

thresholed_igor_reads_df_redused = np.zeros(thresholed_igor_reads.shape)

for column, key in enumerate(thresholed_igor_reads_df.keys()):
    
    max_value = np.max(thresholed_igor_reads_df[key])
        
    if max_value > kick_out_threshold:
       thresholed_igor_reads_df_redused[:, column] = \
        (thresholed_igor_reads_df[key])
       fn=0                                               # we introduce new variable "fn" - file number
       fn = fn + column                                   # the variable is counting sweeps (columns) over threshold
       print(column)
      #  thresholed_igor_reads_df_redused1 = thresholed_igor_reads_df_redused1 + thresholed_igor_reads_df_redused[:, column]
       plt.plot(times, thresholed_igor_reads_df[key], label = file_name[column])
       plt.legend()     
    #else: 
       #np.delete(thresholed_igor_reads_df_redused1[:, column]) 
       #thresholed_igor_reads_df_redused1 = np.delete(thresholed_igor_reads_df_redused, 0, column)
       #thresholed_igor_reads_df_redused1 = thresholed_igor_reads_df_redused.delete(column)
all_igor_reads_red = np.zeros(shape=(x, fn)) 

for column, key in enumerate(thresholed_igor_reads_df.keys()):
    
    max_value = np.max(thresholed_igor_reads_df[key])
        
    if max_value > kick_out_threshold:
       all_igor_reads_red[:, column-1] = \
        (thresholed_igor_reads_df[key])       
        
plt.title("Firing pattern (traces with spikes) ") #printing title and file name (i)
                  
plt.xlabel("time [s]")
plt.ylabel("mV")


#thresholed_igor_reads_df_redused = thresholed_igor_reads_df[thresholed_igor_reads_df>0] 
plt.plot(times, all_igor_reads_red.mean(axis=1), 'k', lw=5, label="avrg")        
plt.legend(loc="best", title="Legend:") 

#%%
# Kick-out point under without threshold

thresholed_igor_reads = all_igor_reads 
kick_out_threshold = -40


thresholed_igor_reads_df = pd.DataFrame(thresholed_igor_reads, columns=list(file_name)) #we create panda datafreme from array
fig = plt.figure(3, figsize=(10, 4))
plt.clf()

thresholed_igor_reads1_df = thresholed_igor_reads_df[thresholed_igor_reads_df>-40] 
# new pd is created with values under threshold

for column, key in enumerate(thresholed_igor_reads1_df.keys()):
    
    max_value = np.max(thresholed_igor_reads1_df[key])
    
    if max_value > kick_out_threshold:
     
       plt.plot(times, thresholed_igor_reads1_df[key])
             
plt.title("Thresholded (anything above -40) ") #printing title and file name (i)
       #plt.legend()
              
plt.xlabel("time [s]")
plt.ylabel("mV")              
#%%
# Searching for peaks (Example)

fig = plt.figure(4, figsize=(10, 4))
plt.clf()
x=all_igor_reads_red[:,1]
#x =array_1d
peaks, _ = find_peaks(x, height=-0)

plt.plot(x)

plt.plot(peaks, x[peaks], "x")
plt.title("Peaks") 
plt.xlabel("points")
plt.ylabel("mV")           
#plt.plot(np.zeros_like(x), "--", color="gray")

plt.show()
    
#%%

test_file = os.path.join(file_path, file_name[0])
test_igor_read = io.IgorIO(test_file).read_analogsignal()
test_igor_read
#test_igor_read.shape
#test_igor_read.sampling_rate
#np.array(test_igor_read.sampling_rate) # 1/s

#%%
