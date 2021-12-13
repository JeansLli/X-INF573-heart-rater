import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb
from sklearn.decomposition import FastICA
import statistics

#create the data to test the function



#select which quantity you want to plot.
plt_all_freq=False
plt_hr = True
plt_signals=False

low_freq=0.75
high_freq=4

#set params to initialize the times


times = []
times_2=[i for i in range(29)] # add 1/framerate manually as it is not a global variable
hr_red = []
hr_green = []
hr_blue = []
hr_mean = []
hr_plot=[]
hr_plot_red=[]
hr_plot_green=[]
hr_plot_blue=[]
def detect_change(buffer_object,Ts, counter_end):
    """
    Input: buffer object, a sequence of frames, The size of the time step and the 
    Output: Either a plot of the heart rate over a 5 second time window
    """

    #print("enter detect_change")
    min_y = buffer_object[0].shape[0]
    min_x = buffer_object[0].shape[1]
    for j in range(len(buffer_object)):
        #if the programm failed to generate a 
        if(buffer_object[j].shape[0]==0):
            buffer_object[j] = buffer_object[j-1]
        else:
            if(buffer_object[j].shape[1]<min_x):
                min_x = buffer_object[j].shape[1]
            if(buffer_object[j].shape[0]<min_y):
                min_y = buffer_object[j].shape[0]
        

    for idx in range(len(buffer_object)):
        #cut the pictures in the right size
        buffer_object[idx] = buffer_object[idx][:min_y,:min_x,::]


    #print("after cutting")
    #for idx in range(len(buffer_object)):
    #    print(buffer_object[j].shape)
    #calculate the maximum red green and blue value, plot it over the frame
    
    #maybe better change to numpy objects.. see whether this is a problem.
    
    #red_channel = buffer_object[:,:,:,0]
    buffer_np = np.array(buffer_object)
    red_channel = buffer_np[:,:,:,0]
    green_channel = buffer_np[:,:,:,1]
    blue_channel = buffer_np[:,:,:,2]
    
    #compute the mean per image to get an array that corresponds to the 
    x_red = red_channel.mean(axis=(1,2))
    x_green = green_channel.mean(axis=(1,2))
    x_blue = blue_channel.mean(axis=(1,2))
    
    #normalize the signal over the entire time, i.e we will create a signal with zero mean and unit variance.
    mean_red = np.mean(x_red, axis=0)
    mean_green = np.mean(x_green, axis=0)
    mean_blue = np.mean(x_blue, axis=0)

    std_red = np.std(x_red, axis=0)
    std_green = np.std(x_green, axis=0)
    std_blue = np.std(x_blue, axis=0)
    #pdb.set_trace()
    x_red =  (x_red-mean_red)/std_red
    x_green =  (x_green-mean_green)/std_green
    x_blue =  (x_blue-mean_blue)/std_blue
    #pdb.set_trace()

    #Linear independent component analysis can be divided into noiseless and noisy cases
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
    X_ = np.vstack((x_red,x_green,x_blue)).T
    #pdb.set_trace()
    transformer = FastICA(n_components=3, random_state = 0)
    S_ = transformer.fit_transform(X_)
    #print("X_.shape", X_.shape)
    #print("S_.shape",S_.shape)
    #pdb.set_trace()
    if plt_signals:
    #print('red_freq',red_freq)

        #create a time array over the 
        times_2 = [counter_end*Ts-i*Ts for i in range(S_.shape[0])]
        times_2.reverse()
        #pdb.set_trace()
        plt.clf()
        plt.title("Plot of the raw color signals extracted from the ICA")
        plt.xlabel("x axis frequency")
        plt.ylabel("t axis value")
        #pdb.set_trace()
        plt.plot(times_2, S_[:,0], color="red")
        plt.plot(times_2, S_[:,1], color = "green")
        plt.plot(times_2, S_[:,2], color = "blue")
        
        plt.draw()
        plt.pause(0.01)

    #pdb.set_trace()

    t = np.arange(S_.shape[0])
    red_fft = np.abs(np.fft.fft(S_[:,0]))**2
    red_freq = np.fft.fftfreq(np.size(S_[:,0],0),Ts)
    
    #extract with the mask only the values that are in the reasonable domain for the heart rate.
    indexing = np.ma.masked_where(np.abs(red_freq)<high_freq, red_freq)
    #indexing = np.ma.masked_where(True, red_freq)
    vals_to_keep = indexing.mask
    red_freq = red_freq[vals_to_keep]
    red_fft = red_fft[vals_to_keep]

    indexing = np.ma.masked_where(red_freq>low_freq, red_freq)
    #indexing = np.ma.masked_where(True, red_freq)
    vals_to_keep = indexing.mask
    red_freq = red_freq[vals_to_keep]
    red_fft = red_fft[vals_to_keep]


    green_fft = np.abs(np.fft.fft(S_[:,1]))**2
    green_freq = np.fft.fftfreq(np.size(S_[:,1],0),Ts)
    
    
    indexing = np.ma.masked_where(np.abs(green_freq)<high_freq, green_freq)
    #indexing = np.ma.masked_where(True, green_freq)
    vals_to_keep = indexing.mask
    green_freq = green_freq[vals_to_keep]
    green_fft = green_fft[vals_to_keep]

    indexing = np.ma.masked_where(green_freq>low_freq, green_freq)
    #indexing = np.ma.masked_where(True, green_freq)
    vals_to_keep = indexing.mask
    green_freq = green_freq[vals_to_keep]
    green_fft = green_fft[vals_to_keep]

    blue_fft = np.abs(np.fft.fft(S_[:,2]))**2
    blue_freq = np.fft.fftfreq(np.size(S_[:,2],0),Ts)
    
    
    indexing = np.ma.masked_where(np.abs(blue_freq)<high_freq, blue_freq)
    #indexing = np.ma.masked_where(True, blue_freq)
    vals_to_keep = indexing.mask
    blue_freq = blue_freq[vals_to_keep]
    blue_fft = blue_fft[vals_to_keep]

    indexing = np.ma.masked_where(blue_freq>low_freq, blue_freq)
    #indexing = np.ma.masked_where(True, blue_freq)
    vals_to_keep = indexing.mask
    blue_freq = blue_freq[vals_to_keep]
    blue_fft = blue_fft[vals_to_keep]
    #pdb.set_trace()
    if plt_all_freq:
    #print('red_freq',red_freq)
        plt.clf()
        plt.title("frequency spectrum")
        plt.xlabel("x axis frequency")
        plt.ylabel("t axis value")
        plt.plot(red_freq, np.real(red_fft), color="red")
        plt.plot(green_freq, np.real(green_fft), color = "green")
        plt.plot(blue_freq, np.real(blue_fft), color = "blue")
        
        plt.draw()
        plt.pause(0.01)
    #pdb.set_trace()
    peak_red = red_freq[np.argmax(np.real(red_fft))]
    peak_green = green_freq[np.argmax(np.real(green_fft))]
    peak_blue = blue_freq[np.argmax(np.real(blue_fft))]
    if plt_hr:
        times.append(counter_end*Ts)
        #raw signal data
        hr_red.append(60*peak_red)
        hr_green.append(60*peak_green)
        hr_blue.append(60*peak_blue)
        hr_mean.append(20*(peak_red+peak_blue+peak_green))
        

        #this means that we will take the heart rate over a 5*framerate window, i.e the mean over 5 seconds.
        if(len(times)<=150):
            hr_plot=hr_mean
            hr_plot_red = hr_red
            hr_plot_green = hr_green
            hr_plot_blue = hr_blue
        if(len(times)==150):
            times.pop(0)

            hr_mean.pop(0)
            hr_red.pop(0)
            hr_green.pop(0)
            hr_blue.pop(0)

            #compute the mean over a five second to denoise the signal.
            heart_rate_mean = statistics.mean(hr_mean)
            hr_mean.pop()
            hr_mean.append(heart_rate_mean)

            
            

            hr_plot.append(heart_rate_mean)
            hr_plot_red.append(statistics.mean(hr_red))
            hr_plot_green.append(statistics.mean(hr_green))
            hr_plot_blue.append(statistics.mean(hr_blue))

            hr_plot.pop(0)
            hr_plot_red.pop(0)
            hr_plot_green.pop(0)
            hr_plot_blue.pop(0)

            print("hr_mean=",heart_rate_mean)
        

        plt.clf()
        plt.title("Heart rate derived from the mean of the face color components.")
        plt.xlabel("time")
        plt.ylabel("heart rate")
        plt.plot(times, hr_plot, color="black")
        #plt.plot(times, hr_green, color="green")
        #plt.plot(times, hr_plot_green, color="green")
        plt.ylim(50,120)
        #plt.plot(times, hr_plot_red, color="red")
        #plt.plot(times,hr_plot_green,color="green")
        #plt.plot(times, hr_plot_blue,color="blue")

        plt.draw()
        plt.pause(0.01)

    
    



#detect_change(buffer,0.1)