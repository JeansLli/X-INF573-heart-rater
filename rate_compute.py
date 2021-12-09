import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb
from sklearn.decomposition import FastICA
import statistics

#create the data to test the function

buffer = []
for j in range(50):
    #im = cv2.imread('/Users/jingyili/Documents/ip-paris/courses_taken/INF573/project/data_face/pic_%i.png'%j)
    #buffer.append(im)
    pass
plt_all_freq=False
plt_hr = False
plt_signals=True

low_freq=0.75
high_freq=4

#set params to initialize the times


times = []
times_2=[i/25 for i in range(24)]
hr_red = []
hr_green = []
hr_blue = []
hr_mean = []
def detect_change(buffer_object,Ts, counter_end):
    """
    Input: buffer object, a sequence of frames
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

    #normalize the vector over the whole time
    x_red =  x_red/np.linalg.norm(x_red)
    x_green =  x_green/np.linalg.norm(x_green)
    x_blue =  x_blue/np.linalg.norm(x_blue)


    #Linear independent component analysis can be divided into noiseless and noisy cases
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
    X_ = np.vstack((x_red,x_green,x_blue)).T
    #pdb.set_trace()
    transformer = FastICA(n_components=3, random_state = 0)
    S_ = transformer.fit_transform(X_)
    #print("X_.shape", X_.shape)
    #print("S_.shape",S_.shape)
    
    if plt_signals:
    #print('red_freq',red_freq)

        times_2.pop(0)
        times_2.append(counter_end*Ts)
        #pdb.set_trace()
        plt.clf()
        plt.title("frequency")
        plt.xlabel("x axis frequency")
        plt.ylabel("t axis value")
        plt.plot(times_2, S_[:,0], color="red")
        plt.plot(times_2, S_[:,1], color = "green")
        plt.plot(times_2, S_[:,2], color = "blue")
        
        plt.draw()
        plt.pause(0.01)

    #pdb.set_trace()

    t = np.arange(S_.shape[0])
    red_fft = np.fft.fft(S_[:,0])
    red_freq = np.fft.fftfreq(np.size(S_[:,0],0),Ts)
    
    
    indexing = np.ma.masked_where(np.abs(red_freq)<high_freq, red_freq)
    vals_to_keep = indexing.mask
    red_freq = red_freq[vals_to_keep]
    red_fft = red_fft[vals_to_keep]

    indexing = np.ma.masked_where(red_freq>low_freq, red_freq)
    vals_to_keep = indexing.mask
    red_freq = red_freq[vals_to_keep]
    red_fft = red_fft[vals_to_keep]


    green_fft = np.fft.fft(S_[:,1])
    green_freq = np.fft.fftfreq(np.size(S_[:,1],0),Ts)
    
    
    indexing = np.ma.masked_where(np.abs(green_freq)<high_freq, green_freq)
    vals_to_keep = indexing.mask
    green_freq = green_freq[vals_to_keep]
    green_fft = green_fft[vals_to_keep]

    indexing = np.ma.masked_where(green_freq>low_freq, green_freq)
    vals_to_keep = indexing.mask
    green_freq = green_freq[vals_to_keep]
    green_fft = green_fft[vals_to_keep]

    blue_fft = np.fft.fft(S_[:,2])
    blue_freq = np.fft.fftfreq(np.size(S_[:,2],0),Ts)
    
    
    indexing = np.ma.masked_where(np.abs(blue_freq)<high_freq, blue_freq)
    vals_to_keep = indexing.mask
    blue_freq = blue_freq[vals_to_keep]
    blue_fft = blue_fft[vals_to_keep]

    indexing = np.ma.masked_where(blue_freq>low_freq, blue_freq)
    vals_to_keep = indexing.mask
    blue_freq = blue_freq[vals_to_keep]
    blue_fft = blue_fft[vals_to_keep]
    #pdb.set_trace()
    if plt_all_freq:
    #print('red_freq',red_freq)
        plt.clf()
        plt.title("frequency")
        plt.xlabel("x axis frequency")
        plt.ylabel("t axis value")
        plt.plot(red_freq, np.real(red_fft), color="red")
        plt.plot(green_freq, np.real(green_fft), color = "green")
        plt.plot(blue_freq, np.real(blue_fft), color = "blue")
        
        plt.draw()
        plt.pause(0.01)

    peak_red = red_freq[np.argmax(np.real(red_fft))]
    peak_green = green_freq[np.argmax(np.real(green_fft))]
    peak_blue = blue_freq[np.argmax(np.real(blue_fft))]
    if plt_hr:
        times.append(counter_end*Ts)
        hr_red.append(60*peak_red)
        hr_green.append(60*peak_green)
        hr_blue.append(60*peak_blue)
        hr_mean.append(20*(peak_red+peak_blue+peak_green))
        
        
        if(len(times)==25):
            times.pop(0)
            hr_mean.pop(0)
            #compute the mean over 100 time steps
            heart_rate = statistics.mean(hr_mean)
            hr_mean.pop()
            hr_mean.append(heart_rate)
            print("hr_mean=",heart_rate)
        

        plt.clf()
        plt.title("heart rate of the green component")
        plt.xlabel("time")
        plt.ylabel("heart rate")
        plt.plot(times, hr_green, color="green")
    

        plt.draw()
        plt.pause(0.01)

    print('heart rate red in bpm',peak_red*60)
    print('heart rate green in bpm',peak_green*60)
    #print('heart rate blue in bpm',peak_blue*60)
    



#detect_change(buffer,0.1)