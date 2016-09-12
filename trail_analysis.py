# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

L'objectif de ce script est de rechercher des transitoires dans les images trainées de TAROT.
"""
#import pdb
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import glob

t1 = time.time()


fitsfile = "/home/echo/Documents/data/trail/IM_20160820_200226731_000000_30756801.fits"
input_path = "/home/echo/Documents/data/trail/"
findit = os.path.join(input_path+'/*.fits')
filelist = glob.glob(findit)


outpath = "/tmp/Trail_Analysis/"


debug = 0
write_images = 1

data = np.zeros([2048,2048])

correlation_map = np.ndarray([len(data),len(data)])
correlation_mapo1 = np.ndarray([len(data),len(data)])
correlation_mapo2 = np.ndarray([len(data),len(data)])
coret2 = np.transpose(np.zeros(len(data)))
coret1 = np.transpose(np.zeros(len(data)))
coret22d = np.zeros_like(correlation_map)
coret23d = np.zeros([46])

for a in range((len(data)-46)/2, (len(data)+46)/2):
    coret2[a] = -1
    coret1[a] = -5*(a-len(data)/2)
    
def correlate_image(data_array, correlation_kernel):
    correlation_result = np.ndarray([len(data),len(data)])
    correlation_bg = np.ndarray([len(data)])
    correlation_max = np.ndarray(len(data))
    for b in range(0,len(data)):
        correlation_result[b,:]= np.correlate(data_array[b,:], correlation_kernel, "same")
        correlation_bg[b] = np.std(correlation_result[b,:])
        correlation_max[b] = np.max(correlation_result[b,:])
    return correlation_result, correlation_bg, correlation_max

def process_file(fitsfile, correlation_kernel):
    
    file_name = os.path.basename(fitsfile)
    
    hdu = fits.open(fitsfile)
    data = hdu[0].data
    HDU_header = hdu[0].header
    #wcs = WCS(hdu[0].header)
    hdu.close()    
    
    time_variation = np.diff(data, axis=1)/4.56
    second_variation = np.diff(data,n=2,axis=1)/4.56


#correlation_map = np.correlate(second_variation[685], coret2, "same")
    time_variation_name = os.path.join(outpath+file_name+'.tvar')
    correlation_map_name = os.path.join(outpath+file_name+'.correlate')

    correlation_mapo2, noise_mapo2, max_mapo2  = correlate_image(second_variation, coret2)


    if (debug ==1):
        print "image:"; print fitsfile
        print "destination of the d/dt"; print time_variation_name  
        print "destination of the d2/dt2"; print correlation_map_name


    #imagetosave = fits.open(new_img_name
    #Saving the image to the disk
    #imagetosave[0].data = new_img
    #imagetosave.close();
    #imagetosave.writeto(new_img_name, clobber = True)

    if write_images:
        imghdu = fits.PrimaryHDU(data=(time_variation),header = HDU_header, do_not_scale_image_data=False, ignore_blank=False,uint=True, scale_back=None)
        imghdu.writeto(time_variation_name,clobber = True )
        imghdu = fits.PrimaryHDU(data=(correlation_map),header = HDU_header, do_not_scale_image_data=False, ignore_blank=False,uint=True, scale_back=None)
        imghdu.writeto(correlation_map_name,clobber = True )




    t2 = time.time()
    print "execution time"; print(t2-t1)



    t = np.arange(0, 500, 1)
    s = np.transpose([correlation_mapo2[640,t]])
    plt.plot(t, s)
    
    plt.xlabel('time (s)')
    plt.ylabel('Value (ADU)')
    plt.title('Sample from the correlation')
    plt.grid(True)
    
    if write_images:
        plt.savefig("test.png")
        plt.savefig('/home/echo/Documents/data/trail/plot1.png')
    plt.show()
    
    
    fig, ax1 = plt.subplots()
    t = np.arange(0, 2046, 1)
    s1 = max_mapo2[t]
    ax1.plot(t, s1, 'b-')
    ax1.set_xlabel('y position (pix)')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('Signal correlation', color='b')
    
    
    
    ax2 = ax1.twinx()
    s2 = noise_mapo2[t]
    ax2.plot(t, s2, 'r.')
    ax2.set_ylabel('Sigma correlation', color='r')
    
    plt.show()
    
    m = max(max_mapo2)
    print "maximum = "; print(m)
    print "at y = "
    print ([i for i, j in enumerate(max_mapo2) if j == m])
    
    return
    
    

    
    
    
    
    #fig, ax1 = plt.subplots()
    #y = np.arange(0, 250, 1)
    #ax1 = max_mapo2[y]
    #ax1.plot(y, ax1)
    #
    #ax1.xlabel('Max', color='b')
    #ax1.ylabel('Value (ADU)')
    #ax1.title('Maximum and noise')
    #ax1.grid(True)
    #ax2 = ax1.twinx()
    #s2 = noise_mapo2[y]
    #ax2.plot(t, s2, 'r.')
    #ax2.set_ylabel('noise', color='r')
    #
    #
    #print "noisebg"; print(noise_mapo2)
    #print "max"; print(max_mapo2)
    #print "max of max"; print(np.max(max_mapo2))
    #plt.show()
    #
    
    

for image_a_traiter in filelist:
    process_file(image_a_traiter, coret2)