# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

L'objectif de ce script est de rechercher des transitoires dans les images trainées de TAROT.
"""
import pdb
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import glob
import scipy.signal
import scipy.optimize




fitsfile = "/home/echo/Documents/data/trail/IM_20160820_200226731_000000_30756801.fits"
input_path = "/home/echo/Documents/data/trail/"
findit = os.path.join(input_path+'/*.fits')
filelist = glob.glob(findit)


outpath = "/tmp/Trail_Analysis/"


debug = 0
write_images = 1
insert_stuff = 1

data = np.zeros([2048,2048])

correlation_map = np.ndarray([len(data),len(data)])
correlation_mapo1 = np.ndarray([len(data),len(data)])
correlation_mapo2 = np.ndarray([len(data),len(data)])

coret22d = np.zeros_like(correlation_map)
coret23d = np.zeros([46])



def create_kernels(fwhm, trail_length = 46, power = 2):
    coret2 = np.transpose(np.zeros(len(data)-2))
    coret1 = np.transpose(np.zeros(len(data)))
    core_o1 = np.transpose(np.zeros(100,float))
    core_o2 = np.transpose(np.zeros(100,float))
    template_function = np.transpose(np.zeros(100,float))
    
    for a in range((len(data)-trail_length)/2, (len(data)+trail_length)/2):
        coret2[a] = -1
        coret1[a] = -5*(a-len(data)/2)
    x = np.arange(0,99,1)
    
    psfraw = scipy.signal.gaussian(100, fwhm/2.355, True)    
    psf = psfraw
    var_psf = np.diff(psf)
    #Power law adjustment parameter
    pla = 2
    #Creation of the function over the length of the trail
    for a in x:
        core_o1[a] = psf[a-trail_length] + psf[a]
        core_o2[a] = var_psf[a-trail_length] + var_psf[a]
        #template_function
        if a<= (len(x)-trail_length)/2:
            template_function[a] = 0
        elif a<(len(x)+trail_length)/2:
            #This must be computed with floats!!!
            template_function[a] = np.power((100. - trail_length)/2 + pla, power) * 1./np.power(a + pla, power)
        else:
            template_function[a] = 0

    #Convolved function
    convolved_template = np.convolve(template_function,psf,"same") / fwhm
    #First derivative of the convolved function
    template_diff1 = np.pad(np.diff(convolved_template),(0,1),"edge")
    #Second derivative of the convolved function
    template_diff2 = np.pad(np.diff(convolved_template,n=2),(0,2),"edge")

        
#    t = np.arange(0, 100, 1)
#    s = np.transpose([template_function, convolved_template, template_diff1, template_diff2,psf])
#    plt.plot(t, s)
#    
#    plt.xlabel('x (pix)')
#    plt.ylabel('Value (ADU)')
#    plt.title('Correlation kernel')
#    plt.grid(True)
#    
#    plt.show()
    
    kernel_o0 = np.pad(convolved_template*10000,((len(data)-100)/2,(len(data)-100)/2),"edge")
    kernel_o1 = np.pad(convolved_template*10000,((len(data)-100)/2,(len(data)-100)/2),"edge")
    kernel_o2 = np.pad(convolved_template*10000,((len(data)-100)/2,(len(data)-100)/2),"edge")
    
    
    return coret2, kernel_o0, kernel_o1, kernel_o2
    
    
def kernel_fuse(k1, k2, C, S):
    fused = k1 + C*np.roll(k2,int(S))
    
    t = np.arange(924, 1124, 1)
    s = np.transpose([fused[t]])
    
    
    plt.plot(t, s)
    
    plt.xlabel('x (pix)')
    plt.ylabel('Value (ADU)')
    plt.title('Correlation kernel')
    plt.grid(True)
    
    plt.show()
    print([C,S])
    
    return fused

def maxfunctiono0(C, S):

    #creation of kernels for stars and transient
    dc,transient0, transient1, transient2 = create_kernels(4, 46, 2)
    dc, norm_star0, norm_star1, norm_star2 = create_kernels(4, 46, 0)
    
    #creation of a dummy signal
    signal = np.zeros(len(data))
    ssignal = signal + np.roll(norm_star0, 0)
    tsignal = signal + np.roll(transient0, 0)
    

#    exp_kernel = kernel_fuse(transient0,norm_star0, -0.8, -5.)
    exp_kernel = kernel_fuse(transient0,norm_star0, C, S)
    tominimize = np.max(scipy.signal.fftconvolve(tsignal, exp_kernel, "same")) / np.max(scipy.signal.fftconvolve(ssignal, exp_kernel, "same"))
    return tominimize, exp_kernel
    
    
def optimize_kernels():

    
    
    #print scipy.optimize.minimize(minfunctiono0(C, S),[-0.4,-5.], "Nelder-Mead")
    
    max_C = 0.
    max_S = 0.
    max_fun = 0
    for C in np.arange(-1,-0.2,0.1):
        for S in np.arange(-10,2,1):
            attempt, dp = maxfunctiono0(C,S)
            print([C,S,max_fun])
            if attempt > max_fun:
                max_C = C
                max_S = S
                max_fun = attempt
                print ("cool")
    return max_C, max_S, max_fun
    
    
    
    
    #order 0 optimization
    
    
    
    
    
def correlate_image(data_array, correlation_kernel):
    correlation_result = np.empty_like(data_array)
    correlation_bg = np.ndarray([len(data)])
    correlation_max = np.ndarray(len(data))

    for b in range(0,len(data)):
        #correlation_result[b,:]= np.correlate(data_array[b,:], correlation_kernel, "same")
        correlation_result[b,:]= scipy.signal.fftconvolve(data_array[b,:], correlation_kernel, mode="same")
        correlation_bg[b] = np.std(correlation_result[b,:])
        correlation_max[b] = np.min(correlation_result[b,:])/np.median(correlation_result[b,:])
#    for b in range(0,len(data)):
#        
#        relative_result = np.divide(correlation_result,data_array)
    return correlation_result, correlation_bg, correlation_max
    


def process_file(fitsfile, correlation_kernel, order):
    t1 = time.time()
    
    file_name = os.path.basename(fitsfile)
    
    hdu = fits.open(fitsfile)
    data = hdu[0].data
    HDU_header = hdu[0].header
    #wcs = WCS(hdu[0].header)
    hdu.close()    
    
    if insert_stuff:
        dp, mysignal, dp, dp = create_kernels(4,46,2)
        data[800,:] = data[800,:] + mysignal/2
        print "inserted signal of peak:"; print(max(mysignal)/2)
    
    time_variation = np.divide(np.pad(np.diff(data, axis=1),((0,0),(0,1)),"edge"),data)
    second_variation = np.divide(np.pad(np.diff(data,n=2,axis=1),((0,0),(0,2)),"edge"),data)


#correlation_map = np.correlate(second_variation[685], coret2, "same")
    time_variation_name = os.path.join(outpath+file_name+'.tvar')
    correlation_map_name = os.path.join(outpath+file_name+'.correlate')

    if order==2:
        datatouse = second_variation
        
    elif order==0:
        datatouse = data
    correlation_mapo2, noise_mapo2, max_mapo2  = correlate_image(datatouse, coret2)

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



    t = np.arange(0, 2048, 1)
    s = np.transpose([correlation_mapo2[800,t]])
    plt.plot(t, s)
    
    plt.xlabel('time (s)')
    plt.ylabel('Value (ADU)')
    plt.title('Sample from the correlation')
    plt.grid(True)
    
    if write_images:
        plt.savefig("test.png")
        plt.savefig('/home/echo/Documents/data/trail/plot1.png')
    plt.show()
    
    fig, ax1 = plt.subplots(figsize=(20, 10))
    t = np.arange(0, 2048, 1)
    s1 = max_mapo2[t]
    ax1.plot(t, s1, 'b.')
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
    
coret2,dp,dp,dp = create_kernels(6,46,2)
dp, mykernel = maxfunctiono0(-0.5,-5)

for image_a_traiter in filelist:
    process_file(image_a_traiter, mykernel, 0)