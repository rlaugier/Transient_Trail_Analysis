# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

L'objectif de ce script est de rechercher des transitoires dans les images trainées de TAROT.
"""

import pdb
from astropy.io import fits
from astropy import wcs
from astropy.convolution import Gaussian2DKernel, convolve
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import glob
import scipy.signal
import scipy.optimize



debug = 0
order = 0
write_images = 0
insert_stuff = 1

data = np.zeros([2048,2048])

correlation_map = np.ndarray([len(data),len(data)])
correlation_mapo1 = np.ndarray([len(data),len(data)])
correlation_mapo2 = np.ndarray([len(data),len(data)])

coret22d = np.zeros_like(correlation_map)
coret23d = np.zeros([46])



outpath = "/tmp/TransientSearch/"

def previewSize (filelist):
    numberOfImg = 0
    numberOfImg = len(filelist)
    return numberOfImg
    




def create_kernels(fwhm, trail_length = 46, power = 2):
    coret2 = np.transpose(np.zeros(len(data)-2))
    coret1 = np.transpose(np.zeros(len(data)))
    core_o1 = np.transpose(np.zeros(100,float))
    core_o2 = np.transpose(np.zeros(100,float))
    template_function = np.transpose(np.zeros(100,float))
    #Creation of a decay template function    
    for a in range(int((len(data)-trail_length)/2), int((len(data)+trail_length)/2)):
        coret2[a] = -1
        coret1[a] = -5*(a-len(data)/2)
    x = np.arange(0,99,1)
    #Creation of a gaussian PSF
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
    
    kernel_o0 = np.pad(convolved_template*3000,(int((len(data)-100)/2),int((len(data)-100)/2)),"edge")
    kernel_o1 = np.pad(template_diff1*3000,(int((len(data)-100)/2),int((len(data)-100)/2)),"edge")
    kernel_o2 = np.pad(template_diff2*3000,(int((len(data)-100)/2),int((len(data)-100)/2)),"edge")
    
    
    return coret2, kernel_o0, kernel_o1, kernel_o2
    
    
def kernel_fuse(k1, k2, C, S, D):
    fused = k1 + C*np.roll(k2,int(S))
    
#    t = np.arange(924, 1124, 1)
#    s = np.transpose([fused[t]])
#    
#    
#    plt.plot(t, s)
#    
#    plt.xlabel('x (pix)')
#    plt.ylabel('Value (ADU)')
#    plt.title('Correlation kernel')
#    plt.grid(True)
#    
#    plt.show()
    print([C,S,D])
    
    return fused
#Creates positive and negative masks of a givent trim width
def create_mask(length,widthofmask):
    goodpart = np.ones(length-2*widthofmask)
    mymask = np.pad(goodpart,(widthofmask,widthofmask),"constant", constant_values = 0)
    badpart = np.zeros(length-2*widthofmask)
    outmask = np.pad(badpart,(widthofmask,widthofmask),"constant", constant_values = 1)
    return mymask, outmask
    
#trims a 1D array, replacing values by a constant
def smart_trim(myarray, widthofmask, fillvalue):
    inmask, outmask = create_mask(len(myarray),widthofmask)
    trimmedarray = np.multiply(myarray,inmask)+np.multiply(outmask,fillvalue)
    return trimmedarray

def maxfunctiono0(C, S, D):

    #creation of kernels for stars and transient
    dc,transient0, transient1, transient2 = create_kernels(4, 46, 2)
    dc, norm_star0, norm_star1, norm_star2 = create_kernels(4, 46, 0)
    dc, antag_kernel0, antag_kernel1, antag_kernel2 = create_kernels(4, 46 +D, 0)
    
    #creation of a dummy signal
    signal = np.ones(len(data))*1000
    ssignal = signal + np.roll(norm_star0, 0)
    tsignal = signal + np.roll(transient0, 0)
    
    
    
    exp_kernel = kernel_fuse(transient0,antag_kernel0, C, S, D)
    tresponse = scipy.signal.fftconvolve(tsignal, exp_kernel, "same")
    sresponse = scipy.signal.fftconvolve(ssignal, exp_kernel, "same")
    
    t = np.arange(924, 1124, 1)
    s = np.transpose([tresponse[t],sresponse[t]])
    
    tresponse = smart_trim(tresponse, 20, 0)
    sresponse = smart_trim(sresponse, 20, 0)
    
#    tominimize = tresponse[len(tresponse)/2-46/2] / max(sresponse)
    tominimize = max(np.divide(tresponse,sresponse))

    
    plt.plot(t, s)
    
    plt.xlabel('x (pix)')
    plt.ylabel('Value (ADU)')
    plt.title('Correlation kernel')
    plt.grid(True)
    
    plt.show()
      
    
    print (tominimize)
    return tominimize, exp_kernel


    
    
def optimize_kernels():

    
    
    #print scipy.optimize.minimize(minfunctiono0(C, S),[-0.4,-5.], "Nelder-Mead")
    
    max_C = 0.
    max_S = 0.
    max_D = 0
    max_fun = 0
    for C in np.arange(-0.6,-0.3,0.05):
        for S in np.arange(-1,1,1):
            for D in np.arange(15,25):
                attempt, dp = maxfunctiono0(C,S,D)
                #print([C,S,D,max_fun])
                if attempt > max_fun:
                    max_C = C
                    max_S = S
                    max_D = D
                    max_fun = attempt
                    print (max_C, max_S, max_D, max_fun)
                    print ("cool")
    return max_C, max_S, max_D, max_fun
    #order 0 optimization
    
    
    
    
    
def correlate_image(data_array, correlation_kernel):
    correlation_result = np.empty_like(data_array)
    correlation_result_trim = np.empty_like(data_array)
    correlation_bg = np.ndarray([len(data)])
    correlation_max = np.ndarray(len(data))
    dp, kernelbg, dp, dp = create_kernels(4,46,0)

    for b in range(0,len(data)):
        #correlation_result[b,:]= np.correlate(data_array[b,:], correlation_kernel, "same")
        correlation_result[b,:]= np.divide(scipy.signal.fftconvolve(data_array[b,:], correlation_kernel, mode="same"), scipy.signal.fftconvolve(data_array[b,:],kernelbg, mode="same"))
        correlation_result_trim[b,:]=smart_trim(correlation_result[b,:],40,np.median(correlation_result[b,:]))
        correlation_bg[b] = np.std(correlation_result_trim[b,:])
        correlation_max[b] =np.max( correlation_result_trim[b,:]) / np.median(correlation_result_trim[b,:])
#        pdb.set_trace()

#    for b in range(0,len(data)):
#        
#        relative_result = np.divide(correlation_result,data_array)
    return correlation_result_trim, correlation_bg, correlation_max
    


def process_file(fitsfile, correlation_kernel, order):
    t1 = time.time()
    
    file_name = os.path.basename(fitsfile)
    
    hdu = fits.open(fitsfile)
    data = hdu[0].data
    HDU_header = hdu[0].header
    #wcs = WCS(hdu[0].header)
    hdu.close()
    w = wcs.WCS(HDU_header)
    print ("WCS information")
    w.wcs.print_contents()
    
    gauss = Gaussian2DKernel(4/2.35)
    data = convolve(data, gauss)
    
    
#correlation_map = np.correlate(second_variation[685], coret2, "same")
    time_variation_name = os.path.join(outpath+file_name+'.tvar')
    correlation_map_name = os.path.join(outpath+file_name+'.correlate')
    
    if insert_stuff:
        dp, mysignal, dp, dp = create_kernels(4,46,2)
        data[800,:] = data[800,:] + mysignal
        print ("inserted signal of peak:") ; print(max(mysignal)/2)
    
    time_variation = np.divide(np.pad(np.diff(data, axis=1),((0,0),(0,1)),"edge"),data)
    second_variation = np.divide(np.pad(np.diff(data,n=2,axis=1),((0,0),(0,2)),"edge"),data)


    if order==2:
        datatouse = second_variation
        
    elif order==0:
        datatouse = data
    correlation_mapo2, noise_mapo2, max_mapo2 = correlate_image(datatouse, coret2)
    

    if (debug ==1):
        print ("image:"); print (fitsfile)
        print ("destination of the d/dt"); print (time_variation_name)
        print ("destination of the d2/dt2"); print (correlation_map_name)


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








#    
#    fig, ax1 = plt.subplots(figsize=(20, 5))
#    t = np.arange(0, 2048, 1)
#    ax1 = np.transpose([smart_trim(correlation_mapo2[800,t],40,np.median(correlation_mapo2[800,:]))])
#    plt.plot(t, ax1)
#    
#    plt.xlabel('time (s)')
#    plt.ylabel('Value (ADU)')
#    plt.title('Sample from the correlation')
#    plt.grid(True)
#    plt.show()
    
    if write_images:
        plt.savefig("test.png")
        plt.savefig('/home/echo/Documents/data/trail/plot1.png')

    
#    fig, ax1 = plt.subplots(figsize=(20, 5))
#    t = np.arange(0, 2048, 1)
#    s1 = max_mapo2[t]
#    ax1.plot(t, s1, 'b.')
#    ax1.set_xlabel('y position (pix)')
#    # Make the y-axis label and tick labels match the line color.
#    ax1.set_ylabel('Signal correlation', color='b')
#    
#    
#    
#    ax2 = ax1.twinx()
#    s2 = noise_mapo2[t]
#    ax2.plot(t, s2, 'r.')
#    ax2.set_ylabel('Sigma correlation', color='r')
#    plt.show()
    
    find_max(correlation_mapo2,data)
    
    imgplot = plt.imshow(correlation_mapo2)
    imgplot.set_cmap('spectral')

    
    m = max(max_mapo2)
    print ("maximum = "); print(m)
    print ("at y = ")
#    print ([i for i, j in enumerate(max_mapo2) if j == m])
    
    
    t2 = time.time()
    print ("execution time"); print(t2-t1)
    
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
def crop(anarray,location):
    cropresult = np.zeros((100,100))
    for u in np.arange(location[0]-50,location[0]+50,1):
        for v in np.arange(location[1]-50,location[1]+50,1):
            cropresult[u,v] = anarray[u,v]
    return cropresult




def find_max(myimgdata,data):
    imgdata = np.copy(myimgdata)
    maxval = np.zeros(10)
    maxposx = np.zeros(10)
    maxposy = np.zeros(10)
    maskval = np.median(imgdata)
    
    for n in np.arange(0,10,1):
        
        maxval[n] = np.max(imgdata)
        maxposa = np.unravel_index(imgdata.argmax(),imgdata.shape)
        maxposx[n] = maxposa[0]
        maxposy[n] = maxposa[1]
        maskmax = np.zeros(np.shape(imgdata))
        print (maskval)
#        zoom = crop(imgdata,[maxposx[n],maxposy[n]])
#        imgplot = plt.imshow(zoom)
#        imgplot.set_cmap('spectral')
#        fig, ax1 = plt.subplots(figsize=(15, 5))
#        t = np.arange(int(maxposy[n]-50),int(maxposy[n]+50),1)
#        ax1 = np.transpose([imgdata[int(maxposx[n]),t]])
#        ax2 = ax1.twinx()
#        s2 = np.transpose([data[int(maxposx[n]),t]])
################################################
#        fig, ax1 = plt.subplots()
        
        #Using min and max to avoid reaching out of range
        t = np.arange(max(int(maxposy[n]-10),0),min(int(maxposy[n]+100),len(imgdata)),1)
        s1 = np.transpose([imgdata[int(maxposx[n]),t]])
#        ax1.plot(t, s1, 'b')
#        ax1.xlabel('Max', color='b')
#        ax1.ylabel('correlation (ADU)')
#        ax1.title('Signal and correlation')
#        ax1.grid(True)
#        ax2 = ax1.twinx()
        s2 = np.transpose([data[int(maxposx[n]),t]])
#        ax2.plot(t, s2, 'r')
#        ax2.set_ylabel('image', color='r')
        
        
        
        plt.plot(t, s1, 'b')
        plt.figure(figsize=(20, 5))
        plt.show()
        plt.plot(t,s2,'r')
        plt.figure(figsize=(20, 5))
        plt.show()
        print ([maxposx[n],maxposy[n]])
        
        

        
        
    
        for m in np.arange(maxposx[n]-100, maxposx[n]+100,1):
            for o in np.arange(maxposy[n]-100, maxposy[n]+100,1):
                if (0 <= m <len(imgdata)-2) & (0 <= o <len(imgdata)-2):
                    imgdata[m,o]=maskval
                    maskmax[maxposx[n],maxposy[n]] = maskval
#        imgplot = plt.imshow(imgdata)
#        imgplot.set_cmap('spectral')
#        plt.show()
    return maxval,np.transpose([maxposx,maxposy])

#singlefile = "/home/echo/Documents/data/log/grenouille_20160622.log"
input_path = "/home/echo/Documents/data/trail/"
findit = os.path.join(input_path+'/IM_*.fits')
filelist = glob.glob(findit)
imageNumber = previewSize(filelist)
    
coret2,dp,dp,dp = create_kernels(4,46,2)
#Maxfunction creates a composite kernel
#dp, mykernel = maxfunctiono0(-0.5,-5,0)
dp, mykernel = maxfunctiono0(-0.40,0,23)

dp, kernelsig, dp, dp = create_kernels(4,46,2)

ImgIdx = 0
imageDone = 0

for image_a_traiter in filelist:
    process_file(image_a_traiter, mykernel, 0)
    
    #Progress calculation
    ImgIdx += 1
    imageDone += 1
    completion = imageDone / imageNumber * 100
    print ("Done %d sur %d soit %d %%" % (imageDone, imageNumber, completion)) #; print ("sur %d" % imageNumber)
    #End of progress calculation