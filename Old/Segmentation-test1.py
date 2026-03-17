import numpy as np
from matplotlib import pyplot as plt
import skimage as ski
import scipy
from os import listdir
from PIL import Image

#############################################################################################
#Read user inputs & search pictures
#############################################################################################

#Read user inputs written as:
#Path= <pathname>
#Filenames= <filenames>
usr_inputs = open("inputs.txt").read()
txt = usr_inputs.split('\n')
path= txt[0].split('= ')[1]
filename_pattern= txt[1].split('= ')[1]
del(usr_inputs, txt)
print("\nUser inputs loaded")

#Search all pictures
files = listdir(path)
filenames=[]
for f in files:
    if filename_pattern in f:
        filenames.append(f)
del(filename_pattern)
print("Pictures found")

#############################################################################################
#Segmentation of pictures picture
#############################################################################################

#Function defintion for N gaussian fit
def n_gauss_fit(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

#Function definition for one gaussian
def gauss_fit(x, *params):     
    return (params[0] ** params[1]) * np.exp(-params[0]) / factorial[params[1]] + params[2]

# def func(x, *args):
#     x = x.reshape(-1, 1)
#     a = np.array(args[0::3]).reshape(1, -1)
#     b = np.array(args[1::3]).reshape(1, -1)
#     c = np.array(args[2::3]).reshape(1, -1)
#     return np.sum(a * np.exp(-b * x) + c, axis=1)

#Open Picture & convert it in 1) gray scale & 2) a np.array
idx = [0, 1]#, 4, 5, 6, 7, 8, 9, 12, 13]
for j in idx:
    f=filenames[j]
    img_full_path = path + '\\' + f
    img_gs_ski=np.array(ski.color.rgb2gray(ski.io.imread(img_full_path)))
    print('\nImage ', f, ':\nSize= ', img_gs_ski.shape)

    #Apply gaussian blur
    r_blur=5
    img_blur = scipy.ndimage.gaussian_filter(img_gs_ski, r_blur)
    print('Imaged blured with gaussian filter with sigma = 5pxl')
    # Fig_blur=plt.figure("Blured BF image " + f)
    # plt.imshow(img_blur)
    # plt.colorbar()
    # plt.show(block=False)

    # #Remove potential shadows on the picture
    # img_divide = np.divide(img_gs_ski, img_blur)
    # Fig_divide=plt.figure("Divided BF image " + f)
    # plt.imshow(img_divide)
    # plt.colorbar()
    # plt.show(block=False)

    #Plot of the histogram of the pixel values from the picture
    Hist_pxl=plt.figure('Histogram of pixel values of image ' + f)
    data =  plt.hist(img_blur.flatten(), 100)
    x=np.zeros(len(data[0]))
    for dx in range(len(data[0])):
        x[dx]=data[1][dx]+((data[1][dx+1]-data[1][dx])/2)
    # plt.scatter(x, data[0])

    #Find peaks on histogram
    peaks = scipy.signal.find_peaks(data[0], distance=10, prominence=1)
    print(peaks)
    plt.plot(x[peaks[0]], data[0][peaks[0]], "xr")

    #Find last peak parameters
    # guess = [
    #     x[peaks[0][-1]], #center of gaussian considered as location of peak
    #     data[0][peaks[0][-1]], #amplitude considered as count of the peak
    #     (x[99] - x[peaks[0][-1]])/2
    # ]
    # print(guess)
    # popt, pcov = scipy.optimize.curve_fit(gauss_fit, x, data[0], p0=guess)
    # print(popt)
    # fit = gauss_fit(x, *popt)
    # plt.scatter(x, fit, c='r')

    #Set initial parameters for fit based on peaks location
    x0 = [None]*len(peaks[0]) ; a0 = [None]*len(peaks[0]) ; w0= [None]*len(peaks[0])
    for k in range(len(peaks[0])):
        x0[k] = x[peaks[0][k]] #center of gaussian considered as location of peak
        a0[k] = data[0][peaks[0][k]] #amplitude considered as count of the peak
        #width of peak consider as 50% of the wideth in between 2 consecutive peaks
        if k==len(peaks[0])-1:
            peak1 = 99
        else:
            peak1 = peaks[0][k+1]
        peak0 = peaks[0][k]
        w0[k] = (x[peak1] - x[peak0])/2
    guess = [None]*len(x0)*3
    for k in range(0, len(x0)*3, 3):
        guess[k] = x0[int(k/3)]
        guess[k+1] = a0[int(k/3)]
        guess[k+2] = w0[int(k/3)]
    print('Number of peaks found in histogram of pixel value:= ', len(peaks[0]))

    #Set threshold (to see later, for now arbitrary value set to verify next steps)


    #Fit histogram for N gaussians
    n_max=len(peaks[0]) #maximum number of gaussian is the number of peaks found previously
    try:
        popt, pcov = scipy.optimize.curve_fit(n_gauss_fit, x, data[0], p0=guess)
        print(popt)
        fit = n_gauss_fit(x, *popt)
        plt.scatter(x, fit, c='r')
        plt.legend(['data', 'peaks', 'gaussian fits'])
        plt.show(block=False)
        print('Fit of the ', len(peaks[0]), ' gaussians done (see histogram)')

    except:
        print('Fit did not converged')
        #popt=None; pcov=None
    
    #Threshold defined as 2sigma of the last peak
    thresh = popt[-3] - 3*popt[-1]
    plt.plot([thresh, thresh], [0, data[0].max()], 'b')
    print(thresh)

    # #Apply threshold on image
    # img_thresh=np.zeros((img_blur.shape))
    # for x in range(img_blur.shape[0]):
    #     for y in range(img_blur.shape[1]):
    #         if img_blur[x,y] >= thresh: #(img_blur[x,y] <= thresh[1]) and (img_blur[x,y] >= thresh[0]):
    #             img_thresh[x,y] = 1
    # Fig_thresh=plt.figure("Thesholded BF image " + f)
    # plt.imshow(img_thresh)
    # plt.colorbar()
    # plt.show(block=False)

    # #Remove borders
    # img_no_borders=ski.segmentation.clear_border(img_thresh)
    # # Fig_no_borders=plt.figure("Thesholded BF image without broders" + f)
    # # plt.imshow(img_no_borders)
    # # plt.colorbar()
    # # plt.show(block=False)

    # #Find different objects (region props)
    # label_img = ski.measure.label(img_no_borders)
    # regions = ski.measure.regionprops(label_img)
    # Fig_thresh=plt.figure("Labeled image " + f)
    # plt.imshow(label_img)
    # plt.colorbar()
    # plt.show(block=False)

    # #Keep only the objects that could be organoids:
    # # circularity in between 0.3-1
    # # area in between 0.007-1.5 mm²
    # labs = []
    # print(regions)
    # for r in regions:
    #     print(r.eccentricity, r.area_filled)
    #     if (r.eccentricity >= 0.8) or (r.area_filled <= 50000) or (r.area_filled >= 5000000):
    #         labs.append(r.label)
    # print(labs)
    # print(label_img)
    # for i in sorted(labs, reverse=True):
    #     del(regions[i-1])
    #     label_img[(label_img == i)] = 0
    # print(regions)
    # Fig_thresh=plt.figure("Labeled image good cond" + f)
    # plt.imshow(label_img)
    # plt.colorbar()
    # plt.show(block=False)

#     #Overlay mask and grayscale image to verify
# plt.show()




# # #Fill holes
# # img_no_holes = img_thresh
# # img_no_holes=scipy.ndimage.binary_fill_holes(img_no_holes).astype(int)
# # Fig_no_holes=plt.figure("No holes BF image " + f)
# # plt.imshow(img_no_holes)
# # plt.colorbar()
plt.show()