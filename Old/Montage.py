import numpy as np
from matplotlib import pyplot as plt
import skimage as ski
import scipy
from os import listdir
from PIL import Image
import re

#############################################################################################
# Read user inputs & search pictures
#############################################################################################

# Read user inputs written as:
# Path= <pathname>
# Filenames= <filenames>
# Microscope= Leica
# Magnification= 5x
# Snake= True
# Color output= True

try:
    usr_inputs = open("inputs.txt").read()
    txt = usr_inputs.split('\n')
    path= txt[0].split('= ')[1]
    filename_pattern= txt[1].split('= ')[1]
    microscope= txt[2].split('= ')[1]
    magnification= txt[3].split('= ')[1]
    snake= txt[4].split('= ')[1]
    gray= txt[5].split('= ')[1]
    del(usr_inputs, txt)
    print("\nUser inputs loaded")
except:
    print("\nError in user inputs.\nScript aborted.")
    quit()

#Search all pictures
files = listdir(path)
filenames=[]
valid_extension=['tif', '.jpg', '.jpeg', '.png', '.gif', '.bmp']
for f in files:
    if (any(ext in f for ext in valid_extension)) and (filename_pattern in f): filenames.append(f)
#Verify there is the good number of picture and sort them by alphabetical order
if len(filenames) > 0:
    filenames.sort()
    if len(filenames) == 19:
        print("Expected number of pictures found")
    elif len(filenames) > 19:
        print("More than 19 pictures are found.\nScript aborted.")
        quit()
    else:
        print("Less than 19 pictures are found, absent pictures will be replaced by empty picture in Montage.")
else:
    print("No pictures found.\nScript aborted.")
    quit()


#############################################################################################
# Do the montage of the entire substrate
#############################################################################################

# Substrate is definied as follow:
# 1st row: 3 hives
# 2nd row: 4 hives
# 3rd row: 5 hives
# 4th row: 3 hives
# 5th row: 4 hives
idx = [[0,1,2], [3,4,5,6], [7,8,9,10,11], [12,13,14,15], [16,17,18]]
if snake=='True':
    i=0
    for row in idx:
        if i%2 !=0: row.reverse()
        i += 1
idx_flat = [item for sublist in idx for item in sublist]
print(idx)
print(idx_flat)

#Open pictures and store it in a list
img = [None]*len(filenames) ; shp= [None]*len(filenames)
for i in range(len(filenames)):
    img_full_path = path + '\\' + filenames[i]
    img[i]=ski.io.imread(img_full_path)
    shp[i]=img[i].shape
    print('class img: ', type(img[i]))
    #print('\nImage ', filenames[i], ':\nSize= ', img[i].shape)

#Verify the shapes are consistent
if len(set(shp)) > 1:
    print("No pictures found.\nScript aborted.")
    quit()
else:
    img_sz = img[0].shape 
    print(img_sz)

#Determine picture type (RGB or gray) & range of pixel values


#Add empty pictures for the missing ones
if len(filenames) < 19:
    img_number = []
    for f in filenames:
        img_number.append(re.search('00[0-9][0-9]', f).group())
    numbers = []
    for i in range(2):
        for j in range(10):
            if str(i)+str(j) != '19': numbers.append('00'+str(i)+str(j))
    missing_img = [x for x in numbers if x not in img_number]
    missing_idx = [int(x) for x in missing_img]
    for i in missing_idx:
        img.insert(i, np.empty(img_sz))
        print('img added class: ', type(img[i]))

#Reconstruct final image. NOTE: issue with exact starting points (+/- 1)
montage = np.empty((img_sz[0]*5, img_sz[1]*5, img_sz[2]), dtype=np.uint8)
img_order = [img[i] for i in idx_flat]
i=0 #counting pictures in following nested loops
c=0
for r in [0,1,2,1,0]:
    #Initiate location of the first picture of the row
    x_min = round((5*img_sz[1] - (r+3)*img_sz[1])/2)
    x_max = round(x_min + img_sz[1])
    y_min = img_sz[0]*(c)
    y_max = y_min + img_sz[0]

    for elt in range(len(idx[r])):
        print('r=', r, 'elt=', elt, 'coord= ', y_min, y_max, x_min, x_max)
        # plt.imshow(img[i])
        # plt.show()
        montage[y_min:y_max, x_min:x_max, :] = img[i] #add picture to montage at good location
        #Set paramaters for next picture from the row
        i += 1
        x_min += img_sz[1]
        x_max = x_min + img_sz[1]
        # plt.imshow(montage)
        # plt.show()
    c+=1
try:
    out_path = path +'\\Montage_' + filename_pattern + '.jpeg'
    ski.io.imsave(out_path, montage)
    print('Imaged saved as: ', out_path)
except:
    print('Image could not be saved in ', out_path)

#ski.io.show()
    
    





    
