import numpy as np

from skimage.feature import hog
from skimage import exposure #Not needed in implementation

def fast_hog(digits, orientations=9, pixels_per_cell=(2,2), cells_per_block=(1,1), face_shape = (28,28), display = False):
    hog_all = []
    for k, digit in enumerate(np.array(digits)):
        face_shape = (28,28)
        digit.shape
        image = digit.reshape(face_shape).T

        if display:
            fd, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                            cells_per_block=cells_per_block, visualise=True)
        else:
            fd = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                            cells_per_block=cells_per_block, visualise=False)
           
        hog_all.append(fd)
        
        
        if display:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            
            ax1.axis('off')
            ax1.imshow(image, cmap=plt.cm.gray)
            ax1.set_adjustable('box-forced')
            
            ax2.axis('off')
            ax2.imshow(hog_image, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            ax2.set_adjustable('box-forced')
            
            # Rescale histogram for better display
            """hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
            ax3.axis('off')
            ax3.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax3.set_title('Histogram of Oriented Gradients - Rescale')
            ax3.set_adjustable('box-forced')"""
            if k >= 0:
                break
            
    return np.array(hog_all)

def compute_cell_hog(x, y, ppc_x, ppc_y, ds_x, ds_y, start_or, end_or, start_row, end_row, start_column, end_column, magnitude, orientation):
    total = 0
    for row in range(start_row, end_row):
        if x+row<0 or x+row >= ds_x:
            continue

        for column in range(start_column, end_column):
            if y+column <0 or y+column >= ds_y:
                continue
            if orientation[x+row, y+column] >= start_or or orientation[x+row, y+column]< end_or:
                continue
                
            total += magnitude[x+row, y+column]

    return float(total) / (ppc_x*ppc_y)
    
def comp_hog(grad_x, grad_y, ppc_x, ppc_y, ds_x, ds_y, nb_cells_x, nb_cells_y, nb_or):
    
    # Calculate magnitude & orientation
    magnitude = np.hypot(grad_y, grad_x)
    orientation = np.arctan2(grad_x, grad_y) * (180 / np.pi) % 180
    
    # Initialisation 
        # Center (x_0, y_0) of the cell
    x_0 = float(ppc_x)/2
    y_0 = float(ppc_y)/2
    trunc_dsx = nb_cells_x * ppc_x
    trunc_dsy = nb_cells_y * ppc_y
        # Orientation
    nb_orientation = 180. / nb_or
        # Cells
    end_row = ppc_x
    start_row = -end_row
    end_column = ppc_y
    start_column = -end_column
        # Histogram
    hist = np.zeros((nb_cells_y, nb_cells_x, nb_or))    
        
    for i in range(nb_or):
        # init orientation
        start_or = nb_orientation*(i+1)
        end_or = nb_orientation*i
        
        # init column
        y = y_0
        y_i = 0
        
        while y < trunc_dsx:
            # init row
            x = x_0
            x_i = 0
            
            while x < trunc_dsy:
                hist[y_i, x_i, i] = compute_cell_hog(x, y, ppc_x, ppc_y, ds_x, ds_y, start_or, end_or, start_row, end_row, start_column, end_column, magnitude, orientation)
                # row incrementation
                x_i += 1
                x += ppc_x
                      
            # column incrementation    
            y_i += 1
            y += ppc_y
            
    return hist
    
def compute_normalisation(histogram, cpb_x, cpb_y, nb_cells_x, nb_cells_y, orientation):
    block_x = nb_cells_x + 1 - cpb_x
    block_y = nb_cells_y + 1 - cpb_y
    norm_histogram = np.zeros((block_y, block_x, cpb_y, cpb_x, orientation ))
    for i in range(block_x):
        for j in range(block_y):
            block = histogram[j:j+cpb_y , i:i+cpb_x, :]
            epsilon = 1.0E-5
            norm_histogram[j, i, :] = block / np.sqrt(block.sum()**2 + epsilon)
    
    return norm_histogram

def digit_hog(digit, pix_per_cell = (2,2), cells_per_block = (1,1),  orientation = 9, digit_shape = (28,28), reshape = False):
 
    # Params initiatialisation
    ppc_x, ppc_y = pix_per_cell
    cpb_x, cpb_y = cells_per_block
    ds_x, ds_y = digit_shape
    nb_cells_x = int(np.floor(ds_x/ppc_x))
    nb_cells_y = int(np.floor(ds_y/ppc_y))
    
    # Reshaping
    if reshape == True:
        digit = digit.reshape((ds_x, ds_y))
        
    # Calculate row gradient : [-1, 0, 1]
    grad_x = np.zeros((ds_x, ds_y))
    grad_x[:, 1:-1] = digit[:,2:] - digit[:, :-2]
    
    # Calculate column gradient : [-1, 0, 1].T
    grad_y = np.zeros((ds_x, ds_y))
    grad_y[1:-1, :] = digit[2:, :] - digit[:-2, :]
    
    # Calculate histogram
    histogram = comp_hog(grad_x, grad_y, ppc_x, ppc_y, ds_x, ds_y, nb_cells_x, nb_cells_y, orientation)
    
    # Normalisation
    norm_histogram = compute_normalisation(histogram, cpb_x, cpb_y, nb_cells_x, nb_cells_y, orientation)
    
    # Vetorisation
    norm_histogram = norm_histogram.ravel()
    
    return norm_histogram
    
def compute_hog(digits, pix_per_cell=(2,2), cells_per_block = (1,1), orientation = 9, digit_shape = (28,28),  reshape = False):
    digits = np.array(digits)
    hist = [digit_hog(digit, pix_per_cell , cells_per_block,  orientation, digit_shape, reshape) for digit in digits]
    return hist


def hog_to_test(digits, pix_per_cell=(2,2), cells_per_block = (1,1), orientation = 9, digit_shape = (28,28),  reshape = True, fast = True, display = False):
    '''
    pix_per_cells   : pixels per cell in x and y
    cells_per_block : cells per block in x and y
    orientation     : number of orientation in the histogram
    digit shape     : initiale shape of the images (only if reshape = True )
    reshape         : only if the input is a vector instead of a image (not raveled)
    fast            : uses the skimage algorithm instead of the hand writed one
    display         : display only the first image in the case it is True, and fast is True (only in skimage mode)
    '''
    
    if fast:
        hist = fast_hog(digits, orientations = orientation, pixels_per_cell = pix_per_cell, cells_per_block = cells_per_block, face_shape = digit_shape, display = display)
    else:
        hist = compute_hog(digits, pix_per_cell, cells_per_block, orientation, digit_shape,  reshape)
    return hist
    