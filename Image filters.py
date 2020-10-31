import numpy as np
import cmath,math,os,collections
from PIL import Image

steps = [
    'ypbpr',
    'rowconv',
    'gradient',
    'background',
    'clipped',
    'matchedfilters',
    'matches',
    'features',
    'confusion',
    'accuracy'
    ]
    

class Dataset(object):
    """
    dataset=Dataset(classes), where classes is a list of class names.
    Result: 
    dataset.data is a list of observation data read from the files,
    dataset.labels gives the class number of each datum (between 0 and len(dataset.classes)-1)
    dataset.classes is a copy of the provided list of class names --- there should be exactly two.
    """
    def __init__(self,classes,nperclass):
        # Number of sets (train vs. test), number of classes (always 2), num toks per set per split (6)
        self.nclasses = 2
        self.nperclass = nperclass
        self.ndata = self.nclasses * self.nperclass
        # Load classes from the input.  If there are more than 2, only the first 2 are used
        self.classes = classes
        # Number of rows per image, number of columns, number of colors
        self.nrows = 200
        self.ncols = 300
        self.ncolors = 3
        # Data sets, as read from the input data directory
        self.labels = np.zeros((self.ndata),dtype='int')
        self.data = np.zeros((self.nrows,self.ncols,self.ncolors,self.ndata),dtype='float64')
        for label in range(0,self.nclasses):
            for num in range(0,self.nperclass):
                datum = label*(self.nperclass) + num
                filename = os.path.join('data','%s%2.2d.jpg'%(self.classes[label],num+1))
                self.labels[datum] = label
                self.data[:,:,:,datum] = np.asarray(Image.open(filename))
        
    # PROBLEM 3.0
    #
    # Convert image into Y-Pb-Pr color space, using the ITU-R BT.601 conversion
    #   [Y;Pb;Pr]=[0.299,0.587,0.114;-0.168736,-0.331264,0.5;0.5,-0.418688,-0.081312]*[R;G;B].
    # Put the results into the numpy array self.ypbpr[:,:,:,m]
    def set_ypbpr(self):
        self.ypbpr = np.zeros((self.nrows,self.ncols,self.ncolors,self.ndata))
        for i in range(self.ndata):
            for j in range(self.nrows):
                for k in range(self.ncols):
                    self.ypbpr[j, k, 0, i] = 0.299*self.data[j,k,0,i] + 0.587*self.data[j,k,1,i] + 0.114*self.data[j,k,2,i]
                    self.ypbpr[j, k, 1, i] = -0.168736*self.data[j,k,0,i] - 0.331264*self.data[j,k,1,i] + 0.5*self.data[j,k,2,i]
                    self.ypbpr[j, k, 2, i] = 0.5*self.data[j,k,0,i] - 0.418688*self.data[j,k,1,i] - 0.081312*self.data[j,k,2,i]
        #
        # TODO: convert each RGB image to YPbPr

    # PROBLEM 3.1
    #
    # Filter each row of ypbpr with two different filters.
    # The first filter is the difference: [1,0,-1]
    # The second filter is the average: [1,2,1]
    # Keep only the 'valid' samples, thus the result has size (nrows,ncols-2,2*ncolors)
    # The output 'colors' are (diffY,diffPr,diffPb,aveY,avePr,avePb).
    def set_rowconv(self):
        self.rowconv = np.zeros((self.nrows,self.ncols-2,2*self.ncolors,self.ndata))
        
        
        for i in range(self.ndata):
            for j in range(self.nrows):
                for k in range(1,self.ncols-1):
                    for l in range(2*self.ncolors):
                        if(l < self.ncolors):
                            self.rowconv[j,k-1,l,i] = self.ypbpr[j,k+1,l,i] - self.ypbpr[j,k-1,l,i]
                        else:
                            self.rowconv[j,k-1,l,i] = self.ypbpr[j,k-1,l%self.ncolors,i] + 2*self.ypbpr[j,k,l%self.ncolors,i] + self.ypbpr[j,k+1,l%self.ncolors,i]
                                   
            
            
        # TODO: calculate the six output planes (diffY,diffPr,diffPb,aveY,avePr,avePb).
                    
    # PROBLEM 3.2
    #
    # Calculate the (Gx,Gy) gradients of the YPbPr images using Sobel mask.
    # This is done by filtering the columns of self.rowconv.
    # The first three "colors" are filtered by [1,2,1] in the columns.
    # The last three "colors" are filtered by [1,0,-1] in the columns.
    # Keep only 'valid' outputs, so size is (nrows-2,ncols-2,2*ncolors)
    def set_gradient(self):
        self.gradient = np.zeros((self.nrows-2,self.ncols-2,2*self.ncolors,self.ndata))
        # TODO: compute the Gx and Gy planes of the Sobel features for each image
        
        for i in range(self.ndata):
            for j in range(1,self.nrows-1):
                for k in range(self.ncols-2):
                    for l in range(2*self.ncolors):
                        if(l < self.ncolors):
                            self.gradient[j-1,k,l,i] = self.rowconv[j-1,k,l,i] + 2*self.rowconv[j,k,l,i] + self.rowconv[j+1,k,l,i]
                            
                        else:
                            self.gradient[j-1,k,l,i] = self.rowconv[j+1,k,l,i] - self.rowconv[j-1,k,l,i]

    # PROBLEM 3.3
    #
    # Create a matched filter, for each class, by averaging the YPbPr images of that class,
    # removing the first two rows, last two rows, first two columns, and last two columns,
    # flipping left-to-right, and flipping top-to-bottom.
    def set_matchedfilters(self):
        self.matchedfilters = np.zeros((self.nrows-4,self.ncols-4,self.ncolors,self.nclasses))
        temp1 = np.zeros((self.nrows-4,self.ncols-4,self.ncolors,self.nclasses))
        temp2 = np.zeros((self.nrows,self.ncols,self.ncolors,self.nclasses))
        average_ypbpr = np.zeros((self.nrows-4,self.ncols-4,self.ncolors,self.nclasses))
        
        for i in range(self.nclasses):
            temp2[:,:,:,i] = np.mean(self.ypbpr[:,:,:,i*self.nperclass:i*self.nperclass+self.nperclass], axis = 3)
        
        temp1 = temp2[2:self.nrows-2,2:self.ncols-2,:,:]
        temp1 = np.fliplr(temp1)
        self.matchedfilters = np.flipud(temp1)
            
        

        # TODO: for each class, average the YPbPr images, fliplr, and flipud

    # PROBLEM 3.4
    #
    # self.matches[:,:,c,d,z] is the result of filtering self.ypbpr[:,:,c,d]
    #   with self.matchedfilters[:,:,c,z].  Since we're not using scipy, you'll have to
    #   implement 2D convolution yourself, for example, by convolving each row, then adding
    #   the results; or by just multiplying and adding at each shift.
    def set_matches(self):
        self.matches = np.zeros((5,5,self.ncolors,self.ndata,self.nclasses))
        temp_filtered_image = np.zeros((5,5))
        for i in range(self.ndata):
            for j in range(self.nclasses):
                for l in range(self.ncolors):
                    for m in range(5):
                        for n in range(5):
                            Filter_kern = np.fliplr(self.matchedfilters[:,:,l,j])
                            Filter_kern = np.flipud(Filter_kern)
                            temp_filtered_image[m,n] = sum(sum(np.multiply(self.ypbpr[m:m+self.nrows-4,n:n+self.ncols-4,l,i],Filter_kern)))
                
                    self.matches[:,:,l,i,j] = temp_filtered_image
        # TODO: compute 2D convolution of each matched filter with each YPbPr image
    
    # PROBLEM 3.5 
    #
    # Create a feature vector from each image, showing three image features that
    # are known to each be useful for some problems.
    # self.features[d,0] is norm(Pb)-norm(Pr)
    # self.features[d,1] is norm(Gx[luminance])-norm(Gy[luminance])
    # self.features[d,2] is norm(match to class 1[all colors]) - norm(match to class 0[all colors])
    def set_features(self):
        self.features = np.zeros((self.ndata,3))
        
        for i in range(self.ndata):
            self.features[i,0] = np.linalg.norm(self.ypbpr[:,:,1,i]) - np.linalg.norm(self.ypbpr[:,:,2,i])              
            self.features[i,1] = np.linalg.norm(self.gradient[:,:,0,i]) - np.linalg.norm(self.gradient[:,:,3,i])
            self.features[i,2] = np.linalg.norm(self.matches[:,:,:,i,1]) - np.linalg.norm(self.matches[:,:,:,i,0])
        #
        # TODO: Calculate color feature, gradient feature, and matched filter feature for each image

    # PROBLEM 3.6
    #
    # self.accuracyspectrum[d,f] = training corpus accuracy of the following classifier:
    #   if self.features[k,f] >= self.features[d,f], then call datum k class 1, else call it class 0.
    def set_accuracyspectrum(self):
        self.accuracyspectrum = np.zeros((self.ndata,3))
        classification_mat = np.zeros((self.ndata,3))
        for d in range(self.ndata):
            for f in range(3):
                for k in range(self.ndata):                
                    if(self.features[k,f] >= self.features[d,f]):
                        classification_mat[k,f] = 1
                    else:
                        classification_mat[k,f] = 0
            
            for f in range(3):
                for k in range(self.ndata):
                    if((self.labels[k]) == classification_mat[k,f]):
                        self.accuracyspectrum[d,f] += 1
                    
                    
        self.accuracyspectrum = (self.accuracyspectrum/self.ndata)           
            
            
    
                
        #
        # TODO: Calculate the accuracy of every possible single-feature classifier

    # PROBLEM 3.7
    #
    # self.bestclassifier specifies the best classifier for each feature
    # self.bestclassifier[f,0] specifies the best threshold for feature f
    # self.bestclassifier[f,1] specifies the polarity:
    #   to specify that class 1 has feature >= threshold, set self.bestclassifier[f,1] = +1
    #   to specify that class 1 has feature < threshold, set self.bestclassifier[f,1] = -1
    # self.bestclassifier[f,2] gives the accuracy of the best classifier for feature f,
    #   computed from the accuracyspectrum as
    #   accuracy[f] = max(max(accuracyspectrum[:,f]), 1-min(accuracyspectrum[:,f])).
    #   If the max is selected, then polarity=+1; if the min is selected, then polarity=-1.
    def set_bestclassifier(self):
        self.bestclassifier = np.zeros((3,3))
        for i in range(3):
            if(max(self.accuracyspectrum[:,i]) >= 1-min(self.accuracyspectrum[:,i])):
                self.bestclassifier[i,0] = self.features[np.argmax(self.accuracyspectrum[:,i]),i]
                self.bestclassifier[i,1] = 1
            else:
                self.bestclassifier[i,0] = self.features[np.argmin(self.accuracyspectrum[:,i]),i]
                self.bestclassifier[i,1] = -1
            self.bestclassifier[i,2] = max(np.max(self.accuracyspectrum[:,i]), 1-np.min(self.accuracyspectrum[:,i]))
        #
        # TODO: find the threshold and polarity of best classifier for each feature