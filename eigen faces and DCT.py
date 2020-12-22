import numpy as np
import cmath,math,os,collections
from PIL import Image

people = ['Arnold_Schwarzenegger','George_HW_Bush','George_W_Bush','Jiang_Zemin']

def basis_func(k, n, num):
        if (k == 0):
            func_val = (1/np.sqrt(num))*math.cos(np.pi*(n+0.5)*k/num)
        else:
            func_val = (np.sqrt(2)/np.sqrt(num))*math.cos(np.pi*(n+0.5)*k/num)
            
            # basis[k1,n1,nrows] = (D/sqrt(nrows)) * cos(pi*(n1+0.5)*k1/nrows)
            # D = 1 if k1==0, otherwise D = sqrt(2)
            
        return func_val
    
class KNN(object):
    """KNN: a class that computes K-nearest-neighbors of each image in a dataset"""
    def __init__(self,datadir,nfeats,transformtype,K):
        self.nfeats = nfeats  # Number of features to keep
        self.transformtype = transformtype # Type of transform, 'dct' or 'pca'
        self.K = K # number of neighbors to use in deciding the class label of each token
        self.npeople = 4
        self.nimages = 12
        self.ndata = self.npeople*self.nimages
        self.nrows = 64
        self.ncols = 64
        self.npixels = self.nrows * self.ncols
        self.data = np.zeros((self.ndata,self.nrows,self.ncols),dtype='float64')
        self.labels = np.zeros(self.ndata, dtype='int')
        for person in range(0,self.npeople):
            for num in range(0,self.nimages):
                datum = 12*person + num
                datafile = os.path.join(datadir,'%s_%4.4d.ppm'%(people[person],num+1))
                img = np.asarray(Image.open(datafile))
                bw_img = np.average(img,axis=2)
                self.data[datum,:,:] = bw_img
                self.labels[datum] = person
    
    #    D = 1 if k1==0, otherwise D = sqrt(2)
        
 
    #
    # set_vectors - reshape self.data into self.vectors.
    #   Vector should scan the image in row-major ('C') order, i.e.,
    #   self.vectors[datum,n1*ncols+n2] = self.data[datum,n1,n2]
    def set_vectors(self):
        self.vectors = np.zeros((self.ndata,self.nrows*self.ncols),dtype='float64')
        for k in range(self.ndata):
            
            for i in range(self.nrows):
                
                for j in range(self.ncols):
                    
                    self.vectors[k,i*self.ncols+j] = self.data[k,i,j]
        #
        # TODO: fill self.vectors


    #
    # set_mean - find the global mean image vector
    def set_mean(self):
        self.mean = np.zeros(self.npixels, dtype='float64')
        sum_images = np.sum(self.vectors,0)
        self.mean = sum_images/self.ndata
        
        # TODO: fill self.mean


    #
    # set_centered - compute the zero-centered dataset, i.e., subtract the mean from each vector.
    def set_centered(self):
        self.centered = np.zeros((self.ndata,self.npixels), dtype='float64')
        for i in range(self.ndata):
            self.centered[i,:] = self.vectors[i,:] - self.mean
            
        # TODO: fill self.centered
 
    #
    # set_transform - compute the feature transform matrix (DCT or PCA)
    #  If transformtype=='dct':
    #    transform[ktot,ntot] = basis[k1,n1,nrows] * basis[k2,n2,ncols]
    #    basis[k1,n1,nrows] = (D/sqrt(nrows)) * cos(pi*(n1+0.5)*k1/nrows)
    #    D = 1 if k1==0, otherwise D = sqrt(2)
    #    Image pixels are scanned in row-major order ('C' order), thus ntot = n1*ncols + n2.
    #    Frequencies are scanned in diagonal L2R order: (k1,k2)=(0,0),(1,0),(0,1),(2,0),(1,1),(0,2),...
    #  If transformtype=='pca':
    #    self.transform[k,:] is the unit-norm, positive-first-element basis-vector in the
    #       principal component direction with the k'th highest eigenvalue.
    #    You can get these from eigen-analysis of covariance or gram, or by SVD of the data matrix.
    #    To pass the autograder, you must check the sign of self.transform[k,0], for each k,
    #    and set self.transform[k,:] = -self.transform[k,:] if self.transform[k,0] < 0.
    def set_transform(self):
        if self.transformtype=='dct':
            self.transform = np.zeros((self.nfeats,self.npixels),dtype='float64')

            num_rows_basis = int(np.sqrt(self.nfeats))
            ind = 0

            for k in range(2*num_rows_basis-1):

                         for j in range(k+1):
                            if(num_rows_basis-1-j >= 0 and k-num_rows_basis+1+j < num_rows_basis):                               
                            
                                for i in range(self.nrows):

                                    for l in range(self.ncols):

                                        if(k < num_rows_basis):
                                            self.transform[ind,i*self.ncols+l] = basis_func(k-j, i, self.nrows)*basis_func(j, l, self.ncols)
                                        else:
                                            self.transform[ind,i*self.ncols+l] = basis_func(num_rows_basis-1-j, i, self.nrows)*basis_func(k-num_rows_basis+1+j, l, self.ncols)
                                ind = ind+1
                                



        elif self.transformtype=='pca':
            self.transform = np.zeros((self.nfeats,self.npixels),dtype='float64')
            # TODO: set self.transform in the PCA case
            u, s, vh = np.linalg.svd(self.centered)
            for i in range(self.nfeats):
                if(vh[i,0] > 0):
                    self.transform[i,:] = vh[i,:]
                else:
                    self.transform[i,:] = -1*vh[i,:]

    #
    # set_features - transform the centered dataset to generate feature vectors.
    def set_features(self):
        self.features = np.zeros((self.ndata,self.nfeats),dtype='float64')
        transformation = np.transpose(self.transform)
        self.features = np.matmul(self.centered, transformation)
        # TODO: fill self.features


    #
    # set_energyspectrum: the fraction of total centered-dataset variance explained, cumulatively,
    #   by all feature dimensions up to dimension k, for 0<=k<nfeats.
    def set_energyspectrum(self):
        self.energyspectrum = np.zeros(self.nfeats,dtype='float64')
        total_energy = sum(sum(self.centered*self.centered))
        temp = sum(self.features*self.features)/total_energy
        for i in range(self.nfeats):
            for j in range(i+1):
                self.energyspectrum[i] += temp[j]
        #
        # TODO: calculate total dataset variances, then set self.energyspectrum
    

    #
    # set_neighbors - indices of the K nearest neighbors of each feature vector (not including itself).
    #    return: a matrix of datum indices, i.e.,
    #    self.features[self.neighbors[n,k],:] should be the k'th closest vector to self.features[n,:].
    def set_neighbors(self):
        self.neighbors = np.zeros((self.ndata,self.K), dtype='int')
        dist_mat = np.zeros((self.ndata,self.ndata), dtype ='float64')
        for i in range(self.ndata):
            for j in range(self.ndata):
                dist_mat[i,j] = np.linalg.norm(self.features[i,:]-self.features[j,:],2)
        
        for i in range(self.ndata):
            temp = np.argsort(dist_mat[i,:])
            self.neighbors[i,:] = temp[1:self.K+1]
      
        
        # TODO: fill self.neighbors
            

    # set_hypotheses - K-nearest-neighbors vote, to choose a hypothesis person label for each datum.
    #   If K>1, then check for ties!  If the vote is a tie, then back off to 1NN among the tied options.
    #   In other words, among the tied options, choose the one that has an image vector closest to
    #   the datum being tested.
    def set_hypotheses(self):
        self.hypotheses = np.zeros(self.ndata, dtype='int')
            
        for i in range(self.ndata):
            temp_count_array = np.zeros(self.npeople, dtype='int')
            temp_index_array = np.zeros(self.npeople, dtype='int')
            conflict = 0
            for j in range(self.K):
                temp_count_array[int(self.neighbors[i,j]/self.nimages)] += 1
            
            max_index = np.argmax(temp_count_array)
            for k in range(self.npeople):
                if(temp_count_array[k] == temp_count_array[max_index]):
                    temp_index_array[k] = 1
                    
                if(k != max_index):
                    conflict = 1
                    
            if(conflict == 0):
                self.hypotheses[i] = max_index
            else:
                for l in range(self.K):
                    if(temp_index_array[int(self.neighbors[i,l]/self.nimages)] == 1):
                        self.hypotheses[i] = int(self.neighbors[i,l]/self.nimages)
                        break
                        
                
        
        # TODO: fill self.hypotheses


    # set_confusion - compute the confusion matrix
    #   confusion[r,h] = number of images of person r that were classified as person h    
    def set_confusion(self):
        self.confusion = np.zeros((self.npeople,self.npeople), dtype='int')
        for i in range(self.ndata):
            self.confusion[int(i/self.nimages),self.hypotheses[i]] +=1
         
        # TODO: fill self.confusion
                

    # set_metrics - set self.metrics = [ accuracy, recall, precision ]
    #   recall is the average, across all people, of the recall rate for that person
    #   precision is the average, across all people, of the precision rate for that person
    def set_metrics(self):
#         self.metrics = [1.0, 1.0, 1.0]  # probably not the correct values!
        # 
        recall = np.zeros(self.npeople, dtype='float64')
        precision = np.zeros(self.npeople, dtype='float64')
        
        for i in range(self.npeople):
            recall[i] = self.confusion[i,i]/sum(self.confusion[i,:])
            precision[i] = self.confusion[i,i]/sum(self.confusion[:,i])
            
        
        accuracy = np.trace(self.confusion)/(sum(sum(self.confusion)))
        recall_avg = (sum(recall))/self.npeople
        precision_avg = (sum(precision))/self.npeople
        self.metrics = [accuracy, recall_avg, precision_avg]  
            
            
#         total_true = sum along row
#         total_classified = sum along column
        
#             precision = true posi/total posi classified
#             recall = true posi/total true
#             accuracy = true posi/
#             a b c d       
#           a 1 2 3 4
#           b 5 6 7 8 
#           c 9 0 1 2
#           d 3 4 5 6
        # TODO: fill self.metrics

    # do_all_steps:
    #   knn=KNN('data',36,'dct',4)
    #   knn.do_all_steps()
    #
    def do_all_steps(self):
        self.set_vectors()
        self.set_mean()
        self.set_centered()
        self.set_transform()
        self.set_features()
        self.set_energyspectrum()
        self.set_neighbors()
        self.set_hypotheses()
        self.set_confusion()
        self.set_metrics()
