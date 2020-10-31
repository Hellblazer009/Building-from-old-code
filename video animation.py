import numpy as np
from PIL import Image

steps = [
    'layer1validate',
    'layer1test',
    'layer2validate',
    'layer2test',
    'grad2validate',
    'grad1validate',
    'corners',
    'cartesian2barycentric',
    'barycentric',
    'pix2tri',
    'refcoordinate',
    'intensities'
]

class Dataset(object):
    """
    dataset=Dataset(testcase): Load the validation and testing data specified by "testcase:"
      Either the 1st half (validate_audio0.txt) or the 2nd half (validate_audio1.txt) of the data.
      Read in the given image, and specified lip-region triangles.
      Use the provided neural net to convert the audio features to lip height and width sequences,
      then use linear interpolation to find trajectories of all other points in the
      triangle mesh, then identify origin-point of every output pixel,
      then generate the images, then sequence them into a video.

    The following are loaded from the data directory:
      dataset.test_audio = test audio features
      dataset.train_audio = train audio features
      dataset.validate_audio = validation audio features
      dataset.train_video = train video features
      dataset.validate_video = validation video features

      dataset.refimage = raw input image
      dataset.refpoints = set of numbered points in the reference image
      dataset.reftriangles = map from points to triangles
      dataset.refcorners = corners of the reference triangles

      dataset.weights1 = weights of the first layer in the neural net
      dataset.bias1 = bias of the first layer in the neural net
      dataset.weights2 = weights of the second layer in the neural net
      dataset.bias2 = bias of the second layer in the neural net
    """
    def __init__(self, testcase):
        self.testcase = testcase
            
        # Load the audio validation and test data, video validation data
        self.test_audio = np.loadtxt('data/test_audio%d.txt'%(testcase))
        self.ntest = self.test_audio.shape[0]
        self.validate_audio = np.loadtxt('data/validate_audio%d.txt'%(testcase))
        self.validate_video = np.loadtxt('data/validate_video%d.txt'%(testcase))
        self.nvalidate = self.validate_audio.shape[0]
        
        # Load reference image
        self.refimage = np.asarray(Image.open('data/refimage.jpg'))
        self.nrows = self.refimage.shape[0]
        self.ncols = self.refimage.shape[1]

        # Load reference points and triangles
        self.refpoints = np.loadtxt('data/refpoints.txt').astype(int) - 1
        self.npoints = self.refpoints.shape[0]
        self.reftriangles = np.loadtxt('data/reftriangles.txt').astype(int) - 1
        self.ntriangles = self.reftriangles.shape[0]

        # self.refcorners[tri_id,corner_id,:] = [x,y,1] of one corner of the tri_id'th triangle
        self.refcorners = np.ones((self.ntriangles,3,3))
        for tri_id in range(self.ntriangles):
            self.refcorners[tri_id,:,0:2] = self.refpoints[self.reftriangles[tri_id],:]

        # Load weights and bias of the neural net
        self.weights1 = np.loadtxt('data/weights1.txt')
        self.bias1 = np.loadtxt('data/bias1.txt')
        self.dimlayer0 = self.weights1.shape[1]
        self.dimlayer1 = self.weights1.shape[0]
        self.weights2 = np.loadtxt('data/weights2.txt')
        self.bias2 = np.loadtxt('data/bias2.txt')
        self.dimlayer2 = self.weights2.shape[0]

    # PROBLEM 6.0
    #
    # Compute layer1 hidden nodes for the validation data:
    # multiply self.validate_audio by weights1, add bias1, and
    # send the result through ReLU nonlinearity.
    def set_layer1validate(self):
        self.layer1validate = np.zeros((self.nvalidate,self.dimlayer1))
        temp_mat = np.zeros((self.nvalidate,self.dimlayer1))
        
        self.layer1validate = np.maximum(np.matmul(self.validate_audio,np.transpose(self.weights1))+self.bias1, temp_mat)
        
        
        # TODO: compute layer1validate

    # PROBLEM 6.1
    #
    # Compute layer1 hidden nodes for the test data:
    #   This should be exactly the same code as 6.0, but self.test_audio instead of self.validate_audio
    def set_layer1test(self):
        self.layer1test = np.zeros((self.ntest,self.dimlayer1))
        temp_mat = np.zeros((self.ntest,self.dimlayer1))
        
        self.layer1test = np.maximum(np.matmul(self.test_audio,np.transpose(self.weights1))+self.bias1, temp_mat)
        # TODO: compute layer1test

    # PROBLEM 6.2
    #a
    # Compute layer2validate, the output layer of the neural net for the validation data:
    #    multiply layer1 by weights2, and add bias2.
    # The result has dimension of dimlayer2==3, i.e., width, height0, and height1 of the lips.
    def set_layer2validate(self):
        self.layer2validate = np.zeros((self.nvalidate, self.dimlayer2))
        temp_mat = np.zeros((self.nvalidate, self.dimlayer2))
        self.layer2validate = np.matmul(self.layer1validate,np.transpose(self.weights2))+self.bias2
        # TODO: compute layer2validate

    # PROBLEM 6.3
    #
    # Compute layer2test: This should be exactly the same code as 6.2, but using
    #    layer1test as input, instead of layer1validate
    # The result has dimension of dimlayer2==3, i.e., width, height0, and height1 of the lips.
    # Results:
    #   layer1 = layer1 outputs for the _validation_ audio input
    #   layer1test = layer1 outputs for the _test_ audio input
    def set_layer2test(self):
        self.layer2test = np.zeros((self.ntest, self.dimlayer2))
        temp_mat = np.zeros((self.ntest, self.dimlayer2))
        self.layer2test = np.matmul(self.layer1test,np.transpose(self.weights2))+self.bias2
        # TODO: compute layer2test

    # PROBLEM 6.4
    #
    # Calculate the loss gradient w.r.t. weights2 and bias2 for the validation data.
    # The mean-squared error is the loss function,
    #    E = np.average(np.square(self.labels-self.layer2validate)).
    # self.grad2validate[:,0:self.dimlayer1] = dE/dweights2, gradient w.r.t. the weights2 matrix.
    # self.grad2validate[:,self.dimlayer1] = dE/dbias2, gradient w.r.t. the bias2 vector.
    def set_grad2validate(self):
        self.grad2validate = np.zeros((self.dimlayer2, 1+self.dimlayer1))
        for i in range(self.dimlayer2):
            for j in range(self.dimlayer1):
                for d in range(self.nvalidate):
                    self.grad2validate[i,j] = self.grad2validate[i,j] - (2*(self.validate_video[d,i]-self.layer2validate[d,i])*self.layer1validate[d,j])
        
            j = self.dimlayer1
            for d in range(self.nvalidate):
                self.grad2validate[i,j] = self.grad2validate[i,j] - (2*(self.validate_video[d,i]-self.layer2validate[d,i]))
        
        self.grad2validate = (1/(self.nvalidate*self.dimlayer2))*self.grad2validate
                
        # TODO: compute loss gradients w.r.t. the weights/biases

    # PROBLEM 6.5
    #
    # Calculate the loss gradient w.r.t. weights1 and bias1 for validation data.
    # The mean-squared error is the loss function,
    #         E = np.average(np.square(self.labels-self.layer2validate)).
    # self.grad1validate[:,0:self.dimlayer0] = dE/dweights1, gradient w.r.t. the weights1 matrix.
    # self.grad1validate[:,self.dimlayer0] = dE/dbias1, gradient w.r.t. the bias1 vector.
    def set_grad1validate(self):
        self.grad1validate = np.zeros((self.dimlayer1, 1+self.dimlayer0))
        for d in range(self.nvalidate):
            for j in range(self.dimlayer1):
                for k in range(self.dimlayer0):
                    temp_sum = 0
                    for i in range(self.dimlayer2):
                        temp_sum += (-2)*(self.validate_video[d,i]-self.layer2validate[d,i])*self.weights2[i,j]
                    if(self.layer1validate[d,j] > 0):
                        self.grad1validate[j,k] += temp_sum*self.validate_audio[d,k]
                    else:
                        self.grad1validate[j,k] += 0
                k = self.dimlayer0
                temp_sum = 0
                for i in range(self.dimlayer2):
                        temp_sum += (-2)*(self.validate_video[d,i]-self.layer2validate[d,i])*self.weights2[i,j]
                if(self.layer1validate[d,j] > 0):
                    self.grad1validate[j,k] += temp_sum
                else:
                    self.grad1validate[j,k] += 0
        
        self.grad1validate = (1/(self.nvalidate*self.dimlayer2))*self.grad1validate
      
        # TODO: compute loss gradients w.r.t. the weights/biases

    # PROBLEM 6.6
    #
    # Create an array self.corners that contains the augmented coordinates
    # of the corners of each triangle, in each frame.
    # self.corners[frame_id,tri_id,corner_id,0] = x-coordinate
    # self.corners[frame_id,tri_id,corner_id,1] = y-coordinate
    # self.corners[frame_id,tri_id,corner_id,2] = 1
    #   0 <= frame_id <= self.ntest, 0 <= tri_id <= self.ntriangles, 0 <= corner_id <= 2
    def set_corners(self):
        self.corners = np.ones((self.ntest,self.ntriangles,3,3))
        # Provided:
        # self.vfeats[frame_id,:] = [0,0,0] if the frame is silence, otherwise layer2test[frame_id,:].
        # self.points[frame_id,point_id,0] = x-coordinate of point_id'th point in frame_id'th frame.
        # self.points[frame_id,point_id,1] = y-coordinate of point_id'th point in frame_id'th frame.
        # self.ptrange[frame_id,:] = [xmin,ymin,xmax,ymax] of the mouth rectangle
        vfeats, points, ptrange = process_vfeats(self.layer2test, self.test_audio, self.refpoints)
        self.vfeats = vfeats
        self.points = points
        self.ptrange = ptrange
        
#         reftriangles each row has point ids
        for i in range(self.ntest):
            for j in range(self.ntriangles):
                for c in range(3):
                    
#                     self.reftriangles[j,0] = pointid for corner 0
#                     self.reftriangles[j,1] = pointid for corner 1
#                     self.reftriangles[j,2] = pointid for corner 2
                    self.corners[i,j,c,0] = self.points[i,self.reftriangles[j,c],0]
                    self.corners[i,j,c,1] = self.points[i,self.reftriangles[j,c],1]
                    self.corners[i,j,c,2] = 1
                    
                
        #
        # TODO: for each frame, use self.reftriangles and self.points to fill self.corners

    # PROBLEM 6.7
    #
    # Find the cartesian2barycentric transform for each triangle, for each frame.
    # self.cartesian2barycentric[frame_id,tri_id,:,:] is the transform, T, such that
    # [lambda1,lambda2,lambda3] = [x,y,1] @ T
    # WARNING: notice that we're using row vectors here, whereas the lecture slides use column vectors.
    def set_cartesian2barycentric(self):
        self.cartesian2barycentric = np.zeros((self.ntest,self.ntriangles,3,3))
        for i in range(self.ntest):
            for j in range(self.ntriangles):
                self.cartesian2barycentric[i,j,:,:] = np.linalg.inv(self.corners[i,j,:,:])
                
        # TODO: compute the cartesian2barycentric transform matrix for each frame and triangle

    # PROBLEM 6.8
    #
    # Find the Barycentric coordinates of each pixel, in each triangle, in each target image.
    # self.barycentric[frame_id,row,col,tri_id,:] =  Barycentric coordinates of pixel (row,col)
    #   as computed against triangle tri_id.  Notice that most pixels are outside of most
    #   triangles, so many of these will be negative numbers.
    def set_barycentric(self):
        self.barycentric = -np.ones((self.ntest,self.nrows,self.ncols,self.ntriangles,3))
        for frame_id in range(self.ntest):
            if not np.array_equal(self.vfeats[frame_id,:], [0,0,0]):
                for col in range(self.ptrange[frame_id,0],self.ptrange[frame_id,2]):
                    for row in range(self.ptrange[frame_id,1],self.ptrange[frame_id,3]):
                        for tri_id in range(self.ntriangles):
                            # TODO: compute barycentric coordinates lambda1,lambda2,lambda3
                            # of (col,row)'th pixel w.r.t. tri_id'th triangle.
                            xy_coord = np.array([col,row,1])
                            self.barycentric[frame_id,row,col,tri_id,:] = np.matmul(np.transpose(xy_coord),self.cartesian2barycentric[frame_id,tri_id,:,:])

    # PROBLEM 6.9
    # For each pixel (row,col) in each frame,
    # figure out if it's actually contained in any triangle.
    # If it's contained in no triangles, set pix2tri[frame_id,row,col]=-1
    # If it's contained in one or more triangles (e.g., maybe it's on the edge between two triangles),
    #   just set pix2tri[frame_id,row,col] equal to any one of the correct triangle IDs
    #   (doesn't matter which one).
    def set_pix2tri(self):
        self.pix2tri = -np.ones((self.ntest,self.nrows,self.ncols),dtype='int')
        for frame_id in range(self.ntest):
            if not np.array_equal(self.vfeats[frame_id,:], [0,0,0]):
                for col in range(self.ptrange[frame_id,0],self.ptrange[frame_id,2]):
                    for row in range(self.ptrange[frame_id,1],self.ptrange[frame_id,3]):
                        # TODO: find out which triangle (row,col) is in, if any
                        if len(tri_ids)>0:
                            self.pix2tri[frame_id,row,col]=-1 # INCORRECT
                            
    # PROBLEM 6.10
    #
    # For each pixel (row,col) in each frame, find corresponding xy-coordinate in refimage.
    # self.refcoordinate[frame_id,row,col,0] = x-coordinate in refimage
    # self.refcoordinate[frame_id,row,col,1] = y-coordinate in refimage
    #  These are calculated using self.barycentric and self.refcorners.
    #
    # If self.refcoordinate is outside of the image, that's fine -- leave it outside the image.
    # If (row,col) is not inside any triangle in the frame_id'th frame, then the provided code does this:
    #  (1) If (row,col) is inside the open mouth region, coordinates set to [-1,-1], i.e., nowhere.
    #  (2) If (row,col) is outside of the mouth region, coordinates set to [col,row], i.e., unchanged.
    def set_refcoordinate(self):
        # The next three lines generate a default value of
        #   refcoordinate[frame_id,row,col,:] = [col,row], i.e., unchanged from refimage to current frame
        r = np.tile(np.arange(self.nrows).reshape((1,self.nrows,1,1)),(1,1,self.ncols,1))
        c = np.tile(np.arange(self.ncols).reshape((1,1,self.ncols,1)),(1,self.nrows,1,1))
        self.refcoordinate = np.tile(np.concatenate((c,r),axis=3),(self.ntest,1,1,1))
        # Now we loop through all frames
        for frame_id in range(self.ntest):
            if not np.array_equal(self.vfeats[frame_id,:], [0,0,0]):
                for col in range(self.ptrange[frame_id,0],self.ptrange[frame_id,2]):
                    for row in range(self.ptrange[frame_id,1],self.ptrange[frame_id,3]):
                        # If pixel is in the open mouth region, set refcoordinate=[-1,-1]
                        if self.pix2tri[frame_id,row,col] == -1 and open_mouth_region(row,col):
                            self.refcoordinate[frame_id, row, col, :] = [-1,-1]
                        elif self.pix2tri[frame_id,row,col] >= 0:
                            # TODO: Use self.barycentric and self.refcorners[pix2tri,:,:]
                            # to get the reference coordinates.
                            self.refcoordinate[frame_id,row,col,:] = [-1,-1] # INCORRECT

    # PROBLEM 6.11
    #
    # For each pixel (row,col) in each frame, calculate its target intensity in the output image.
    # You can use either piece-wise constant or piece-wise bilinear interpolation.
    #
    # If the refcoordinate is outside of the image (e.g., negative, or larger than nrows),
    #  then set the output pixel to black (intensity=0).
    def set_intensities(self):
        # Default image = reference image in every frame
        self.intensities = np.tile(self.refimage.reshape(1,self.nrows,self.ncols),(self.ntest,1,1))
        for frame_id in range(self.ntest):
            if not np.array_equal(self.vfeats[frame_id,:], [0,0,0]):
                for col in range(self.ptrange[frame_id,0],self.ptrange[frame_id,2]):
                    for row in range(self.ptrange[frame_id,1],self.ptrange[frame_id,3]):
                        # TODO:
                        # If refcoordinate is inside the refimage, figure out what it should be.
                        # If refcoordinate is outside the refimage, set intensity of this pixel to 0
                        rc = self.refcoordinate[frame_id, row, col, :]
                        if not np.all(rc>=0) and np.all(rc < np.array([self.ncols,self.nrows])):
                            self.intensities[frame_id, row, col] = 0

# No work to do here. Helper functions for calculating meshes from video features
# Cleans video features corresponding to silence, smooths features, and returns corresponding meshes
def process_vfeats(vfeats, audio, refpoints):
    # Clean video features before getting meshes
    vfeats = clean_silence(audio, vfeats)
    points = interpolate_vfeats(vfeats, refpoints)
    
    ptrange = np.concatenate((np.amin(points,axis=1),np.amax(points,axis=1)),axis=1).astype(int)

    return vfeats, points, ptrange

def interpolate_vfeats(vfeats, refpoints):

    # Get the mesh points for every frame
    nframes = len(vfeats)
    nverts = refpoints.shape[0]
    points = np.zeros((nframes, nverts, 2))

    for f in range(nframes):
        # Initialize the warped mesh points to the reference mesh
        points[f,:,:] = refpoints

        # Get lip features from network predictions. 
        w = vfeats[f, 0]
        h1 = vfeats[f, 1]
        h2 = vfeats[f, 2]

        # Upper mid (points are in (x,y) coordinates)
        points[f, 3, 1] = refpoints[3,1] - h1
        # Inner upper mid
        points[f, 25, 1] = refpoints[25,1] - h1
        # Inner lower mid
        points[f, 30, 1] = refpoints[30, 1] + h2
        # Left
        points[f, 0, 0] = refpoints[0, 0] - w/2
        # Right
        points[f, 6, 0] = refpoints[6, 0] + w/2

        # Interpolate lower mid
        lowmid = (refpoints[8,:] + refpoints[9,:])/2
        retlowmid = lowmid.copy()
        retlowmid[1] += h2

        idx =  [1, 2, 23, 24]
        for i in idx:
            width_ratio = (refpoints[i,0] - refpoints[0,0])/(refpoints[3,0] - refpoints[0,0])
            points[f, i, 0] += (points[f, 0, 0] - refpoints[0,0]) * width_ratio
            points[f, i, 1] += (points[f, 3, 1] - refpoints[3,1]) * width_ratio

        idx =  [4, 5, 26, 27]
        for i in idx:
            width_ratio = (refpoints[i,0] - refpoints[6,0])/(refpoints[3,0] - refpoints[6,0])
            points[f, i, 0] += (points[f, 6, 0] - refpoints[6,0]) * width_ratio
            points[f, i, 1] += (points[f, 3, 1] - refpoints[3,1]) * width_ratio

        idx =  [7, 8, 28, 29]
        for i in idx:
            width_ratio = (refpoints[i,0]-refpoints[0,0])/(lowmid[0]-refpoints[0,0])
            points[f, i, 0] += (points[f, 0, 0] - refpoints[0,0]) * width_ratio
            points[f, i, 1] += (retlowmid[1] - lowmid[1]) * width_ratio

        idx =  [9, 10, 31, 32]
        for i in idx:
            width_ratio = (refpoints[i,0]-refpoints[6,0])/(lowmid[0]-refpoints[6,0])
            points[f, i, 0] += (points[f, 6, 0] - refpoints[6,0]) * width_ratio
            points[f, i, 1] += (retlowmid[1] - lowmid[1]) * width_ratio

        if h1 < 0:
            idx = [23, 24, 25, 26, 27]
            for i in idx:
                points[f, i, 1] = refpoints[i, 1]

        if h2 < 0:
            idx = [28, 29, 30, 31, 32]
            for i in idx:
                points[f, i, 0] = refpoints[i, 0]
                points[f, i, 1] = refpoints[i, 1]

    return points


# Used to smooth video features between frames in output
def smooth_frames(vfeats):
    smoothed = vfeats.copy()
    nframes = vfeats.shape[0]
    smoothed[1:nframes-1,:] = (smoothed[:nframes-2,:] + smoothed[1:nframes-1,:] + smoothed[2:,])/3

    return smoothed

# Used to clear predicted video features that correspond to silence.
def clean_silence(audio, video):
    silenceModel = np.array([-2.35629, 0.37765])
    num_samples = len(audio) 
    vfeats = np.zeros(video.shape)
    for i in range(num_samples):
        if audio[i,0] < silenceModel[0] or audio[i,1] < silenceModel[1]:
            vfeats[i,:] = 0
        else:
            vfeats[i,:] = video[i,:]

    vfeats = smooth_frames(vfeats)

    return vfeats

# Return True if (row,col) part of the open mouth region, False otherwise
def open_mouth_region(row, col):
    if col>15 and col<120 and row>20 and row<60:
        return True
    else:
        return False

    