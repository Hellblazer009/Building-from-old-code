import numpy as np
import cmath,math

class Spectrograph(object):
    """Spectrograph: a device that computes a spectrogram."""
    def __init__(self,signal,samplerate,framelength,frameskip,numfreqs,maxfreq,dbrange):
        self.signal = signal   # A numpy array containing the samples of the speech signal
        self.samplerate = samplerate # Sampling rate [samples/second]
        self.framelength = framelength # Frame length [samples]
        self.frameskip = frameskip # Frame skip [samples]
        self.numfreqs = numfreqs # Number of frequency bins that you want in the spectrogram
        self.maxfreq = maxfreq # Maximum frequency to be shown, in Hertz
        self.dbrange = dbrange # All pixels that are dbrange below the maximum will be set to zero

    # PROBLEM 1.1
    #
    # Figure out how many frames there should be
    # so that every sample of the signal appears in at least one frame,
    # and so that none of the frames are zero-padded except possibly the last one.
    #
    # Result: self.nframes is an integer
    def set_nframes(self):
        self.nframes = int(np.floor((len(self.signal) + self.frameskip - self.framelength)/self.frameskip)) # Not the correct value
        
        #
        # TODO: set self.nframes to something else

    # PROBLEM 1.2
    #
    # Chop the signal into overlapping frames
    # Result: self.frames is a numpy.ndarray, shape=(nframes,framelength), dtype='float64'
    def set_frames(self):
        self.frames = np.zeros((self.nframes,self.framelength),dtype='float64')
        for i in range(self.nframes):
            
            for j in range(self.framelength):
                self.frames[i,j] = self.signal[i*self.frameskip + j]
        
        #
        # TODO: fill self.frames

    # PROBLEM 1.3
    #
    # Window each frame with a Hamming window of the same length (use np.hamming)
    # Result: self.hammingwindow is a numpy.ndarray, shape=(framelength), dtype='float64'
    def set_hammingwindow(self):
        self.hammingwindow = np.zeros(self.framelength, dtype='float64')
        self.hammingwindow = np.hamming(self.framelength)
        #
        # TODO: fill self.hammingwindow

    # PROBLEM 1.4
    #
    # Window each frame with a Hamming window of the same length (use np.hamming)
    # Result: self.wframes is a numpy.ndarray, shape=(nframes,framelength), dtype='float64'
    def set_wframes(self):
        self.wframes = np.zeros(self.frames.shape, dtype='float64')
        [num_frames, frame_len] = self.frames.shape
        for i in range(num_frames):
            self.wframes[i,:] = np.multiply(self.frames[i],self.hammingwindow)
            
        # np.multiply(self.signal[i*self.frameskip:i*self.frameskip + self.framelength - 1], self.hammingwindow)
        # TODO: fill self.wframes

    # PROBLEM 1.5
    #
    # Time alignment, in seconds, of the first sample of each frame, where signal[0] is at t=0.0
    # Result: self.timeaxis is a numpy.ndarray, shape=(nframes), dtype='float32'
    def set_timeaxis(self):
        self.timeaxis = np.zeros(self.nframes, dtype='float32')
        #
        for i in range(self.nframes):
            self.timeaxis[i] = (i*self.frameskip)/self.samplerate
        # TODO: fill self.timeaxis    

    # PROBLEM 1.6
    #
    #   Length of the desired DFT.
    #   You want this to be long enough so that, in numfreqs bins, you get exactly maxfreq Hertz.
    #   result: self.dftlength is an integer
    def set_dftlength(self):
        self.dftlength = int(np.ceil(self.samplerate*(self.numfreqs)/self.maxfreq))
        
        #(N-1/N)*self.framelength = self.maxfreq

        # TODO: set self.dftlength

    # PROBLEM 1.7
    #
    # Compute the Z values (Z=exp(-2*pi*k*n*j/dftlength) that you will use in each DFT of the STFT.
    #    result (numpy array, shape=(numfreqs,framelength), dtype='complex128')
    #    result: self.zvalues[k,n] = exp(-2*pi*k*n*j/self.dftlength)
    def set_zvalues(self):
        self.zvalues = np.zeros((self.numfreqs,self.framelength), dtype='complex128')
        #
        for k in range(self.numfreqs):
            for n in range(self.framelength):
                self.zvalues[k,n] = np.exp(-2j*np.pi*k*n/self.dftlength)
        # TODO: fill self.zvalues

    # PROBLEM 1.8
    #
    # Short-time Fourier transform of the signal.
    #    result: self.stft is a numpy array, shape=(nframes,numfreqs), dtype='complex128'
    #    self.stft[m,k] = sum(wframes[m,:] * zvalues[k,:])
    def set_stft(self):
        self.stft = np.zeros((self.nframes,self.numfreqs), dtype='complex128')
        for m in range(self.nframes):
            
            for k in range(self.numfreqs):
                self.stft[m,k] = np.inner(self.wframes[m,:], self.zvalues[k,:])
        #
        # TODO: fill self.stft

    # PROBLEM 1.9
    #
    # Find the level (in decibels) of the STFT in each bin.
    #    Normalize so that the maximum level is 0dB.
    #    Cut off small values, so that the lowest level is truncated to -60dB.
    #    result: self.levels is a numpy array, shape=(nframes,numfreqs), dtype='float64'
    #    self.levels[m,k] = max(-dbrange, 20*log10(abs(stft[m,k])/maxval))
    def set_levels(self):
        self.levels = np.zeros((self.nframes,self.numfreqs), dtype='float64')
        
        maxvalue = np.amax(abs(self.stft))
#         for m in range(self.nframes):
#             for k in range(self.numfreqs):
#                 maxvalue = np.maximum(maxvalue, self.stft[m,k])
        
        for m in range(self.nframes): 
            for k in range(self.numfreqs):
                self.levels[m,k] = np.maximum(-1*self.dbrange, 20*np.log10(abs(self.stft[m,k])/maxvalue) )
                
               
        # TODO: fill self.levels

    # PROBLEM 1.10
    #
    # Convert the level-spectrogram into a spectrogram image:
    #    Add 60dB (so the range is from 0 to 60dB), scale by 255/60 (so the max value is 255),
    #    and convert to data type uint8.
    #    result: self.image is a numpy array, shape=(nframes,numfreqs), dtype='uint8'
    def set_image(self):
        self.image = np.zeros((self.nframes,self.numfreqs), dtype='uint8')
        self.image = (self.levels + self.dbrange)*(255/self.dbrange)
        self.image = self.image.astype(int)
        #
        # TODO: fill self.image

        
        
        
        