import numpy as np
import wave,math

steps = [
    'frames',
    'autocor',
    'lpc',
    'stable',
    'pitch',
    'logrms',
    'logsigma',
    'samplepitch',
    'excitation',
    'synthesis'
]
    

class Dataset(object):
    """
    dataset=Dataset(testcase): load the waveform for the specified testcase
    Result: 
    dataset.signal is the waveform, as a numpy array
    dataset.samplerate is the sampling rate
    dataset.framelength is set to 30ms always
    dataset.frameskip is set to half of framelength always
    dataset.nframes is set to the right number of frames
    dataset.order is set to 12 always, the useful order of the LPC
    """
    def __init__(self,testcase):
        w = wave.open('data/file%d.wav'%(testcase),'rb') 
        self.samplerate = w.getframerate()
        self.signal = np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype('float32')/32768
        w.close()
        self.framelength = round(0.03*self.samplerate)
        self.frameskip = round(0.015*self.samplerate)
        self.nframes = 1+int(math.ceil((len(self.signal)-self.framelength)/self.frameskip))
        self.order = 12
        
   # PROBLEM 4.0
    #
    # Chop the waveform into frames
    # self.frames[t,n] should be self.signal[t*self.frameskip+n]
    def set_frames(self):
        self.frames = np.zeros((self.nframes,self.framelength))
        #
        for i in range(self.nframes):
            for j in range(self.framelength):
                if(i*self.frameskip+j < len(self.signal)):
                    self.frames[i,j] = self.signal[i*self.frameskip+j]
                else:
                    self.frames[i,j] = 0
        # TODO: fill the frames

    # PROBLEM 4.1
    #
    # Find the autocorrelation function of each frame
    # self.autocor[t,self.framelength+m-1] should equal R[m],
    #   where R[m] = sum_n frame[n] frame[n+m].
    def set_autocor(self):
        self.autocor = np.zeros((self.nframes,2*self.framelength-1))
        for i in range(self.nframes):
                self.autocor[i,:] = np.correlate(self.frames[i,:], self.frames[i,:],"full")
        # TODO: compute autocor for each frame
        
    # PROBLEM 4.2
    #
    # Calculate the LPC coefficients in each frame
    # lpc = inv(R)*gamma, where R and gamma are the autocor matrix and vector, respectively
    def set_lpc(self):
        self.lpc = np.zeros((self.nframes,self.order))
        r = np.zeros((self.order,self.order))
        gamma = np.zeros((self.order))
        
        for k in range(self.nframes):
            temp = np.flip(self.autocor[k,0:self.framelength],0)
            
            for i in range(self.order):
                for j in range(self.order):
                    r[i,j] = temp[abs(i-j)]
                    
            gamma = temp[1:self.order+1]
            self.lpc[k,:] = np.matmul(np.linalg.inv(r),gamma)
            
        # TODO: for each frame, compute R, compute gamma, compute lpc
            
    # PROBLEM 4.3
    #
    # Create the inverse of a stable synthesis filter.
    #   First, find the LPC inverse filter polynomial: [1, -a[0], -a[1], ..., -a[order-1]].
    #   Second, find its roots, using np.roots.
    #   Third, truncate magnitude of the roots:
    #     if any root, r, has absolute(r)>0.999, then replace it with 0.999*np.exp(1j*np.angle(r)).
    #   Finally, reconstruct a stable inverse filter (of length order+1) using np.poly(r).
    def set_stable(self):
        self.stable = np.zeros((self.nframes,self.order+1))
        
        for i in range(self.nframes):
            poly = np.concatenate((np.array([-1]),self.lpc[i,:]), axis = 0)
            roots_poly = np.roots(poly)
            for j in range(len(roots_poly)):
                if(abs(roots_poly[j]) > 0.999):
                    roots_poly[j] = 0.999*np.exp(1j*np.angle(roots_poly[j]))
            
            self.stable[i,:] = np.poly(roots_poly)
            
            
        # TODO: (1) create the inverse filter, (2) find its roots, (3) truncate magnitude, (4) find poly
            
    # PROBLEM 4.4
    #
    # Calculate the pitch period in each frame:
    #   self.pitch[t] = 0 if the frame is unvoiced
    #   self.pitch[t] = pitch period, in samples, if the frame is voiced.
    #   Pitch period should maximize R[pitch]/R[0], in the range ceil(0.004*Fs) <= pitch < floor(0.013*Fs)
    #   Call the frame voiced if and only if R[pitch]/R[0] >= 0.25.
    def set_pitch(self):
        self.pitch = np.zeros(self.nframes)
        
        for i in range(self.nframes):
            temp = np.flip(self.autocor[i,0:self.framelength],0)
            pmin = int(np.ceil(0.004*self.samplerate))
            pmax = int(np.floor(0.013*self.samplerate))
            temp_pitch =np.argmax(temp[pmin:pmax])
            if(temp[temp_pitch+pmin]/temp[0] >= 0.25):
                self.pitch[i] = temp_pitch+pmin
            
        # TODO: for each frame, find maximum normalized autocor in the range between minpitch and maxpitch

    # PROBLEM 4.5
    #
    # Calculate the log(RMS) each frame
    # RMS[t] = root(mean(square(frame[t,:])))
    def set_logrms(self):
        self.logrms = np.zeros((self.nframes))
        for i in range(self.nframes):
            self.logrms[i] = np.log(np.sqrt(np.mean(np.square(self.frames[i,:]))))
        
        
        # TODO: calculate log RMS of samples in each frame


    # PROBLEM 4.6
    #
    # Linearly interpolate logRMS, between frame boundaries,
    #  in order to find the log standard deviation of each output sample.
    #  logsigma[t,n] = logrms[t]*(frameskip-n)/frameskip + logrms[t+1]*n/frameskip
    def set_logsigma(self):
        self.logsigma = np.zeros((self.nframes-1,self.frameskip))
        for t in range(self.nframes-1):
            for n in range(self.frameskip):
                self.logsigma[t,n] = (self.logrms[t]*(self.frameskip-n)/self.frameskip) + (self.logrms[t+1]*n/self.frameskip) 
        # TODO: linearly interpolate logRMS between frame boundaries
                
    # PROBLEM 4.7
    #
    # Linearly interpolate pitch, between frame boundaries,
    # If t and t+1 voiced: samplepitch[t,n] = pitch[t]*(frameskip-n)/frameskip + pitch[t+1]*n/frameskip
    # If only t voiced: samplepitch[t,n] = pitch[t]
    # If t unvoiced: samplepitch[t,n] = 0
    def set_samplepitch(self):
        self.samplepitch = np.zeros((self.nframes-1,self.frameskip))
        for t in range(self.nframes-1):
            for n in range(self.frameskip):
                if(self.pitch[t] == 0):
                    self.samplepitch[t,n] = 0
                elif(self.pitch[t+1] == 0):
                    self.samplepitch[t,n] = self.pitch[t]
                else:
                    self.samplepitch[t,n] = self.pitch[t]*(self.frameskip-n)/self.frameskip + self.pitch[t+1]*n/self.frameskip
        # TODO: linearly interpolate pitch between frame boundaries (if both nonzero)
                
    # PROBLEM 4.8
    #
    # Synthesize the output excitation signal
    # unvoiced: self.excitation[t,:]=np.random.normal
    #    WARNING: the call to np.random.seed(0), below, makes your "random" numbers the same as
    #    the "random" numbers in the solution, if you generate the same number of them.
    #    Keep that line there -- it's the only way to get your code to pass the autograder.
    # voiced: you need to keep a running tally of the pitch phase, from voiced frame to voiced frame.
    #    phase increments, at every sample, by 2pi/samplepitch[n].
    #    Whenever the phase passes 2pi, there is a pitch pulse.
    #    The magnitude of the pitch pulse is sqrt(samplepitch), to make sure RMS=1.
    def set_excitation(self):
        self.excitation = np.zeros((self.nframes-1,self.frameskip),dtype = "float64")
        np.random.seed(0)
        pitch_phase = 0.0
        for i in range(self.nframes-1):
            if(self.pitch[i] == 0):
                self.excitation[i,:] = np.random.normal(0,1,self.frameskip)
            else:
                for j in range(self.frameskip):
                    pitch_phase += (2*np.pi)/self.samplepitch[i,j]
                    if(pitch_phase > 2*np.pi):
                        #(pitch_phase != 0 and abs((pitch_phase%2*np.pi) - 2*np.pi) <= 1E-9):
                        self.excitation[i,j] = np.sqrt(self.samplepitch[i,j])
                        pitch_phase -= 2*np.pi 
                        
                    elif(pitch_phase <= 2*np.pi):
                        self.excitation[i,j] = 0
                       
                        
      
            # TODO: create white noise for unvoiced frames, pulse train for voiced frames

    # PROBLEM 4.9
    #
    # Synthesize the speech.
    #    x = np.reshape(np.exp(self.logsigma)*self.excitation,-1)
    #    synthesis[n] = x[n] - sum_{m=1}^{order}(stable[t,m]*synthesis[n-m])
    #    where t=int(np.floor(n/frameskip))
    #    and you can assume that synthesis[n-m]=0 for n-m < 0.
    def set_synthesis(self):
        self.synthesis = np.zeros((self.nframes-1)*self.frameskip)
        x = np.reshape(np.exp(self.logsigma)*self.excitation,-1)
        for i in range((self.nframes-1)*self.frameskip):
            temp = 0
            for j in range(1,self.order+1):
                if (i-j >= 0):
                    temp += self.stable[int(np.floor(i/self.frameskip)),j]*self.synthesis[i-j]
            self.synthesis[i] = x[i] - temp
        #
        # TODO: fill the filter buffer, then reshape it to create self.synthesis