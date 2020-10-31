import numpy as np
import wave,math,mfcc,json,os,heapq,io,re

steps = [
    'surprisal',
    'alphahat',
    'betahat',
    'gamma',
    'xi',
    'mu',
    'var',
    'tpm',
    'trainedmodel'
]

class Dataset(object):
    """
    dataset=Dataset(epoch): If epoch==0, no model is loaded.
      If epoch>0, the model is loaded from the previous epoch.
    Result: 
    dataset.model is the model, with these fields:
      dataset.model['tpm'] is the transition probability matrix. 
      dataset.model['mu'] is the maxtoksXnfeats matrix of mean vectors (nparray)
      dataset.model['var'] is the maxtoksXnfeats matrix of variances (nparray)

    The following are loaded from the data directory:
      dataset.utts = list of the filenames of utterances
      dataset.nutts = len(dataset.utts)

      dataset.mfcc = concatenation of the MFCC feature matrices from all utterances
      dataset.startframe = list of starting frames of each utterance
      dataset.endframe = list of ending frames of each utterance
      dataset.nframes = list of the number of frames in each utterance
      dataset.nfeats = number of features in each frame

      dataset.toks = a string, containing the IPA transcription of each utterance, in order
      dataset.starttok = list of starting tok indices for each utterance
      dataset.endtok = list of ending tok indices for each utterance
      dataset.ntoks = list of the number of tokens in each utterance
    """
    def __init__(self,epoch):
        self.epoch = epoch
        # List of the utterances
        self.utts = sorted([ re.sub('\.wav','',filename) for filename in os.listdir('data/audio') ])
        self.nutts = len(self.utts)
            
        # Load the waveforms, MSTFTs, MFCCs, and phoneseqs
        self.nfeats = 13
        self.starttok = []
        self.endtok = []
        self.startframe = []
        self.endframe = []
        self.signal = np.zeros(0)
        self.mstft = np.zeros((0,257))
        self.filterbank = np.zeros((0,24))
        self.mfcc = np.zeros((0,self.nfeats))
        self.toks = str()
        for (u,utt) in enumerate(self.utts):
            self.startframe.append(self.mfcc.shape[0])
            w = wave.open(os.path.join('data','audio','%s.wav'%(utt)),'rb')
            wav=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype('float32')/32768
            w.close()
            self.signal = np.concatenate((self.signal,wav))
            mstft = mfcc.wav2mstft(wav,512,110,276)
            self.mstft = np.concatenate((self.mstft,mstft),axis=0)
            filterbank = mfcc.mstft2filterbank(mstft,11025,24)
            self.filterbank = np.concatenate((self.filterbank,filterbank),axis=0)
            Z = mfcc.znorm(mfcc.filterbank2mfcc(filterbank,self.nfeats))
            self.mfcc = np.concatenate((self.mfcc,Z),axis=0)
            self.endframe.append(self.mfcc.shape[0])
            self.starttok.append(len(self.toks))
            with io.open('data/phones/%s.txt'%(utt),encoding="utf-8",errors="ignore") as f:
                self.toks += re.sub(r'\s+',' ',f.read())
            self.endtok.append(len(self.toks))
        self.ntoks = [ self.endtok[u]-self.starttok[u] for u in range(self.nutts) ]
        self.maxtoks = max(self.ntoks)

        # Load the previous model, if epoch>0; if epoch==0, create a dummy initial model
        if epoch>0:
            modelfile = 'models/model%2.2d.json'%(epoch)
            with open(modelfile) as f:
                model = json.load(f)
                self.model = {
                    'mu':np.array(json.loads(model['mu'])),
                    'var':np.array(json.loads(model['var'])),
                    'tpm':np.array(json.loads(model['tpm'])),
                    'phones':model['phones']
                }
        else:
            phones = ' aelmnoruøǁɘɤɨɯɵɹɺɾʉʘʙ'
            self.ntypes = len(phones)
            self.model = {
                'phones':phones,
                'mu':np.zeros((len(phones),self.nfeats)),
                'var':np.ones((len(phones),self.nfeats)),
                'tpm':0.5*np.diag(np.ones(len(phones)))+0.5*np.ones((len(phones),len(phones)))/len(phones)
            }
            self.model['mu'][0:2,0]=(-1,1)  # Initial means for silence and /a/: -1 and +1, respectively
            
        # Create a map from tok2type -- from phone token indices to the corresponding type indices
        self.ntypes = len(self.model['phones'])
        self.tok2type = [ str.find(self.model['phones'],x) for x in self.toks ]

    def get_params_for_utt(self,u):
        '''Get local model parameters for the u'th utterance'''
        #
        # types[i] = type ID of the i'th tok in utterance[u]
        types = self.tok2type[self.starttok[u]:self.endtok[u]]
        #
        # mu[i,:] = mean vector of the i'th token in utterance[u]
        mu = self.model['mu'][types,:]
        #
        # var[i,:] = vector of variances of the i'th token in utterance u
        var = self.model['var'][types,:]
        #
        # A[i,j] = probability of a transition from the i'th to the j'th tok in utterance u
        A = np.array([[ self.model['tpm'][i,j] for j in types] for i in types])
        #
        return(mu,var,A,types)

    # PROBLEM 4.0
    #
    # Calculate the Gaussian surprisal in each frame.
    # We use "surprisal" (negative log probability), instead of probability, because it
    # suffers from fewer floating-point roundoff errors when you move from one computer to another.
    # surprisal[t,i] = 0.5* sum_d ((mfcc[t,d]-mu[i,d])^2/var[i,d] + log(2*pi*var[i,d]))
    def set_surprisal(self):
        self.surprisal = np.zeros((self.endframe[-1], self.maxtoks))
        for u in range(self.nutts):
            (mu,var,A,types) = self.get_params_for_utt(u)
            for t in range(self.startframe[u],self.endframe[u]):
                for i in range(self.ntoks[u]):
                    # TODO: subtract the mean, square, and divide by the variance
                    zsquared = np.zeros(self.nfeats) # INCORRECT!
                    temp = np.square(self.mfcc[t,:]- mu[i,:])
                    temp = temp/var[i,:]
                    zsquared = 0.5*(temp + np.log(2*np.pi*var[i,:]))
                    self.surprisal[t,i] = sum(zsquared) 

    # PROBLEM 4.1
    #
    # Calculate the scaled alpha (alphahat):
    # alphahat[t,i] = Pr(tok[frame t]=i|x[0],...,x[t-1],x[t]),
    #   where t is the frame index, and m indicates the m'th tok within each utterance.
    #
    def set_alphahat(self):
        self.alphahat = np.zeros((self.endframe[-1],self.maxtoks))
        for u in range(self.nutts):
            (mu,var,A,types) = self.get_params_for_utt(u)
            # Initialize the forward algorithm
            self.alphahat[self.startframe[u],0] = 1
            # Iterate the forward algorithm
            for t in range(self.startframe[u],self.endframe[u]):
                for j in range(self.ntoks[u]):
                    for i in range(self.ntoks[u]):
                        # TODO: accumulate alphahat[t,j] from alphahat[t-1,i]
                        self.alphahat[t,j] += self.alphahat[t-1,i]*A[i,j]*np.exp(-1*self.surprisal[t,j]) # INCORRECT!
                # TODO: normalize alphahat[t,:] so that it adds up to one
                self.alphahat[t,:] /= sum(self.alphahat[t,:]) # INCORRECT!
                
    # PROBLEM 4.2
    #
    # Calculate the scaled beta:
    # betahat[t,i] = g[t] Pr(x[t+1],...,x[T-1]|tok[t]=i), where g[t] ensures summation to 1
    def set_betahat(self):
        self.betahat = np.zeros((self.endframe[-1],self.maxtoks))
        for u in range(self.nutts):
            (mu,var,A,types) = self.get_params_for_utt(u)
            # Initialize the backward algorithm for this utterance
            self.betahat[self.endframe[u]-1,self.ntoks[u]-1] = 1
            for t in range(self.endframe[u]-2,self.startframe[u]-1,-1):
                for i in range(self.ntoks[u]):
                    for j in range(self.ntoks[u]):
                        # TODO: accumulate betahat[t,i] from betahat[t+1,j]
                        self.betahat[t,i] += A[i,j]*np.exp(-1*self.surprisal[t+1,j])*self.betahat[t+1,j] 
                # TODO: normalize betahat[t,:] so that it adds up to one
                self.betahat[t,:] /= sum(self.betahat[t,:])
        
    # PROBLEM 4.3
    #
    # Calculate the gamma probability, from alphahat and betahat
    def set_gamma(self):
        self.gamma = np.zeros((self.endframe[-1],self.maxtoks))
        for u in range(self.nutts):
            for t in range(self.startframe[u],self.endframe[u]):
                # TODO: multiply alphahat and betahat
                self.gamma[t,:] = self.alphahat[t,:]*self.betahat[t,:]# INCORRECT!
                # TODO: normalize gamma[t,:] so that it adds up to one
                self.gamma[t,:] /= sum(self.gamma[t,:])
                                           
    # PROBLEM 4.4
    #
    # Calculate xi, from alphahat, betahat, tpm, and likelihood
    def set_xi(self):
        self.xi = np.zeros((self.endframe[-1],self.maxtoks,self.maxtoks))
        for u in range(self.nutts):
            (mu,var,A,types) = self.get_params_for_utt(u)
            for t in range(self.startframe[u],self.endframe[u]-1):
                for i in range(self.ntoks[u]):                
                    for j in range(self.ntoks[u]):                
                # TODO: calculate xi[t,i,j] from alphahat[t,i] and betahat[t+1,j]
                        self.xi[t,i,j] = self.alphahat[t,i]*A[i,j]*np.exp(-1*self.surprisal[t+1,j])*self.betahat[t+1,j]
                # TODO: normalize so that sum over all (i,j) of xi[t,i,j] is 1
                self.xi[t,:,:] /= sum(sum(self.xi[t,:,:]))
                        
    # PROBLEM 4.5
    #
    # Re-estimate mu using gamma and mfcc
    def set_mu(self):
        self.mu = np.zeros((self.ntypes,self.nfeats))
        numerator = np.zeros((self.ntypes,self.nfeats))
        denominator = np.zeros((self.ntypes,1))
        for u in range(self.nutts):
            (mu,var,A,types) = self.get_params_for_utt(u)
            for i in range(self.ntoks[u]):
                for t in range(self.startframe[u],self.endframe[u]):
                    # TODO: accumulate the EM numerator
                    numerator[types[i],:] += self.gamma[t,i]*self.mfcc[t,:] # INCORRECT!
                    # TODO: accumulate the EM denominator
                    denominator[types[i]] += self.gamma[t,i] # INCORRECT!
        for i in range(self.ntypes):
            # TODO: re-estimate mu[i,:] as numerator[i,:] / denominator[i]
            self.mu[i,:] = numerator[i,:]/denominator[i]# INCORRECT!

    # PROBLEM 4.6
    #
    # Re-estimate var using gamma and mfcc
    def set_var(self):
        self.var = np.zeros((self.ntypes,self.nfeats))
        numerator = np.zeros((self.ntypes,self.nfeats))
        denominator = np.zeros((self.ntypes,1))
        for u in range(self.nutts):
            (mu,var,A,types) = self.get_params_for_utt(u)
            for i in range(self.ntoks[u]):
                for t in range(self.startframe[u],self.endframe[u]):
                    # TODO: accumulate the EM numerator
                    numerator[types[i],:]+=self.gamma[t,i]*np.square(self.mfcc[t,:]-self.mu[types[i],:]) # INCORRECT!
                    # TODO: accumulate the EM denominator
                    denominator[types[i]] += self.gamma[t,i] # INCORRECT!
        for i in range(self.ntypes):
            # TODO: re-estimate var[i,:] as numerator[i,:]/denominator[i]
            self.var[i,:] = numerator[i,:]/denominator[i] # INCORRECT!
            # For numerical stability: make sure self.var is at least 1e-4
            self.var[i,:] = np.maximum(1e-4, self.var[i,:])

    # PROBLEM 4.7
    #
    # Re-estimate transition probabilities:
    # Set numerator[i,j] = sum_t of xi[t,i,j]
    # Set denominator = sum_j sum_t xi[t,i,j]
    #   then set tpm = numerator/denominator
    def set_tpm(self):
        self.tpm = np.zeros((self.ntypes,self.ntypes))
        numerator = np.zeros((self.ntypes,self.ntypes))
        for u in range(self.nutts):
            (mu,var,A,types) = self.get_params_for_utt(u)
            for t in range(self.startframe[u],self.endframe[u]):
                for i in range(self.ntoks[u]):
                    for j in range(self.ntoks[u]):
                        # TODO: accumulate EM numerator[i_type,j_type] from xi[t,i,j]
                        numerator[types[i],types[j]] += self.xi[t,i,j] # INCORRECT!
        for i in range(self.ntypes):
            # TODO: tpm[i,:] = numerator[i,:], normalized so that it sums to one
            self.tpm[i,:] = numerator[i,:]/sum(numerator[i,:]) # INCORRECT!

    # PROBLEM 4.8
    #
    # Actually, you don't have to do anything for this.
    # This code just saves the trained model in a JSON file
    def set_trainedmodel(self):
        dict = {}
        dict['mu'] = json.dumps([ list(row) for row in self.mu ])
        dict['var'] = json.dumps([ list(row) for row in self.var ])
        dict['tpm'] = json.dumps([ list(row) for row in self.tpm ])
        dict['phones'] = self.model['phones']
        self.trainedmodel = json.dumps(dict)
        with open('models/model%2.2d.json'%(self.epoch+1),'w') as f:
            f.write(self.trainedmodel)
