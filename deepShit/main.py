import numpy as np

class Glass:
    def __init__(self, something):
        """
        Deep state of the art python glassifier implementation. 
        
        Params:
        something : could be anything.
        """
        self.s = something
        self.data = None
        
    def make_data_great_again(self, data):
        ndata = np.zeros((10000))
        ndata[:data.flatten().shape[0]] = data.flatten()[:10000]
        if ndata.max() != ndata.min():
            ndata = (ndata - ndata.min()) / (ndata.max() - ndata.min())
        else:
            ndata = ndata - ndata.min()
        ndata = ndata.reshape((100, 100))
        return ndata
    
    def get_data(self):
        if isinstance(self.s, np.ndarray):
            self.data = self.make_data_great_again(self.s)
            return self.data
        else:
            try:
                data = np.array(self.s)
                self.data = self.make_data_great_again(data)

                return self.data

            except Exception:
                pass

    def glassify(self):
        self.get_data()
        glasses = np.load('data/glasses.npy')
        
        self.data += glasses
        
        return self.data
