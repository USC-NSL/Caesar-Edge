class VideoReader:
    def Setup(self):
        pass

    def PreProcess(self, input):
        pass

    def Apply(self):
        pass 
        
    def PostProcess(self):
        pass 

    def log(self, s):
        print('[VideoReader] %s' % s)