'''
just a class to store a rolling average 

Returns:
    [type]: measure
'''
class RollingMeasure(object):
    def __init__(self) -> None:
        super().__init__()
        self.measure = 0.0
        self.iter = 0

    def __call__(self, measure):
        # first time call initial 
        if self.iter == 0:
            self.measure = measure
        else:
            self.measure = (1.0 / self.iter * measure) + (1 - 1.0 / self.iter) * self.measure
        
        self.iter += 1

        return self.measure