import numpy as np

def preprocess(image, constant):
    image = image[34:194, :, :] # 160, 160, 3
    image = np.mean(image, axis=2, keepdims=False) # 160, 160
    image = image[::2, ::2] # 80, 80
    image = image/256
    image = image - constant/256# remove background
    return image

class SumTree:
    
    def __init__(self, capacity):
        
        self.capacity = capacity
        # the first capacity-1 positions are not leaves
        self.vals = [0 for _ in range(2*capacity - 1)] # think about why if you are not familiar with this
        
    def retrive(self, num):
        '''
        This function find the first index whose cumsum is no smaller than num
        '''
        ind = 0 # search from root
        while ind < self.capacity-1: # not a leaf
            left = 2*ind + 1
            right = left + 1
            if num > self.vals[left]: # the sum of the whole left tree is not large enouth
                num -= self.vals[left] # think about why?
                ind = right
            else: # search in the left tree
                ind = left
        return ind - self.capacity + 1
    
    def update(self, delta, ind):
        '''
        Change the value at ind by delta, and update the tree
        Notice that this ind should be the index in real memory part, instead of the ind in self.vals
        '''
        ind += self.capacity - 1
        while True:
            self.vals[ind] += delta
            if ind == 0:
                break
            ind -= 1
            ind //= 2
            
    