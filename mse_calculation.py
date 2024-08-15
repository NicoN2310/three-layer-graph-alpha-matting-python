import numpy as np
from PIL import Image


# Note: Image is the doll and the trimap is the small one from https://www.alphamatting.com
# The best matte is the one from the same website for the LFPNet Model

def evaluation_mse(pred, target, trimap):
    error_map = np.abs(pred - target) / 255.0
    loss = np.sum(error_map**2 * np.float32(trimap != 0.0)) / np.sum(np.float32(trimap != 0.0))
    return loss * 100 

if __name__ == '__main__':
    trimap = np.array(Image.open('imgs/test_trimap.png').convert('L'), dtype=np.float32)
    python_matte = np.array(Image.open("imgs/test_matte.png").convert('L'), dtype=np.float32)
    best_matte = np.array(Image.open("imgs/best_matte.png").convert('L'), dtype=np.float32)
    
    
    print(f"MSE between Python Matte and Best Matte: {evaluation_mse(python_matte, best_matte, trimap)}")
