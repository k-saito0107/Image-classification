from PIL import Image


class Resize():
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def __call__(self, img):
        re_img = img.resize(self.width, self.height)

        return re_img


