import cv2
import os
from google.colab import files

class Cartoonizer:
    """Cartoonizer effect
    A class that applies a cartoon effect to an image.
    The class uses a bilateral filter and adaptive thresholding to create
    a cartoon effect.
    """
    def __init__(self):
        pass

    def render(self, img_rgb):
        img_rgb = cv2.imread(img_rgb)
        if img_rgb is None:
            raise ValueError(f"Image not found or unable to read: {img_rgb}")
        img_rgb = cv2.resize(img_rgb, (1366, 768))
        numDownSamples = 2  # number of downscaling steps
        numBilateralFilters = 50  # number of bilateral filtering steps

        # -- STEP 1 --
        img_color = img_rgb
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)

        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)

        # -- STEPS 2 and 3 --
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)

        # -- STEP 4 --
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)

        # -- STEP 5 --
        (x, y, z) = img_color.shape
        img_edge = cv2.resize(img_edge, (y, x))
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

        return cv2.bitwise_and(img_color, img_edge)

# Upload the image file
uploaded = files.upload()

# Assuming you only upload one image, get its file name
file_name = next(iter(uploaded))

# Initialize the cartoonizer and process the image
tmp_canvas = Cartoonizer()
res = tmp_canvas.render(file_name)

# Save the result
cv2.imwrite("Cartoon_version.jpg", res)

# Display the result
from google.colab.patches import cv2_imshow
cv2_imshow(res)

cv2.waitKey(0)
cv2.destroyAllWindows()