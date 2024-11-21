# ANPR

STEP-1 :!pip install easyocr
        !pip install imutils
        Requirement already satisfied: easyocr in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (1.7.1)
        Requirement already satisfied: torch in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from easyocr) (2.2.2+cu118)
        Requirement already satisfied: torchvision>=0.5 in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from easyocr) (0.17.2)
        Requirement already satisfied: opencv-python-headless in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from easyocr) (4.9.0.80)
        Requirement already satisfied: scipy in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from easyocr) (1.13.0)
        Requirement already satisfied: numpy in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from easyocr) (1.26.4)
        Requirement already satisfied: Pillow in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from easyocr) (10.3.0)
        Requirement already satisfied: scikit-image in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from easyocr) (0.22.0)
        Requirement already satisfied: python-bidi in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from easyocr) (0.4.2)
        Requirement already satisfied: PyYAML in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from easyocr) (6.0.1)
        Requirement already satisfied: Shapely in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from easyocr) (2.0.3)
        Requirement already satisfied: pyclipper in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from easyocr) (1.3.0.post5)
        Requirement already satisfied: ninja in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from easyocr) (1.11.1.1)
        Requirement already satisfied: filelock in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from torch->easyocr) (3.13.3)
        Requirement already satisfied: typing-extensions>=4.8.0 in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from torch->easyocr) (4.11.0)
        Requirement already satisfied: sympy in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from torch->easyocr) (1.12)
        Requirement already satisfied: networkx in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from torch->easyocr) (3.3)
        Requirement already satisfied: jinja2 in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from torch->easyocr) (3.1.3)
        Requirement already satisfied: fsspec in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from torch->easyocr) (2024.3.1)
        Requirement already satisfied: six in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from python-bidi->easyocr) (1.16.0)
        Requirement already satisfied: imageio>=2.27 in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from scikit-image->easyocr) (2.34.0)
        Requirement already satisfied: tifffile>=2022.8.12 in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from scikit-image->easyocr) (2024.2.12)
        Requirement already satisfied: packaging>=21 in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from scikit-image->easyocr) (24.0)
        Requirement already satisfied: lazy_loader>=0.3 in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from scikit-image->easyocr) (0.4)
        Requirement already satisfied: MarkupSafe>=2.0 in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from jinja2->torch->easyocr) (2.1.5)
        Requirement already satisfied: mpmath>=0.19 in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (from sympy->torch->easyocr) (1.3.0)
        Requirement already satisfied: imutils in c:\users\dell\appdata\local\programs\python\python311\lib\site-packages (0.5.4)

        
STEP-2: import cv2 
        from matplotlib import pyplot as plt
        import numpy as np
        import imutils
        import easyocr
        img = cv2.imread('image1.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
        <matplotlib.image.AxesImage at 0x1d4c8f29310>
        No description has been provided for this image

STEP-3 :bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
        edged = cv2.Canny(bfilter, 30, 200) #Edge detection
        plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
        <matplotlib.image.AxesImage at 0x1d4c8f8a650>
        No description has been provided for this image

        
STEP-4: keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(keypoints)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
                location = None
                for contour in contours:
                    approx = cv2.approxPolyDP(contour, 20, True)
                    if len(approx) == 4:
                        location = approx
                        break
                location        
                      array([[[122, 219]],
                      
                             [[246, 227]],
                      
                             [[252, 200]],
                      
                             [[132, 191]]], dtype=int32)
                      mask = np.zeros(gray.shape, np.uint8)
                      new_image = cv2.drawContours(mask, [location], 0,255, -1)
                      new_image = cv2.bitwise_and(img, img, mask=mask)
                      plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
                      <matplotlib.image.AxesImage at 0x1d4ca05fd90>
                      No description has been provided for this image
STEP-5: (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        <matplotlib.image.AxesImage at 0x1d4ca17b450>
        No description has been provided for this image
STEP-6: reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        result
        Neither CUDA nor MPS are available - defaulting to CPU. Note: This module is much faster with a GPU.
        [([[6, 4], [128, 4], [128, 34], [6, 34]], 'HR.26 BR.9044', 0.5728024180498563)]
        text = result[0][-2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, 
        fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        <matplotlib.image.AxesImage at 0x1d4c9feae90>
        No description has been provided for this image
 
