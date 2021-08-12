import cv2
import numpy as np

'''
Detection class to handle detection process  single frames.
'''


class detection:


    '''
    This function is for presentation purpose, to show how properly we can segemnt overlapping .

    Parameters
    ----------
    frame: image array, needed to process watershed algorithm on
    '''

    def watershed(self, frame):  ### just to show the quality of seperation
        #first approximation
        #frame = cv2.imread('water_coins.jpg')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)
        # Marker labelling

        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        markers = cv2.watershed(frame, markers)
        frame[markers == -1] = [0, 0, 255]


        #cv2.imwrite('watershedout/'+str(self.nr) + '.png', frame)
        self.nr += 1

        cv2.imshow("watershed", frame)

    '''
    Constructor for initialization

    Parameters
    ----------
    kerSize: integer, kernel size for sturcture element
    '''

    def __init__(self, kerSize = 3):

        self.kerSizeBuf =kerSize
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kerSize, kerSize))
        self.bgModel = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.settings()

        self.nr = 0

        self.ext_nb = 1
        self.ext_thresh = 1500


    '''
    This function is used to set user-inputs.

    Parameters
    ----------
    morphKernel: integer, should be 3 (maybe i remove this parameter)
    distMaskSize: integer, used as mask Size for distance transfrom
    binThresh: float, binary threshold
    distkernel: integer, used to dilate results
    '''

    def settings(self, distMaskSize = 3, binThresh = 0.89, distkernel= 3): #binThresh = 0.4 simulation: 0.89
        self.kerSize = 3

        self.maskSize = distMaskSize
        self.binaryThresh = binThresh
        self.distKernel = distkernel

        #only on change
        if self.kerSizeBuf!= self.kerSize:
            self.kernel = cv2.getStructuringElement(cv2.MORPH_CLOSE, (self.kerSize, self.kerSize))
        self.kerSizeBuf = self.kerSize

    '''
    This function is needed to detect Objects in frame.

    Parameters
    ----------
    frame: image-array, which hopefully contains objects to detect
    
    Returns
    ----------
    detections: array of 2D-points containing detected object-positions
    dist: binary-image-array, containing centers of detected objects
    None, None: if nothing is found
    '''

    def detectObjects(self, frame):


        detections = []
        fgmask = self.bgModel.apply(frame)
        #cv2.imshow("fgmask", fgmask)

        if fgmask is None:
            return None, None
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel, iterations= 2)

        # perform distance transform algorithm

        dist = cv2.distanceTransform(fgmask, cv2.DIST_L2, self.maskSize)
        # normalize, for presentation purpose
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

        #cv2.imshow("distance", dist)
        _, dist = cv2.threshold(dist, self.binaryThresh, 1.0, cv2.THRESH_BINARY)
        # dilating the dist image

        kernel1 = np.ones((self.distKernel, self.distKernel), dtype=np.uint8)
        dist = cv2.dilate(dist, kernel1)


        #find contours
        dist_8u = dist.astype('uint8')

        contours, hierarchy = cv2.findContours(dist_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        for cnt in contours:
            M = cv2.moments(cnt)
            "new moment\n"
            #print(M)
            if M['m00']!= 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                detections.append((cx, cy))
        #self.test(frame)
        return detections, dist
