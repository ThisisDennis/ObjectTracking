import time, sys, cv2, json
from datetime import date
from datetime import datetime
from detection import detection
import numpy as np
import scipy.spatial.distance as scipydist
from munkres import Munkres
from tracks import track
from simulation import SimulationEngine


#initalize tracking
strikelimit = 6
threshold = 50
detections = []
delay = 1
lostTracks = []
simulation = SimulationEngine.gameEngine()

switch = 'Tracking\n0 : OFF \n1 : ON'

'''Write data into json file

Parameters
----------
filePathName: path and file Name
data: data to write

'''
def toJson(filePathName, data):

    with open(filePathName, 'w') as fp:
        json.dump(data, fp)

''' 
Save trackdata 

Parameters
----------
input: tracks with data to save

'''

def saveTracks(input):
    print("saving tracks...")
    now = datetime.now()
    filename = str(date.today())+ "_" + str(now.strftime("%H-%M"))
    filePathName = './' + './output' + '/' + filename + '.json'
    d = {}
    da = {}
    data = {}

    for x in range(0, len(input)-1):
        path = input[x]
        d['fps'] = path.fps
        for point in range(0, len(path.track)-1):

            vel = path.velocity[point]
            acc = path.acceleration[point]
            dirVec= path.directionVect[point]
            pos = path.track[point]
            pred = path.track[point]

            d['velocityx'] =vel[0]
            d['velocityy'] = vel[1]
            d['accelerationx'] = acc[0]
            d['accelerationx'] = acc[1]
            d['directionVectorx'] = dirVec[0]
            d['directionVectory'] = dirVec[1]
            d['datetime'] = str(path.datetime[point])
            d['position x'] =pos[0]
            d['position y'] = pos[1]
            d['predicted x'] = pred[0]
            d['predicted y'] = pred[1]
            da[point] = d
            d={}

        data[x]=da

    toJson(filePathName, data)

'''does nothing'''
def nothing(x):
    pass

'''
Assigns two point-arrays, based on euclidean distance and Munkres-algorithm.

Parameters
----------
rowMat: array 1
columnMat: array 2

Returns
----------
assignment: array of assignment indieces
distance: distance matrix
'''

def assign(rowMat, columnMat):

    distance = scipydist.cdist(rowMat, columnMat, 'euclidean') #predicted row, existing column
    dist = distance.copy()
    m = Munkres()
    return m.compute(dist), distance

'''
Swap row and column in Array of 2D-tupels.

Parameters
----------
mat: Input-array of Tupels

Returns
----------
swapped tupel-array
'''

def swap(mat):
    buf = []
    for element in mat:
        buf.append((element[1], element[0]))

    return buf

'''
This function fixes rectangular shapes and assigs two arrays.

Parameters
----------
row: array 1
column: array2

Returns
----------
assignment: 2D-Array with indices to assign arrays (array 1, array 2)
distances: 2D-Array filled with distances, of array 1 (row) and array 2 (column)
'''

def rectAssing(row, column):
    #mat1:z.b. last, mat2, existing

    if len(row) > len(column):
        assignment, distance = assign(column, row)
        assignment = swap(assignment)
        distance = np.transpose(distance)
        # sort
        assignment.sort(key=lambda x: x[0])

    else:
        assignment, distance = assign(row, column)

    return assignment, distance


'''
This function tries to find the right tracks, predict the next position and correct false estimations.
It also creates new tracks, if there are new detections.
Parameters
----------
newDetected: 2D-Array with new detections (x, y)
assignment: 2D-Array with indieces to assigned points (old detection, new detection)
distance: 2D-Array with distances between assigned points
'''

def correctPosition(newDetected = None, assignment= None, distance = None):

    cols = []

    if distance is not None:

        # row= prediction in previous detections, column = new detections

        for row, column in assignment:
            cols.append(column)

            if distance[row, column] > threshold:

                #correct false estimation and predict new postion

                correction = scipydist.pdist([detections[row].pos, newDetected[column]], 'euclidean')
                if threshold > correction:
                    distance[row, column] = correction

                    pos = newDetected[column]
                    detections[row].predict(pos)
                    detections[row].strk(False)

                #correct lost tracks, and predict new position
                #elif strikelimit >= detections[row].strike:
                else:
                    o = detections[row]
                    pred = o.pred
                    o.predict(pred)
                    o.strk()
                    cv2.circle(image, o.pos, 1, (0,0,200),20, 5)

                # one strike more

            #predict new postion
            else:

                pos = newDetected[column]
                detections[row].predict(pos)
                detections[row].strk(False)


    # appending all new
    #invert column list
    mask = np.ones(len(newDetected), np.bool)
    mask[cols] = 0

    for x in range(0, len(newDetected)):
        if mask[x]:
            mp = newDetected[x]

            detect = track(mp, len(detections), dt, fps)
            detect.predict(mp)
            detections.append(detect)



'''
This function tries to identify detections and concatenate old and new in tracks.
----------
xy: this is a 2D-Array of detected points
frame: 2D-Array of an image

Returns
----------
nothing yet

'''
def buildTracks(xy, frame):

    oldprediction = []

    #first detection
    if len(detections) == 0:
        correctPosition(xy)

    #next detections
    else:

        delete = []
        #check for lost tracks
        for x in range(0, len(detections)):
            obj = detections[x]

            if obj.strike >= strikelimit:
                delete.append(x)
            else:
                oldprediction.append(obj.pred)

        #delete lost tracks
        for delete in sorted(delete, reverse=True):
            lostTracks.append(detections[delete])
            del detections[delete]

        if xy is None:
            return

        #assign and correct new detections
        if len(detections)!=0 and len(xy)!= 0:
            assignment, distance = rectAssing(oldprediction, xy)
            correctPosition(xy, assignment, distance)


'''
This function is used to set change process-setting via userinput.

Parameters
----------
blobs: is a detection-object, we need to update the detection-setting

Returns
----------
Boolean: decision if tracking should be acitvated or not

'''
def update(blobs):
    global bt, dk, strikelimit, threshold, dt
    #mk = cv2.getTrackbarPos("Morph Kernel", "Settings")
    bt = cv2.getTrackbarPos("Bin Threshold (%)", "Settings")
    dk = cv2.getTrackbarPos("Distance Kernel", "Settings")
    strikelimit = cv2.getTrackbarPos("Reject track at:", "Settings")
    threshold = cv2.getTrackbarPos("Assignm. Distance:", "Settings")
    blobs.settings(3, bt/100, dk)

    #simulation
    asize = cv2.getTrackbarPos("number of agents", "Settings")
    simulation.agentNumbers(asize)

    return cv2.getTrackbarPos(switch, "Settings")


'''
This function is used to set initialize the trackbar for userinputs.

Parameters
----------
blobs: is a detection-object, we need to set start values
'''

def trackbar(blobs):

    cv2.namedWindow('Settings')

    wnd = 'Kernel'

    # create trackbar for simulation
    cv2.createTrackbar("number of agents", "Settings", 1, 200, nothing)

    #create Trackbar for object detection
    #cv2.createTrackbar("Morph Kernel", "Settings", 1, 20, nothing)
    cv2.createTrackbar("Bin Threshold (%)", "Settings", 2, 100, nothing)
    cv2.createTrackbar("Distance Kernel", "Settings", 1, 20, nothing)

    #create trackbar for trackbuilding
    cv2.createTrackbar(switch, "Settings", 0, 1, nothing)
    cv2.createTrackbar("Reject track at:", "Settings", 1, 20, nothing)
    cv2.createTrackbar("Assignm. Distance:", "Settings",10,100, nothing)

    #set Values
    cv2.setTrackbarPos("number of agents", "Settings", simulation.getAgentsize())

   # cv2.setTrackbarPos("Morph Kernel", "Settings", blobs.kerSize)
    cv2.setTrackbarPos("Bin Threshold (%)", "Settings", int(blobs.binaryThresh*100))
    cv2.setTrackbarPos("Distance Kernel", "Settings", blobs.distKernel)

    cv2.setTrackbarPos("Reject track at:", "Settings", strikelimit)
    cv2.setTrackbarPos("Assignm. Distance:", "Settings",threshold)
    cv2.setTrackbarPos(switch, "Settings", 1)

    #set min values
    cv2.setTrackbarMin("number of agents", "Settings", 1)

    #cv2.setTrackbarMin("Morph Kernel", "Settings", 1)
    cv2.setTrackbarMin("Bin Threshold (%)", "Settings", 1)
    cv2.setTrackbarMin("Distance Kernel", "Settings", 1)

    cv2.setTrackbarMin("Reject track at:", "Settings", 1)
    cv2.setTrackbarMin("Assignm. Distance:", "Settings", 10)


#initalize
blobs = detection()
w = h = 5
trackbar(blobs)
framebuffer = []
cap = cv2.VideoCapture("Tomaten2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
#dt = fps*0.25/fps#fps*0.5/fps
#dt = 1
pSpeed = 1/fps
dt = pSpeed * 2
#dt*2

ret, frame = cap.read()
frNr = 0
image = frame
objects = []

#face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(1):


    simulation.rederLoop()
    img = simulation.im

    #ret, img = cap.read()
    #img = cv2.blur(img,(10,10))
    framebuffer.append(img)

    trackswitch = update(blobs)

    if frNr%delay==0:
        pass
        image = framebuffer.pop(0)

    objects, dist = blobs.detectObjects(image)

    #
    # #faceversion
    # dist = image
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    #
    # for (x, y, w, h) in faces:
    #     print(x, y)
    #     objects.append((int((x+w/2)), int((y+h/2))))
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # #####


    if trackswitch:
        buildTracks(objects, image)

    # draw stored position, predicted position and calculated velocity
    for det in detections:
        cv2.circle(image, det.pos, 5, det.color)
        cv2.circle(image, det.pred, 10, (200,200,0))
        if len(det.veloVec)>0:
            v = det.veloVec[len(det.veloVec)-2]
            cv2.arrowedLine(image, det.pos, v, (150,20,20))
        tracks = det.track
        s=0
        # if len(tracks)>7:
        #     s = len(tracks)-7

        # draw tracks
        for x in range(s, len(tracks) - 1):
            pos = tracks[x]
            pred = tracks[x + 1]
            x = int(pos[0])
            y = int(pos[1])
            px = int(pred[0])
            py = int(pred[1])
            color = det.color
            cv2.line(image, (x, y), (px, py), color)
    # draw detected objects
    if objects is not None:
        for obj in objects:
            cv2.circle(image, obj, 5, (0,0,200))

    # draw detected cores
    if dist is not None:
        pass
        cv2.imshow('Distance transformed Image', dist)

    # show image and draws
    if image is not None:
        cv2.imshow("Detected", image)
    else:
        break

    time.sleep(pSpeed)

    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break

    elif k == 27:
        break
    frNr += 1
input = lostTracks
for det in detections:
    input.append(det)
saveTracks(input)
cv2.destroyAllWindows()