import cv2, numpy as np
import random
import datetime
'''
Class to handle information, predictions about tracks

Parameters
----------
object: object for access to variables, without instances
'''


class track(object):

    '''
    Constructor for initialization of variables and Kalman-filter

    Parameters
    ----------
    pos: tupel, with position information (x, y)
    id: integer, used as identifier for tracks
    binThresh: float, binary threshold default value
    dt: float, delta time
    '''


    def __init__(self, pos, id, dt=0.004, fps = 0):
        x = pos[0]
        y = pos[1]
        mp = np.array([np.float32(x), np.float32(y)])
        self.dt = dt

        self.kalman = cv2.KalmanFilter(4, 2, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.001
        self.kalman.controlMatrix = np.array([0,1], np.float32)

        self.kalman.statePost[0][0] = x
        self.kalman.statePost[1][0] = y

        self.pos = pos
        self.meas = []
        self.kalman.correct(mp)
        self.pred = self.kalman.predict()
        self.pred = (int(self.pred[0]), int(self.pred[1]))

        self.strike = 0
        self.id = id
        self.color = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
        self.track = []
        self.veloVec = []
        self.acceleration = []
        self.velocity = []
        self.directionVect = []
        self.datetime=[]
        self.fps = fps
        # self.kalman.correct(mp)
        self.correct = []
        self.false = []

        #print(self.pos, self.pred, flag)


    '''
    Function to correct Kalman-filter and predict positions and velocity.

    Parameters
    ----------
    meas: 2D-position-Tupel, used to correct kalman and predict new state
    '''

    def predict(self, meas):
        'time'
        date = datetime.datetime.now()
        self.datetime.append(date)

        'kalman'
        #print(self.kalman.controlMatrix)
        self.false.append(self.pred)
        px = self.pos[0]
        py = self.pos[1]
        self.pos = meas
        x = meas[0]
        y = meas[1]
        mp = np.array([np.float32(x), np.float32(y)])

        self.track.append(meas)

        #velocity and acceleration
        vx = (px - x) / self.dt
        vy = (py - y) / self.dt
        if x != 0:
            ax = (vx*vx)/(2*x)
        else:
            ax = 0
        if y != 0:
            ay = (vy*vy)/(2*y)
        else:
            ay = 0
        self.acceleration.append([ax, ay])
        self.velocity.append([vx, vy]) #ohne Startpunkt
        self.directionVect.append([x+px, y+py])

        ############


        self.kalman.correct(mp)

        self.pred = self.kalman.predict()

        'velocity'

        if np.shape(self.pred)[0]>2:

            a = x - int(vx)
            b = y - int(vy)
            self.veloVec.append((a,b))

        self.pred = (int(self.pred[0]), int(self.pred[1]))


        if self.pred == (0,0):
            print(mp, self.pred, self.id, self.strike)


    '''
    Function to count lost track in a row.

    Parameters
    ----------
    flag: boolean, true means track lost, false reduce strikes
    '''
    def strk(self, flag=True):
        if flag:
            self.strike += 1
        else:
            if self.strike > 0:
                self.strike =0
