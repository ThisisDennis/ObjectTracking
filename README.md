# Object tracking with kalman-filter

An old project from student days that is not being developed further. With this it is possible to track moving objects, even if they cross each other or disappear for a short time. It uses among others the Kalman filter and the munkres algorithm.

## Note
Originally needs python 2.7 but works with 3 also.
In python 2.7 you may change the line 144 in detection.py from `contours, hierarchy = cv2.findContours(dist_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)` to `img, contours, hierarchy = cv2.findContours(dist_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)`
If you use python 2.7 you want to change the setup.sh or install the requierments manually.

## Usage

In startTracking.py (around line 370) you can change, if you want to use the embedded simulation or a video. You could also grab a camera stream.

Uncomment the following for simulation / comment for video:
```
    simulation.rederLoop() 
    img = simulation.im 
```
Uncomment the following for video / comment for simulation:
```
    #ret, img = cap.read()
```
in line 351 you can choose your video:
`cap = cv2.VideoCapture("Tomaten2.mp4")`
## Userinterface
![userinterface](https://user-images.githubusercontent.com/24722709/129219401-2654ccaa-dc24-4506-904e-8c8bea598708.png)

## Transformed Image
![distancewindow](https://user-images.githubusercontent.com/24722709/129219395-a61e3ac5-4e19-41d3-a9ed-ba0e92a31d69.png)

## Simulation detection:
![detectedwindow](https://user-images.githubusercontent.com/24722709/129219400-fe4ab271-ccf3-4794-911c-1bdc980fdf04.png)

## Video detection
![Tomate](https://user-images.githubusercontent.com/24722709/129219404-d326eee1-7f2e-4f63-950b-91f17168603d.png)

