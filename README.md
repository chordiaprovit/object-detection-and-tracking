# object-detection-and-tracking
object detection and tracking
This repo uses COCO dataset for object detection. The pre trained model used is ssd_mobilenet_2018_07_03. 
In the video it can be seen that whenever car crosses red line, the counter is increased. Also white and clack color cars are identified. 
All the cars crossing the lines are not detected because either they are out of frame or the detection of vehicle stops. 

To detect if the car has crossed the line, center of detected car is used. Whenever the y-axis of the center is greater than y coordinate of red line, the counter is incremented.

Issues and resolution:
However, due to instable detetion of cars, the centers keeps on getting moved and sometimes the car was counted twice. To overcome that; frame number of deteted car and x,y coordinates are tracked frame by frame so that a car is not counted twice. 
rpi.py uses version greater than tensorflow 2. This is used to do object detectin using raspberry pi. The objects are detected fine,but due to limited processing power of raspberrypi, it hangs freuently.
