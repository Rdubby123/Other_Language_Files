# Rayan, Ryan, Christian, Jayden

diameter = 120
radius = diameter/2
x = radius
speed = 5
direction = 1

def setup():
    size(480, 240)
    fill(125,100,50)
def draw():
    global x, direction
  
    background(0)
    if(direction == 1):
       x = x + speed # Increase the value of x
    else:
       x = x - speed # Increase the value of x
       if(x-radius == 0):
           direction = 1
    circle(x, height/2, diameter)
    if(x + radius == width):
        x = width-radius
        direction = -1
