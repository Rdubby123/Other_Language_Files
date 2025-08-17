x=25
y=50
xdir=1
ydir=1
rate=0
ballSpeed=5

def setup():
    size(800,800)
    frameRate(60)
def draw():
    global x,y,xdir,ydir,rate,ballSpeed
    if(ballSpeed>=10):
        ballSpeed/=2
        draw() # go back
        
# animation logic
    if(rate%4==0):
        drawPlanets(10)
    if(rate%ballSpeed==0):
        background(0)
    drawComet(x,y)
    rate+=1
    
# movement logic
    if(xdir==1):
        x+=ballSpeed
    elif(xdir==-1):
        x-=ballSpeed
    if(ydir==1):
        y+=2*ballSpeed
    elif(ydir==-1):
        y-=ballSpeed
        
# direction logic
    if(x>=775):
        xdir=-1
    elif(x<=25):
        xdir=1
    if(y>=775):
        ydir=-1
    elif(y<=25):
        ydir=1


def drawPlanets(numPlanets):
    for i in range(numPlanets):
        x = random(0,800)     
        y = random(0,800)
        R = random(0,256)        
        G = random(0,256)
        B = random(0,256)
        stroke(R,G,B)
        randWeight = random(7,25) 
        strokeWeight(randWeight)
        point(x,y)    
    strokeWeight(1)
    noStroke()  
    noFill()
             
def drawComet(x,y):
    fill(250,0,50)
    stroke(0)
    circle(x, y, 80)
    noStroke()
    
    
