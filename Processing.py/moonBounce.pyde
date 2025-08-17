x=25
y=50
xpol=1
ypol=1
rate=0
ballSpeed=5

def setup():
    size(800,800)
    frameRate(60)
def draw():
    global x,y,xpol,ypol,rate,ballSpeed
    if(ballSpeed>=10):
        ballSpeed/=2
        draw() # go back
# animation logic
    if(rate%4==0):
        drawPlanets(10)
    if(rate%ballSpeed==0):
        background(0)
    drawMoon()
    rate+=1
    
# movement logic
    if(xpol==1):
        x+=ballSpeed
    elif(xpol==-1):
        x-=ballSpeed
    if(ypol==1):
        y+=2*ballSpeed
    elif(ypol==-1):
        y-=ballSpeed
        
# polarity logic
    if(x>=775):
        xpol=-1
    elif(x<=25):
        xpol=1
    if(y>=775):
        ypol=-1
    elif(y<=25):
        ypol=1


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

def drawMoon():
    fill(255)
    stroke(0)
    circle(x, y, 80)
    circle(x+19, y-15, 6)        
    circle(x-13, y-13, 6)
    circle(x-22, y+1, 9)       
    circle(x+10, y, 10)
    circle(x, y+20, 18)
    noStroke()
        
    
