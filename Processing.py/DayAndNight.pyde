# Global Variables
time = 0
x = 150
y = 150

def setup():
    size(400, 400)
    background(120)
    frameRate(60)
    noStroke()

def draw():
    global time, x, y
    time = time + 1
    flag=1
    threeseconds = time/180
    if threeseconds % 2 == 0:
        flag = 0
    if flag == 0:
        NightTime()
        flag=1
    else:
        DayTime(x,y)
    
def DayTime(x, y):
    background(50, 70,150)
    house(x,y,100)
    makeRain(1,20)
    drawClouds(20)

def NightTime():
        background(0)
        drawMoon()
        drawPlanets(5)
        drawSpaceman()
        drawSpaceboard()
        
def house(x,y,houseSize=100):
    re = 150
    gre = 100 # just colors for houses
    bl = 140
    fill(0,100,0)
    square(0, y+75, houseSize*4)
    fill(170,170,100)
    circle(25, 25, 80)
    fill(re, gre, bl)
    square(x, y, houseSize)
    fill(250,220,50)
    stroke(40)
    square(x+houseSize/3, y+houseSize/6, houseSize/3)
    noStroke()
    fill(re,gre,bl)
    triangle(x-(0.25*houseSize), y, x+houseSize/2, y-0.75*houseSize, x+1.25*houseSize, y)
    fill(75,75,0)
    rect(x+.4*houseSize, y+.6*houseSize, houseSize/2-0.3*houseSize, houseSize*0.4) 
    fill(250,220,50)
    square(x+.45*houseSize, y+.65*houseSize, houseSize/10)
    fill(170,150,180)
    circle(x+.55*houseSize, y+0.85*houseSize, houseSize/13)
        
def drawMoon():
    fill(255)
    stroke(0)
    circle(25, 25, 80)
    circle(44, 10, 6)        
    circle(12, 12, 6)
    circle(3, 26, 9)       
    circle(35, 25, 10)
    circle(25, 45, 18)
    noStroke()
    
def drawClouds(numClouds):
    fill(255)
    noStroke()
    for i in range (numClouds*10):
        xrand = random(100,400)
        yrand = random(0,75)
        sizeRand = random(15,40)
        circle(xrand, yrand, sizeRand)
        circle(xrand-yrand, yrand, sizeRand)
        circle(xrand+xrand, yrand, sizeRand)

def drawPlanets(numPlanets):
    for i in range(numPlanets):
        x = random(50,400)     
        y = random(50,125)
        R = random(0,256)        
        G = random(0,256)
        B = random(0,256)
        stroke(R,G,B)
        randWeight = random(7,25) 
        strokeWeight(randWeight)
        point(x,y)    
    strokeWeight(1)
    noStroke()  
             
def makeRain(dropSize, numDrops=50):    
        fill(50,100,255)
        for iterate in range(0,numDrops):
            x = random(0,500)
            y = random(50,500)
            w = random(0,dropSize/2)
            h = random(0,dropSize*10)
            ellipse(x+10,y+10,w+10,h+10)
            ellipse(x-10,y-10,w-10,h-10)
            ellipse(x+10,y-10,w+10,h-10)
            ellipse(x-10,y+10,w-10,h+10) 
        noFill()
        
def drawSpaceboard():
    centerX = 200
    centerY = 200
    stroke(200)
    line(centerX-90, centerY+160, centerX+90, centerY+160)
    line(centerX-92, centerY+158, centerX+92, centerY+158)
    line(centerX-94, centerY+156, centerX+94, centerY+156)
    line(centerX-96, centerY+154, centerX+96, centerY+154)
    line(centerX-98, centerY+152, centerX+98, centerY+152)
    line(centerX-100, centerY+150, centerX+100, centerY+150)
    line(centerX-100, centerY+150, centerX-90, centerY+160)
    line(centerX+90, centerY+160, centerX+100, centerY+150)
    
def drawSpaceman():
    centerX = 200
    centerY = 200
    strokeWeight(1)
    fill(100,50,250)         
    circle(centerX, centerY, 100)
    fill(0)         
    ellipse(centerX-25, centerY-5, 25, 50)
    ellipse(centerX+25, centerY-5, 25, 50)
    ellipse(centerX, centerY+35, 40, 10)
    strokeWeight(3)  
    stroke(225,225,225)
    fill(100,0,100)          
    rect(centerX-10, centerY+50, 20, 80)
    line(centerX-10, centerY+125, centerX-30, centerY+150)
    line(centerX+10, centerY+125, centerX+30, centerY+150)
    strokeWeight(1)
    noStroke()
