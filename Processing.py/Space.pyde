# Ryan DuBrueler Final Project

"""

 - This project uses many random numbers to generate the location of fibonacci spirals
   It plots two spirals at a time each at different coordinates. Over many repetitions these 
   random contributions start to make a starry space void. 

 - Run the program and watch the animation until it stops, 
   it is set to stop after 1 minute if you would rather skip
   to the resulting image. 

 - Each run will create a different result image 

"""

# Initial global declarations
x, time = 0,0
angle = QUARTER_PI
drawx, drawy, drawx2, drawy2 = 400, 400, 400, 400
sizex, sizey = 800,800
endTime = 3600 # 1 minute @ 60 fps

def setup():
    global sizex,sizey
    size(sizex,sizey)
    frameRate(60)
    background(0)
    noStroke()
    
def draw():
    global x, drawx, drawy, drawx2, drawy2
    global time, sizex, sizey, endTime
    
    if(time>=endTime):
        noLoop()
        
    if (time == 0):
        drawSpaceman()
        drawSpaceboard()
        drawPlanets(30) 
    # random location of fibonacci spirals      
    if(x%7==0):
        drawx=random(0,sizex)
        drawy=random(0,sizey)
        drawx2=random(0,sizex)
        drawy2=random(0,sizey)
    # random size    
    starSize=random(3,7)
    drawSpiral(spiralRadius=starSize, B=250, centerx=drawx, centery=drawy)
    drawSpiral(spiralRadius=starSize,B=250, centerx=drawx2, centery=drawy2)
    x += 1
    time += 1   
    
# Used to draw the spaceman
def drawSpaceman():
    centerX = 400
    centerY = 400
    stroke(20,225,225) 
    strokeWeight(1)
    
    # Blue Body
    fill(100,100,200)          
    rectMode(CENTER)
    rect(centerX, centerY+50, 20, 150)
    fill(20,225,225)
    rect(centerX, centerY+85, 10, 50)
    
    # Blue Head
    fill(100,100,200)         
    circle(centerX, centerY, 100)
    
    # Green Eyes
    fill(20,225,225)          
    ellipse(centerX-25, centerY-5, 25, 50)
    ellipse(centerX+25, centerY-5, 25, 50)
    
    # Legs
    strokeWeight(3)        
    line(centerX-10, centerY+125, centerX-30, centerY+150)
    line(centerX+10, centerY+125, centerX+30, centerY+150)
    
    # Arms
    line(centerX-11, centerY+75, centerX-30, centerY+55)
    line(centerX+11, centerY+75, centerX+30, centerY+55)

# Used to draw the spaceboard
def drawSpaceboard():
    centerX = 400
    centerY = 400
    stroke(200)
    line(centerX-90, centerY+160, centerX+90, centerY+160)
    line(centerX-92, centerY+158, centerX+92, centerY+158)
    line(centerX-94, centerY+156, centerX+94, centerY+156)
    line(centerX-96, centerY+154, centerX+96, centerY+154)
    line(centerX-98, centerY+152, centerX+98, centerY+152)
    line(centerX-100, centerY+150, centerX+100, centerY+150)
    line(centerX-100, centerY+150, centerX-90, centerY+160)
    line(centerX+90, centerY+160, centerX+100, centerY+150)

# draws the planets
def drawPlanets(numPlanets):
    for i in range(numPlanets):
        x = random(0,750)     
        y = random(30,300)
        R = random(0,256)        
        G = random(0,256)
        B = random(0,256)
        stroke(R,G,B)
        randWeight = random(10,35) 
        strokeWeight(randWeight)
        point(x,y)    
    strokeWeight(1)
    noStroke()  
    
# returns fib(n)    
def fib(n): 
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for i in range(1, n):
            a, b = b, a + b
        return b 
    
# fibonacci spiral, parameters for size, color, and location
def drawSpiral(spiralRadius=10, R=0, G=0, B=0, centerx=500, centery=500):
    global x, angle
    # center to coord and rotate
    translate(centerx, centery)
    rotate(angle)
    # each spiral will be its own color
    # and resets to 0 once it iterates through radius
    if (x >= spiralRadius or x==0):
            # Defaults are R=0 G=0 B=0
            if(R==0 or G==0 or B==0):
                if R==0:
                    R=random(100,256)
                if G==0:
                    G=random(100,256)
                if B==0:
                    B=random(100,256)
            fill(R,G,B)
            x=0 
    # arbitrary variable names
    f1 = fib(x) 
    f2 = fib(x+1)
    # x=fib(x)+fib(x+1), y=fib(x+1)-fib(x), width=fib(x), height=fib(x+1), 
    # from 0 to pi/2
    arc(f2+f1, f2-f1, f1, f2, 0, HALF_PI,OPEN)
    # increment angle and reset after full rotation
    angle += QUARTER_PI/2
    angle %= TWO_PI

    
