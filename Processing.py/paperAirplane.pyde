# global iterative variable
i = 0
flag = 0

# this iterates through the fibonacci sequence to fib(n), reusing this function from the last project
def fib(n): 
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        x=0
        y=1
        for i in range(1, n):
            x, y = y, x + y   # simultaeneous execution of x=y and y=x+y
        return y
def makeSnow(flakeSize):      
        fill(255,255,255)
        for iterate in range(0,10):
            x = random(0,500)
            y = random(0,500)
            w = random(0,flakeSize)
            h = random(0,flakeSize*10)
            ellipse(x+10,y+10,w+10,h+10)
            ellipse(x-10,y-10,w-10,h-10)
            ellipse(x+10,y-10,w+10,h-10)
            ellipse(x-10,y+10,w-10,h+10)
        
def setup():
    size(500,500)
    strokeWeight(10)
    noStroke()
    frameRate(30)
    
def draw():
    global i, flag
    background(100,100,255) # refresh background
    fill(255,255,0)         # golden sun
    circle(20,20,150)
    makeSnow(flakeSize=2)   # snow
    if(flag==1):
        fill(0)
        noStroke()
        beginShape()
        vertex(width/2-fib(i), fib(i+1))
        vertex(width/2, fib(i))           # Draws Airplane
        vertex(width/2+fib(i), fib(i+1))
        endShape()
        i-=1
        if(i<1):
            flag = 0
            
    if(flag==0):
        i+=1
        if(i>5):
            fill(0)
            textSize(40)
            text("WARNING,\nAIRPLANE!", width/2-100, height/2-50)
        if(i>50):
            flag = 1
    
    
