x = 0
angle = QUARTER_PI
drawx=400
drawy=400
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

def setup():
    size(800, 800)
    background(0)
    noStroke()
    frameRate(60)
def draw():
    global x, angle,drawx, drawy
    drawSpiral(spiralRadius=15, R=250, G=150, B=0,centerx=400,centery=400)
    x+=1
    
# parameters for size, color, and location
def drawSpiral(spiralRadius=10, R=0, G=0, B=0, centerx=500, centery=500, lim=150):
    global x, angle
    translate(centerx, centery)
    rotate(angle)
        
    if (x >= spiralRadius or x==0):
            if(R==0 and G==0 and B==0):
                R=random(0,256)
                G=random(0,256)
                B=random(0,256)
            fill(R,G,B)
            x=1    
    r1 = fib(x) 
    r2 = fib(x+1)
    
    fill(R,G,B)
    arc(r2/r1, r2/r1, r2, r1, 0, PI,OPEN)
    arc(r2/r1, r2/r1, r2, r1, PI, 2*PI,OPEN)
    fill(0)
    circle(r2*r2/r1, r2*r2/r1, r2*r2/r1)
    circle(r2, r2, r2*r2/r1)
    circle(r1,r2, r2*r2/r1)
    circle(r2,r1, r2*r2/r1)
    
    angle += QUARTER_PI/2
    angle %= TWO_PI

    
