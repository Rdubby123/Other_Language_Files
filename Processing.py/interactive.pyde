# Used to draw the astronaut
def drawSpaceman():
    centerX = 200
    centerY = 200
    background(0,0,0)      # Black Background
    stroke(225,225,225)    # White Outline
    strokeWeight(1)
    fill(100,0,0)          # Red Body
    rectMode(CENTER)
    rect(centerX, centerY+50, 20, 150)
    fill(100,50,250)         # Blue Head
    circle(centerX, centerY, 100)
    fill(0,200,0)          # Green Eyes
    ellipse(centerX-25, centerY-5, 25, 50)
    ellipse(centerX+25, centerY-5, 25, 50)
    strokeWeight(3)        # For Legs
    line(centerX-10, centerY+125, centerX-30, centerY+150)
    line(centerX+10, centerY+125, centerX+30, centerY+150)
    
# Used to draw the Planets
def drawPlanets(numPlanets):
    for i in range(numPlanets):
        x = random(25,375)        # random coords
        y = random(25,100)
        R = random(0,256)         # random RGB
        G = random(0,256)
        B = random(0,256)
        stroke(R,G,B)
        randWeight = random(7,25) # random weight
        strokeWeight(randWeight)
        point(x,y)                # Each colored circle

def setup():
    size(400,400)
    noStroke()
    fill(0,100,100)
    println("Mouse Y = Number of Planets")
    print("Up = Less, Down = More")
    
def draw():
    background(175)
    drawSpaceman()
    drawPlanets(numPlanets=(mouseY)*15/400)
    frameRate(30)
    noStroke()
    fill(0,250)
    ellipse(width/2, height/2+20,23, (mouseY*40/400)+2)   # Spaceman's mouth          
                      
                             
   
