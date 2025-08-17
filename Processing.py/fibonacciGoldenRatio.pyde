def setup():
    size(400, 400)
    noStroke()
    textSize(22)
    frameRate(15)
    global timer
    global tenFrame
    timer = 1
    tenFrame = 0
    
# Golden ratio = limit as n->infinity of ( Fibonacci(n+1) / Fibonacci(n) ) = 1.618
def draw():
    global timer, tenFrame
    background(0)
    
    timer += 1
    tenFrame = timer/10+1
    
    # All text here
    fill(0, 250, 0)
    textAlign(CENTER,CENTER)
    text('Fibonacci(n+1) / Fibonacci(n): ', width / 2, height / 2 - 50)
    text(fib(tenFrame+1), width / 2-50, height / 2 - 25)
    text('/', width / 2, height / 2 - 25)
    text(fib(tenFrame), width / 2+50, height / 2 - 25)
    fill(250, 250, 0)
    text('Golden Ratio: ', width / 2, height / 2 + 25)
    text(float(fib(tenFrame + 1)) / fib(tenFrame), width / 2, height / 2 + 50)
    
    # 8 seconds elapsed 105F/15FPS = 8s
    if(timer >= 105):
        delay(2000)     # delay time to stare at the result
        exit()          # after time, kills program

# this iterates through the fibonacci sequence to fib(n)
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
    
    # the above lines remove the need for recursion, 
    # recursion in python is a bad use of memory
    # recursion puts python's overallocated memory regions onto the stack, 
    # meaning the stack will overflow prematurely and yield bad results 
            
        
