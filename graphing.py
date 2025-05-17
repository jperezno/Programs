import math,cmath
import matplotlib.pyplot as plt

# lets define some equations then
# here copy paste the algebra

def avg_increment(F, epsilon_g,epsilon_r,p_z):
    # this is to second order
    eq13=F*(F*(-8*epsilon_g*epsilon_r*p_z - 4*epsilon_g*p_z + 6*epsilon_r - 1) + 16*epsilon_g*epsilon_r*p_z + 8*epsilon_g*p_z - 12*epsilon_g - 12*epsilon_r + 2)/6

    return eq13
def avg_acceptance(F, epsilon_g,epsilon_r,p_z):
    arg1=8*F/3 + 8*epsilon_g*p_z*(F - 1) + 4*epsilon_g*p_z - 32*epsilon_g*(F - 1)/3 - 4*epsilon_g - 4*epsilon_r*(10*F + 20*epsilon_g*p_z*(F - 1) + 6*epsilon_g*p_z - 40*epsilon_g*(F - 1) - 12*epsilon_g - 7)/3 - 5/3
    log1=cmath.log(arg1)
    arg2=8*F*(10*epsilon_g*epsilon_r*p_z - 20*epsilon_g*epsilon_r - 3*epsilon_g*p_z + 4*epsilon_g + 5*epsilon_r - 1) - 56*epsilon_g*epsilon_r*p_z + 112*epsilon_g*epsilon_r + 12*epsilon_g*p_z - 20*epsilon_g - 28*epsilon_r + 5
    log2=cmath.log(arg2)
    num=8*F*(10*epsilon_g*epsilon_r*p_z - 20*epsilon_g*epsilon_r - 3*epsilon_g*p_z + 4*epsilon_g + 5*epsilon_r - 1)*log1 - 8*F*(10*epsilon_g*epsilon_r*p_z - 20*epsilon_g*epsilon_r - 3*epsilon_g*p_z + 4*epsilon_g + 5*epsilon_r - 1) - (56*epsilon_g*epsilon_r*p_z - 112*epsilon_g*epsilon_r - 12*epsilon_g*p_z + 20*epsilon_g + 28*epsilon_r - 5)*log2
    den=80*epsilon_g*epsilon_r*p_z - 160*epsilon_g*epsilon_r - 24*epsilon_g*p_z + 32*epsilon_g + 40*epsilon_r - 8
    return num/den


def thirdorderincrement(F,epsilon_g,epsilon_r,p_x,p_y,p_z):
    thirdoreq13=F*(4*F**2*(4*epsilon_g*epsilon_r*p_x + 4*epsilon_g*epsilon_r*p_y + 16*epsilon_g*epsilon_r*p_z + 22*epsilon_g*epsilon_r + 2*epsilon_g*p_x + 2*epsilon_g*p_y + 4*epsilon_g*p_z - epsilon_r - 2) 
                   + 3*F*(-16*epsilon_g*epsilon_r*p_x - 16*epsilon_g*epsilon_r*p_y - 88*epsilon_g*epsilon_r*p_z - 88*epsilon_g*epsilon_r - 8*epsilon_g*p_x - 8*epsilon_g*p_y - 28*epsilon_g*p_z + 22*epsilon_r + 5) 
                   + 48*epsilon_g*epsilon_r*p_x + 48*epsilon_g*epsilon_r*p_y + 336*epsilon_g*epsilon_r*p_z + 264*epsilon_g*epsilon_r + 24*epsilon_g*p_x + 24*epsilon_g*p_y + 120*epsilon_g*p_z - 108*epsilon_g - 120*epsilon_r - 6)/54
    return thirdoreq13
def thirdorder_acceptance(F,epsilon_g,epsilon_r,p_x,p_y,p_z):
    arg1=2*F + 8*epsilon_g*p_x*(F - 1)**2/9 + 8*epsilon_g*p_y*(F - 1)**2/9 - 32*epsilon_g*p_z*(F - 1)**2/9 - 8*epsilon_g*p_z*(F - 1)/3 - 40*epsilon_g*(F - 1)**2/9 - 28*epsilon_g*(F - 1)/3 - 4*epsilon_g + 4*epsilon_r*(-21*F - 4*epsilon_g*p_x*(F - 1)**2 - 4*epsilon_g*p_y*(F - 1)**2 + 12*epsilon_g*p_z*(F - 1)**2 + 12*epsilon_g*p_z*(F - 1) + 30*epsilon_g*(F - 1)**2 + 84*epsilon_g*(F - 1) + 36*epsilon_g - 10*(F - 1)**2 + 12)/9 + (F - 1)**2 - 1
    log1=abs(cmath.log(arg1))
    arg2=(F*(16*epsilon_g*epsilon_r*p_x + 16*epsilon_g*epsilon_r*p_y - 48*epsilon_g*epsilon_r*p_z - 120*epsilon_g*epsilon_r - 8*epsilon_g*p_x - 8*epsilon_g*p_y + 32*epsilon_g*p_z + 40*epsilon_g + 40*epsilon_r - 9) - 16*epsilon_g*epsilon_r*p_x - 16*epsilon_g*epsilon_r*p_y + 24*epsilon_g*epsilon_r*p_z - 48*epsilon_g*epsilon_r + 8*epsilon_g*p_x + 8*epsilon_g*p_y - 20*epsilon_g*p_z + 2*epsilon_g + 2*epsilon_r + 6*abs(cmath.sqrt(64*epsilon_g**2*epsilon_r**2*p_x + 64*epsilon_g**2*epsilon_r**2*p_y + 16*epsilon_g**2*epsilon_r**2*p_z**2 + 32*epsilon_g**2*epsilon_r**2*p_z + 304*epsilon_g**2*epsilon_r**2 - 48*epsilon_g**2*epsilon_r*p_x - 48*epsilon_g**2*epsilon_r*p_y - 16*epsilon_g**2*epsilon_r*p_z**2 + 8*epsilon_g**2*epsilon_r*p_z - 112*epsilon_g**2*epsilon_r + 8*epsilon_g**2*p_x + 8*epsilon_g**2*p_y + 4*epsilon_g**2*p_z**2 - 4*epsilon_g**2*p_z + 9*epsilon_g**2 - 16*epsilon_g*epsilon_r**2*p_x - 16*epsilon_g*epsilon_r**2*p_y - 8*epsilon_g*epsilon_r**2*p_z - 112*epsilon_g*epsilon_r**2 + 12*epsilon_g*epsilon_r*p_x + 12*epsilon_g*epsilon_r*p_y - 4*epsilon_g*epsilon_r*p_z + 36*epsilon_g*epsilon_r - 2*epsilon_g*p_x - 2*epsilon_g*p_y + 2*epsilon_g*p_z - 2*epsilon_g + 9*epsilon_r**2 - 2*epsilon_r)))/(16*epsilon_g*epsilon_r*p_x + 16*epsilon_g*epsilon_r*p_y - 48*epsilon_g*epsilon_r*p_z - 120*epsilon_g*epsilon_r - 8*epsilon_g*p_x - 8*epsilon_g*p_y + 32*epsilon_g*p_z + 40*epsilon_g + 40*epsilon_r - 9)
    log2=abs(cmath.log(arg2))
    arg3=((F*(16*epsilon_g*epsilon_r*p_x + 16*epsilon_g*epsilon_r*p_y - 48*epsilon_g*epsilon_r*p_z - 120*epsilon_g*epsilon_r - 8*epsilon_g*p_x - 8*epsilon_g*p_y + 32*epsilon_g*p_z + 40*epsilon_g + 40*epsilon_r - 9) - 16*epsilon_g*epsilon_r*p_x - 16*epsilon_g*epsilon_r*p_y + 24*epsilon_g*epsilon_r*p_z - 48*epsilon_g*epsilon_r + 8*epsilon_g*p_x + 8*epsilon_g*p_y - 20*epsilon_g*p_z + 2*epsilon_g + 2*epsilon_r - 6*abs(cmath.sqrt(64*epsilon_g**2*epsilon_r**2*p_x + 64*epsilon_g**2*epsilon_r**2*p_y + 16*epsilon_g**2*epsilon_r**2*p_z**2 + 32*epsilon_g**2*epsilon_r**2*p_z + 304*epsilon_g**2*epsilon_r**2 - 48*epsilon_g**2*epsilon_r*p_x - 48*epsilon_g**2*epsilon_r*p_y - 16*epsilon_g**2*epsilon_r*p_z**2 + 8*epsilon_g**2*epsilon_r*p_z - 112*epsilon_g**2*epsilon_r + 8*epsilon_g**2*p_x + 8*epsilon_g**2*p_y + 4*epsilon_g**2*p_z**2 - 4*epsilon_g**2*p_z + 9*epsilon_g**2 - 16*epsilon_g*epsilon_r**2*p_x - 16*epsilon_g*epsilon_r**2*p_y - 8*epsilon_g*epsilon_r**2*p_z - 112*epsilon_g*epsilon_r**2 + 12*epsilon_g*epsilon_r*p_x + 12*epsilon_g*epsilon_r*p_y - 4*epsilon_g*epsilon_r*p_z + 36*epsilon_g*epsilon_r - 2*epsilon_g*p_x - 2*epsilon_g*p_y + 2*epsilon_g*p_z - 2*epsilon_g + 9*epsilon_r**2 - 2*epsilon_r)))/(16*epsilon_g*epsilon_r*p_x + 16*epsilon_g*epsilon_r*p_y - 48*epsilon_g*epsilon_r*p_z - 120*epsilon_g*epsilon_r - 8*epsilon_g*p_x - 8*epsilon_g*p_y + 32*epsilon_g*p_z + 40*epsilon_g + 40*epsilon_r - 9))
    log3=abs(cmath.log(arg3))
    num15=F*(log1 - 2)*(16*epsilon_g*epsilon_r*p_x + 16*epsilon_g*epsilon_r*p_y - 48*epsilon_g*epsilon_r*p_z - 120*epsilon_g*epsilon_r - 8*epsilon_g*p_x - 8*epsilon_g*p_y + 32*epsilon_g*p_z + 40*epsilon_g + 40*epsilon_r - 9) + 2*(-8*epsilon_g*epsilon_r*p_x - 8*epsilon_g*epsilon_r*p_y + 12*epsilon_g*epsilon_r*p_z - 24*epsilon_g*epsilon_r + 4*epsilon_g*p_x + 4*epsilon_g*p_y - 10*epsilon_g*p_z + epsilon_g + epsilon_r + 3*abs(cmath.sqrt(64*epsilon_g**2*epsilon_r**2*p_x + 64*epsilon_g**2*epsilon_r**2*p_y + 16*epsilon_g**2*epsilon_r**2*p_z**2 + 32*epsilon_g**2*epsilon_r**2*p_z + 304*epsilon_g**2*epsilon_r**2 - 48*epsilon_g**2*epsilon_r*p_x - 48*epsilon_g**2*epsilon_r*p_y - 16*epsilon_g**2*epsilon_r*p_z**2 + 8*epsilon_g**2*epsilon_r*p_z - 112*epsilon_g**2*epsilon_r + 8*epsilon_g**2*p_x + 8*epsilon_g**2*p_y + 4*epsilon_g**2*p_z**2 - 4*epsilon_g**2*p_z + 9*epsilon_g**2 - 16*epsilon_g*epsilon_r**2*p_x - 16*epsilon_g*epsilon_r**2*p_y - 8*epsilon_g*epsilon_r**2*p_z - 112*epsilon_g*epsilon_r**2 + 12*epsilon_g*epsilon_r*p_x + 12*epsilon_g*epsilon_r*p_y - 4*epsilon_g*epsilon_r*p_z + 36*epsilon_g*epsilon_r - 2*epsilon_g*p_x - 2*epsilon_g*p_y + 2*epsilon_g*p_z - 2*epsilon_g + 9*epsilon_r**2 - 2*epsilon_r)))*log2 - 2*(8*epsilon_g*epsilon_r*p_x + 8*epsilon_g*epsilon_r*p_y - 12*epsilon_g*epsilon_r*p_z + 24*epsilon_g*epsilon_r - 4*epsilon_g*p_x - 4*epsilon_g*p_y + 10*epsilon_g*p_z - epsilon_g - epsilon_r + 3*abs(cmath.sqrt(64*epsilon_g**2*epsilon_r**2*p_x + 64*epsilon_g**2*epsilon_r**2*p_y + 16*epsilon_g**2*epsilon_r**2*p_z**2 + 32*epsilon_g**2*epsilon_r**2*p_z + 304*epsilon_g**2*epsilon_r**2 - 48*epsilon_g**2*epsilon_r*p_x - 48*epsilon_g**2*epsilon_r*p_y - 16*epsilon_g**2*epsilon_r*p_z**2 + 8*epsilon_g**2*epsilon_r*p_z - 112*epsilon_g**2*epsilon_r + 8*epsilon_g**2*p_x + 8*epsilon_g**2*p_y + 4*epsilon_g**2*p_z**2 - 4*epsilon_g**2*p_z + 9*epsilon_g**2 - 16*epsilon_g*epsilon_r**2*p_x - 16*epsilon_g*epsilon_r**2*p_y - 8*epsilon_g*epsilon_r**2*p_z - 112*epsilon_g*epsilon_r**2 + 12*epsilon_g*epsilon_r*p_x + 12*epsilon_g*epsilon_r*p_y - 4*epsilon_g*epsilon_r*p_z + 36*epsilon_g*epsilon_r - 2*epsilon_g*p_x - 2*epsilon_g*p_y + 2*epsilon_g*p_z - 2*epsilon_g + 9*epsilon_r**2 - 2*epsilon_r)))*log3
    den15=16*epsilon_g*epsilon_r*p_x + 16*epsilon_g*epsilon_r*p_y - 48*epsilon_g*epsilon_r*p_z - 120*epsilon_g*epsilon_r - 8*epsilon_g*p_x - 8*epsilon_g*p_y + 32*epsilon_g*p_z + 40*epsilon_g + 40*epsilon_r - 9
    return num15/den15
def main():
    # we first need some empty lists to store values
    target_fidelitites=[] #this is the x axis
    lambdas=[]            #this is the y axis

    # constants that we won't touch
    max_fid = 1
    epsilon_g = 0
    epsilon_r = 0.01
    p_z = 0.5
    p_x= 0.25
    p_y= 0.25
    F_t = 0.7
    sourceFid = 2*F_t -1
    Ps=1
    #  this rounds our values for iterations
    i = round(F_t*100)
    iter = round(max_fid*100)
    print("Hey I'm alive")
    while i <= iter:
        deltaF = (F_t-sourceFid)
        ## 2nd order approx
        incr_target=avg_increment(F_t,epsilon_g,epsilon_r,p_z)
        incr_source=avg_increment(sourceFid,epsilon_g,epsilon_r,p_z)
        ## 3rd oreder approx
        # incr_target=thirdorderincrement(F_t,epsilon_g,epsilon_r,p_x,p_y,p_z)
        # incr_source=thirdorderincrement(sourceFid,epsilon_g,epsilon_r,p_x,p_y,p_z)
        g = (incr_target-incr_source)/deltaF
        m = (deltaF)/g
        print('m=', m,'g=',g)
        acc_target=avg_acceptance(F_t,epsilon_g,epsilon_r,p_z)
        acc_source=avg_acceptance(sourceFid,epsilon_g,epsilon_r,p_z)
    
        # acc_target=thirdorder_acceptance(F_t,epsilon_g,epsilon_r,p_x,p_y,p_z)
        # acc_source=thirdorder_acceptance(sourceFid,epsilon_g,epsilon_r,p_x,p_y,p_z)
        absdif=acc_target-acc_source
        P_f = cmath.exp((absdif/deltaF))
        print("P_f=",P_f)
        Bsign = (2/(Ps*P_f))**m
        lambdasign = cmath.log(Bsign,2).real

        #save values
        lambdas.append(lambdasign)
        target_fidelitites.append(F_t)
        #Prepare next round
        F_t = F_t + 0.01 
        sourceFid=2*F_t-1
        i=i+1



    if not target_fidelitites:
        print("No data to plot. Check for invalid math operations.")
        return
    #lets plot
    plt.figure(figsize=(8,5))
    plt.plot(target_fidelitites,lambdas,label='smo')
    plt.xlabel('F_t')
    plt.ylabel('lambda')
    plt.title('Resource estimation')
    plt.legend()
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    main()