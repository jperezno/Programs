import math,cmath
import matplotlib.pyplot as plt

# lets define some equations then
# here copy paste the algebra

def avg_increment(F, epsilon_g,epsilon_r,p_z):
    eq13=F*(F*(-8*epsilon_g*epsilon_r*p_z - 4*epsilon_g*p_z + 6*epsilon_r - 1) + 16*epsilon_g*epsilon_r*p_z + 
            8*epsilon_g*p_z - 12*epsilon_g - 12*epsilon_r + 2)/6
    return eq13
def avg_acceptance(F, epsilon_g,epsilon_r, p_z):
    arg1=2*F - 8*epsilon_g*p_z*(F - 1)/3 - 28*epsilon_g*(F - 1)/3 - 4*epsilon_g + 4*epsilon_r*(-7*F + 4*epsilon_g*p_z*(F - 1) + 28*epsilon_g*(F - 1) + 12*epsilon_g + 4)/3 - 1
    log1=abs(cmath.log(arg1))
    arg2=2*F*(8*epsilon_g*epsilon_r*p_z + 56*epsilon_g*epsilon_r - 4*epsilon_g*p_z - 14*epsilon_g - 14*epsilon_r + 3) - 16*epsilon_g*epsilon_r*p_z - 64*epsilon_g*epsilon_r + 8*epsilon_g*p_z + 16*epsilon_g + 16*epsilon_r - 3
    log2=abs(cmath.log(arg2))
    num=2*F*(8*epsilon_g*epsilon_r*p_z + 56*epsilon_g*epsilon_r - 4*epsilon_g*p_z - 14*epsilon_g - 14*epsilon_r + 3)*log1 - 2*F*(8*epsilon_g*epsilon_r*p_z + 56*epsilon_g*epsilon_r - 4*epsilon_g*p_z - 14*epsilon_g - 14*epsilon_r + 3) - (16*epsilon_g*epsilon_r*p_z + 64*epsilon_g*epsilon_r - 8*epsilon_g*p_z - 16*epsilon_g - 16*epsilon_r + 3)*log2    
    den=16*epsilon_g*epsilon_r*p_z + 112*epsilon_g*epsilon_r - 8*epsilon_g*p_z - 28*epsilon_g - 28*epsilon_r + 6    
    return num/den


def main():
    # we first need some empty lists to store values
    target_fidelitites=[] #this is the x axis
    lambdas=[]            #this is the y axis

    # constants that we won't touch
    max_fid = 1
    epsilon_g = 0
    epsilon_r = 0
    p_z = 0.5

    F_t = 0.8
    sourceFid = 2*F_t -1
    Ps=1
    #  this rounds our values for iterations
    i = round(F_t*100)
    iter = round(max_fid*100)
    print("Hey I'm alive")
    while i <= iter:
        deltaF = (F_t-sourceFid)

        incr_target=avg_increment(F_t,epsilon_g,epsilon_r,p_z)
        incr_source=avg_increment(sourceFid,epsilon_g,epsilon_r,p_z)
        g = (incr_target-incr_source)/deltaF
        m = (deltaF)/g
        print('m=', m,'g=',g)

        acc_target=avg_acceptance(F_t,epsilon_g,epsilon_r,p_z)
        acc_source=avg_acceptance(sourceFid,epsilon_g,epsilon_r,p_z)
        P_f = math.exp((acc_target-acc_source)/deltaF)
        
        Bsign = (2/(Ps*P_f))**m
        lambdasign = math.log2(Bsign)+1
        #save values
        lambdas.append(lambdasign)
        target_fidelitites.append(F_t)
        i = i+1
        F_t = F_t + 0.01 #prepares next round


    if not target_fidelitites:
        print("No data to plot. Check for invalid math operations.")
        return
    #lets plot
    plt.figure(figsize=(8,5))
    plt.plot(target_fidelitites,lambdas,label='1st try')
    plt.xlabel('F_t')
    plt.ylabel('lambda')
    plt.title('Resource estimation')
    plt.legend()
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    main()