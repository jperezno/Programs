import math
import matplotlib.pyplot as plt

# lets define some equations then
# here copy paste the algebra

def avg_increment(F, epsilon_g,eta,p_z):
    firstterm13 = -F**2*(60*epsilon_g*eta**2 - 60*epsilon_g*eta + 4*epsilon_g*p_z + 22*epsilon_g - 16*eta**2 + 16*eta - 5)**2 + 2*F*(60*epsilon_g*eta**2 - 60*epsilon_g*eta + 4*epsilon_g*p_z + 22*epsilon_g - 16*eta**2 + 16*eta - 5)*(88*epsilon_g*eta**2 - 88*epsilon_g*eta + 6*epsilon_g*p_z + 36*epsilon_g - 15*eta**2 + 15*eta - 6) 
    secondterm13 = 3*(2*eta**2 - 2*eta + 1)*(8*epsilon_g**2*eta**2 - 8*epsilon_g**2*eta - 12*epsilon_g**2 - 8*epsilon_g*eta**2 + 8*epsilon_g*eta + 2*epsilon_g*p_z + 8*epsilon_g + eta**2 - eta - 1)
    deneq13 = 2*(60*epsilon_g*eta**2 - 60*epsilon_g*eta + 4*epsilon_g*p_z + 22*epsilon_g - 16*eta**2 + 16*eta - 5)**2
    log_arg13 = 2*F*(60*epsilon_g*eta**2 - 60*epsilon_g*eta + 4*epsilon_g*p_z + 22*epsilon_g - 16*eta**2 + 16*eta - 5) - 96*epsilon_g*eta**2 + 96*epsilon_g*eta - 8*epsilon_g*p_z - 32*epsilon_g + 26*eta**2 - 26*eta + 7
    if log_arg13 <= 0:
        return float('nan')
    logterm13=math.log(log_arg13)
    eq13=(firstterm13 + secondterm13*logterm13)/deneq13
    return eq13


def avg_acceptance(F, epsilon_g,eta, p_z):
    term1=2*F*(140*epsilon_g*eta**2 - 140*epsilon_g*eta + 4*epsilon_g*p_z + 44*epsilon_g - 24*eta**2 + 24*eta - 7)
    term2=2*F*(140*epsilon_g*eta**2 - 140*epsilon_g*eta + 4*epsilon_g*p_z + 44*epsilon_g - 24*eta**2 + 24*eta - 7) - (244*epsilon_g*eta**2 - 244*epsilon_g*eta + 8*epsilon_g*p_z + 70*epsilon_g - 42*eta**2 + 42*eta - 11)
    # takes aruments and
    firstlog_arg15=-544320*F*epsilon_g*eta**8 + 2177280*F*epsilon_g*eta**7 - 15552*F*epsilon_g*eta**6*p_z - 4253472*F*epsilon_g*eta**6 + 46656*F*epsilon_g*eta**5*p_z + 5139936*F*epsilon_g*eta**5 - 69984*F*epsilon_g*eta**4*p_z - 4171824*F*epsilon_g*eta**4 + 62208*F*epsilon_g*eta**3*p_z + 2317248*F*epsilon_g*eta**3 - 34992*F*epsilon_g*eta**2*p_z - 861192*F*epsilon_g*eta**2 + 11664*F*epsilon_g*eta*p_z + 196344*F*epsilon_g*eta - 1944*F*epsilon_g*p_z - 21384*F*epsilon_g + 93312*F*eta**8 - 373248*F*eta**7 + 727056*F*eta**6 - 874800*F*eta**5 + 705672*F*eta**4 - 388800*F*eta**3 + 142884*F*eta**2 - 32076*F*eta + 3402*F + 474336*epsilon_g*eta**8 - 1897344*epsilon_g*eta**7 + 15552*epsilon_g*eta**6*p_z + 3693600*epsilon_g*eta**6 - 46656*epsilon_g*eta**5*p_z - 4440096*epsilon_g*eta**5 + 69984*epsilon_g*eta**4*p_z + 3576960*epsilon_g*eta**4 - 62208*epsilon_g*eta**3*p_z - 1967328*epsilon_g*eta**3 + 34992*epsilon_g*eta**2*p_z + 721224*epsilon_g*eta**2 - 11664*epsilon_g*eta*p_z - 161352*epsilon_g*eta + 1944*epsilon_g*p_z + 17010*epsilon_g - 81648*eta**8 + 326592*eta**7 - 633744*eta**6 + 758160*eta**5 - 606528*eta**4 + 330480*eta**3 - 119556*eta**2 + 26244*eta - 2673
    secondlog_arg15=2*F*(140*epsilon_g*eta**2 - 140*epsilon_g*eta + 4*epsilon_g*p_z + 44*epsilon_g - 24*eta**2 + 24*eta - 7) - 244*epsilon_g*eta**2 + 244*epsilon_g*eta - 8*epsilon_g*p_z - 70*epsilon_g + 42*eta**2 - 42*eta + 11
    
    if firstlog_arg15 <= 0 or secondlog_arg15 <= 0:
        return float('nan')
    
    deneq15= 280*epsilon_g*eta**2 - 280*epsilon_g*eta + 8*epsilon_g*p_z + 88*epsilon_g - 48*eta**2 + 48*eta - 14
    eq15=(term1*math.log(firstlog_arg15) - term2*math.log(secondlog_arg15))/deneq15
    return eq15


def main():
    # we first need some empty lists to store values
    target_fidelitites=[] #this is the x axis
    lambdas=[]            #this is the y axis

    # constants that we won't touch
    max_fid = 0.99
    epsilon_g = 1e-10
    eta = 0.99
    p_z = 0.5

    F_t = 0.8
    sourceFid = 2*F_t-1
    Ps=1
    #  this rounds our values for iterations
    i = round(F_t*100)
    iter = round(max_fid*100)
    print("Hey I'm alive")
    while i <= iter:
        deltaF = (F_t-sourceFid)

        incr_target=avg_increment(F_t,epsilon_g,eta,p_z)
        incr_source=avg_increment(sourceFid,epsilon_g,eta,p_z)

        acc_target=avg_acceptance(F_t,epsilon_g,eta,p_z)
        acc_source=avg_acceptance(sourceFid,epsilon_g,eta,p_z)
        
        # check for invalid values
        if any(math.isnan(v) for v in [incr_target, incr_source, acc_target, acc_source]):
            continue

        g = (incr_target-incr_source)/deltaF
        m = (deltaF)/g
        print('m=', m)

        P_f = math.exp((acc_target-acc_source)/deltaF)
        Bsign = (2/(Ps*P_f))**m
        lambdasign = math.log2(Bsign)+1
        #save values
        lambdas.append(lambdasign)
        target_fidelitites.append(F_t)
        i = i+1
        F_t = F_t + 0.01 #prepares next round

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