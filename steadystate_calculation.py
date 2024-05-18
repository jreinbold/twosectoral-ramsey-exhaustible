from scipy.optimize import root
import sys


phi = 0.65
alpha = 0.36
beta = 0.36*0.9
gamma = (1-alpha)*0.9
rho=0.0195
delta=0.05

L = 1
F_s = 0.21/2
E = 5


for i in range(9,10):

    print(i)
    initial_guess_k_ss = i
    
    def equations(vars):
        K_f, L_f, K = vars
        const = (beta*(1-alpha))/(alpha*gamma)
        
        eq1 = L * ((K-K_f)/K_f * const + 1)**(-1) - L_f
        eq2 = (1 - (1-phi)/phi * beta/alpha * (K-K_f)/K_f) * K_f**beta * E**(1-beta-gamma) * L_f**gamma - F_s
        eq3 = (alpha/(rho+delta))**(1/(1-alpha)) - (K-K_f)/(L-L_f)
        return [eq1,eq2,eq3]
    
    solution_ss = root(equations, [initial_guess_k_ss/2,0.5,initial_guess_k_ss])
    if solution_ss.success:
        print("success")
    
        K_f = solution_ss.x[0]
        L_f = solution_ss.x[1]
        K= solution_ss.x[2]
        K_c = K-K_f
        L_c = L-L_f
        
        Q_c = K_c**alpha * L_c**(1-alpha)
        Q_f = K_f**beta * L_f**gamma * E**(1-beta-gamma)
        
        p = alpha/beta * (beta*(1-alpha)/(gamma*alpha))**(1-alpha) * L_f**(1-alpha-gamma) * K_f**(alpha-beta) * E**(beta+gamma-1)
        
        r_c = alpha * K_c**(alpha-1) * L_c**(1-alpha) - delta
        r_f= p * beta * K_f**(beta-1) * L_f**gamma * E**(1-beta-gamma)-delta
        
        C_ss = phi * (K_c**alpha * L_c**(1-alpha) + p * K_f**beta * L_f**gamma * E**(1-beta-gamma) - delta * K - p * F_s)
        F_ss = C_ss* ((1-phi)/phi) * p**(-1) + F_s

        print(f"K: {K}")   
        print(f"K_f: {K_f}")
        print(f"K_c: {K_c}")
        print(f"L_f: {L_f}")
        print(f"L_c: {L_c}")
        print(f"p: {p}")
        print(f"r_c: {r_c}")
        print(f"r_f: {r_f}")
        print(f"Q_c: {Q_c}")
        print(f"Q_f: {Q_f}")
        print(f"C_ss: {C_ss}")
        print(f"F_ss: {F_ss}")


        sys.exit()



