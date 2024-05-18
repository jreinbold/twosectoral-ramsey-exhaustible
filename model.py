import matplotlib.pyplot as plt
from scipy.optimize import root
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import math
import sys




class model:
    def __init__(self, K_0, alpha, beta, gamma, rho, sigma, phi, delta, 
                 F_s, L, periods, E_func, E_0, E_ss, redistribution, distribution):
        self.K_0 = K_0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.phi = phi
        self.delta = delta
        self.F_s = F_s
        self.L = L
        self.periods = periods
        self.E_func = E_func
        self.E_0 = E_0
        self.E_ss = E_ss
        self.redistribution = redistribution
        self.distribution = distribution
        

        
    def simulate(self):
        
        const0 = self.alpha/self.beta * (self.alpha*self.gamma/((1-self.alpha)*self.beta))**(1-self.alpha)
        const1 = (self.beta*(1-self.alpha))/(self.alpha*self.gamma)

        def calculate_p(K_f,L_f,E):
            return const0 * L_f**(1-self.alpha-self.gamma) * K_f**(self.alpha-self.beta) * E**(self.beta+self.gamma-1)
        
        # =============================================================================
        # Calculate steady state values
        # =============================================================================
        
        initial_guess_k_ss= 10
        
        def ss_values(vars):
            K_f_ss, L_f_ss, K_ss = vars    
            
            eq1 = self.L * ((K_ss-K_f_ss)/K_f_ss * const1 + 1)**(-1) - L_f_ss
            eq2 = (1 - (1-self.phi)/self.phi * self.beta/self.alpha * (K_ss-K_f_ss)/K_f_ss) * K_f_ss**self.beta * self.E_ss**(1-self.beta-self.gamma) * L_f_ss**self.gamma - self.F_s
            eq3 = (self.alpha/(self.rho+self.delta))**(1/(1-self.alpha)) - (K_ss-K_f_ss)/(self.L-L_f_ss)
            return [eq1,eq2,eq3]

        solution_ss = root(ss_values, [initial_guess_k_ss/2,0.5,initial_guess_k_ss])
        if solution_ss.success:
            print("")
        else:
            print(solution_ss)
            sys.exit()

        K_f_ss = solution_ss.x[0]
        L_f_ss = solution_ss.x[1]
        K_ss= solution_ss.x[2]
        K_c_ss = K_ss-K_f_ss
        L_c_ss = self.L-L_f_ss
        p_ss = calculate_p(K_f_ss,L_f_ss,self.E_ss)
        r_ss = self.alpha * K_c_ss**(self.alpha-1) * L_c_ss**(1-self.alpha) - self.delta
        w_ss = (1-self.alpha)*K_c_ss**(self.alpha) * L_c_ss**(-self.alpha)
        
        
        if self.redistribution:
            # C_ss for inpriced p_e
            C_ss = self.phi * (K_c_ss**self.alpha * L_c_ss**(1-self.alpha) + p_ss * K_f_ss**self.beta * L_f_ss**self.gamma * self.E_ss**(1-self.beta-self.gamma) - self.delta * K_ss - p_ss * self.F_s)
        else:
            # C_ss for outpriced p_e
            C_ss = self.phi * (self.alpha * K_c_ss**(self.alpha-1) * L_c_ss**(1-self.alpha) * K_ss + (1-self.alpha) * K_c_ss**(self.alpha) * L_c_ss**(-self.alpha) * self.L - self.delta * K_ss - p_ss * self.F_s)


        print(f"K_ss: {K_ss}")   
        print(f"K_f_ss: {K_f_ss}")
        print(f"K_c_ss: {K_c_ss}")
        print(f"L_f_ss: {L_f_ss}")
        print(f"L_c_ss: {L_c_ss}")
        print(f"p_ss: {p_ss}")
        print(f"C_ss: {C_ss}")
        print(f"r_ss: {r_ss}")
        print(f"w_ss: {w_ss}")



        # =============================================================================
        # Calculate initial values
        # =============================================================================

        if self.K_0 == 0:
            K_0 = K_ss
        else:
            K_0 = self.K_0

        def initial_values(vars):
            K_f_0, L_f_0 = vars
            
            eq1 = self.L * ((K_0-K_f_0)/K_f_0 * const1 + 1)**(-1) - L_f_0
            eq2 = (1 - (1-self.phi)/self.phi * self.beta/self.alpha * (K_0-K_f_0)/K_f_0) * K_f_0**self.beta * self.E_0**(1-self.beta-self.gamma) * L_f_0**self.gamma - self.F_s
            return [eq1,eq2]

        solution_initial_values = root(initial_values, [K_ss/2,0.5])

        if solution_initial_values.success:
            print("")
        else:
            print(solution_initial_values)
            sys.exit()

        L_f_0 = solution_initial_values.x[1]
        L_c_0 = self.L-L_f_0
        K_f_0 = solution_initial_values.x[0]
        K_c_0 = K_0-K_f_0
        p_0 = calculate_p(K_f_0,L_f_0,self.E_0)

        print(f"K_0: {K_0}")   
        print(f"K_f_0: {K_f_0}")
        print(f"K_c_0: {K_c_0}")
        print(f"L_f_0: {L_f_0}")
        print(f"L_c_0: {L_c_0}")
        print(f"p_0: {p_0}")
        
        


        # =============================================================================
        # Calculate transition
        # =============================================================================

        n = self.periods + 1

        def equationSystem(vars):
                
            C = vars[:n]
            p = vars[n:2*n]
            K = vars[2*n:3*n]
            K_c = vars[3*n:4*n]
            K_f = vars[4*n:5*n]
            L_c = vars[5*n:6*n]
            L_f = vars[6*n:]
            
                
            equations = [K[0]-K_0,      # initial condition
                          p[0]-p_0,     # initial condition
                          L_f[0]-L_f_0, # initial condition
                          L_c[0]-L_c_0, # initial condition
                          K_f[0]-K_f_0, # initial condition
                          K_c[0]-K_c_0] # initial condition
            
            for i in range(self.periods):
            
                equations.append( C[i] - C[i+1] + (C[i] / self.sigma) * ( self.alpha * K_c[i]**(self.alpha-1) * L_c[i]**(1-self.alpha) - self.delta - self.rho + (self.sigma-1)*(1-self.phi) * (p[i+1]-p[i])/p[i]) )
                
                if self.redistribution:
                    # K_dot for inpriced p_e
                    equations.append( K[i]-K[i+1]+ K_c[i]**self.alpha * L_c[i]**(1-self.alpha) + p[i]*K_f[i]**self.beta * L_f[i]**self.gamma * self.E_func(i)**(1-self.beta-self.gamma) - self.delta * K[i] - self.phi**(-1) * C[i] - p[i] * self.F_s )
                else:
                    # K_dot for outpriced p_e
                    equations.append( K[i]-K[i+1]+ self.alpha * K_c[i]**(self.alpha-1) * L_c[i]**(1-self.alpha) * K[i] + (1-self.alpha) * K_c[i]**(self.alpha) * L_c[i]**(-self.alpha) * self.L - self.delta * K[i] - self.phi**(-1) * C[i] - p[i] * self.F_s )
                
                equations.append( const0 * L_f[i+1]**(1-self.alpha-self.gamma) * K_f[i+1]**(self.alpha-self.beta) * self.E_func(i+1)**(self.beta+self.gamma-1) - p[i+1] )
                
                equations.append( (1 - (1-self.phi)/self.phi * self.beta/self.alpha * (K[i+1]-K_f[i+1])/K_f[i+1]) * K_f[i+1]**self.beta * self.E_func(i+1)**(1-self.beta-self.gamma) * L_f[i+1]**self.gamma - self.F_s )
                        
                equations.append( self.L * ((K[i+1]-K_f[i+1])/K_f[i+1] * const1 + 1)**(-1) - L_f[i+1] )
                        
                equations.append( K[i+1] -K_f[i+1] - K_c[i+1] )
                        
                equations.append( self.L -L_f[i+1] - L_c[i+1] )
                
                     
            equations.append(C[self.periods] - C_ss) # terminal condition

            
            return equations


        initial_guesses = n * [C_ss] + n * [p_ss] + n * [K_ss] + n * [K_c_ss] + n * [K_f_ss] + n * [L_c_ss] + n * [L_f_ss] 
        solution = root(equationSystem, initial_guesses)
        
        if solution.success:
            print("")
        else:
            print(solution)
            sys.exit()

        C = solution.x[:n]
        p = solution.x[n:2*n]
        K = solution.x[2*n:3*n]
        K_c = solution.x[3*n:4*n]
        K_f = solution.x[4*n:5*n]
        L_c = solution.x[5*n:6*n]
        L_f = solution.x[6*n:]
        
        r = [self.alpha * k_c**(self.alpha-1) * l_c**(1-self.alpha) - self.delta for k_c,l_c in zip(K_c,L_c)]
        w = [(1-self.alpha) * k_c**(self.alpha) * l_c**(-self.alpha)  for k_c,l_c in zip(K_c,L_c)]    
        F = [c * (1-self.phi)/(self.phi*p_t) + self.F_s for c,p_t in zip(C,p)]
        F = np.array(F)
        E = [self.E_func(t) for t in range(self.periods+1)]
        Y = [k_c**self.alpha * l_c**(1-self.alpha) + p_t*k_f**self.beta * l_f**self.gamma * e**(1-self.beta-self.gamma) for k_c,l_c,p_t,k_f,l_f,e in zip(K_c,L_c,p,K_f,L_f,E)]        

        # =============================================================================
        # Calculating mu
        # =============================================================================
        
        r_list = np.array(r)
        p_list = np.array(p)
        t_list = np.arange(len(r_list))
        integrating_t = 2000

        func_r_cont = interp1d(t_list, r_list, axis=0,
                bounds_error=False,
                kind='cubic',
                fill_value=(r_ss))
           
        func_p_cont = interp1d(t_list, p_list, axis=0,  
                bounds_error=False,
                kind='cubic',
                fill_value=(p_ss))


        def r_bar_at_0 (tau):
            integral,error = quad(func_r_cont, 0, tau)
            return integral
        
        r_discount = [math.e**(-r_bar_at_0(t)) for t in range(integrating_t+1)]
        r_discount_func = interp1d(range(integrating_t+1), r_discount, axis=0,bounds_error=False,kind='cubic')
        
        def mu_inverse_func_at_0 (tau):
                return  (func_p_cont(tau)/func_p_cont(0))**((1-self.phi)*((self.sigma-1)/self.sigma)) * (r_discount_func(tau) *  math.e**(tau*(self.rho/(1-self.sigma))))**((self.sigma-1)/self.sigma)

        def mu_at_0 (tau):
            integral,error = quad(mu_inverse_func_at_0, 0, tau)
            return 1/integral
        
        mu = mu_at_0(integrating_t)
        print(f"mu: {mu}")

        # =============================================================================
        # Calculating eta
        # =============================================================================
        
        def eta_inverse_func_at_0 (tau):
                return  func_p_cont(tau)**((1-self.phi)*((self.sigma-1)/self.sigma)) * (r_discount_func(tau) *  math.e**(tau*(self.rho/(1-self.sigma))))**((self.sigma-1)/self.sigma)
        
        def eta_at_0 (tau):
            integral,error = quad(eta_inverse_func_at_0, 0, tau)
            return 1/integral
        
        eta = eta_at_0(integrating_t)
        print(f"eta at t=0: {eta}")
        
        
        timeseries = {  
                "Y": Y,
                'C': C,
                'F': F,
                "E":E,
                'p': p,
                'K': K,
                'K_c': K_c,
                'K_f': K_f,
                'L_c': L_c,
                'L_f': L_f,
                'r': r,
                'w': w,
            }
        
        # =============================================================================
        # Check if mu is correct
        # =============================================================================
        
        
        if self.redistribution:    
            # w_disposable for inpriced p_e
            w_disposable = [w_i-p_i*self.F_s + (1-self.beta-self.gamma) * p_i*k_f**self.beta * l_f**self.gamma * e**(1-self.beta-self.gamma) for w_i,p_i,k_f,l_f,e in zip(w,p,K_f,L_f,E)] 
        else:
            # w_disposable for outpriced p_e
            w_disposable = [w_i-p_i*self.F_s  for w_i,p_i in zip(w,p)]
                
        w_disposable_list = np.array(w_disposable)

        w_disposablelist_func = interp1d(t_list, w_disposable_list, axis=0,
                bounds_error=False,
                kind='cubic',
                fill_value=(w_ss-p_ss*self.F_s))
                
        def func_w_bar_at_0 (tau):
            return w_disposablelist_func(tau) * r_discount_func(tau)


        w_avaiable_bar,error = quad(func_w_bar_at_0, 0, integrating_t)
        print(f"w_avaiable_bar: {w_avaiable_bar}") 


        total_wealth = w_avaiable_bar + K_0
        C_0_calculated = total_wealth * self.phi * mu
        print(f"total_wealth: {total_wealth}")

        print(f"C_0_mü: {C_0_calculated}")
        print(f"C_0_opt: {C[0]}")
        print(f"F_0_opt: {F[0]}")
        print(f"F_ss_opt: {F[100]}")
        
        # =============================================================================
        # Distributional total wealth levels
        # =============================================================================
        
        self.distribution["total_wealth"] = None
        
        #m = self.distribution.shape[0]
       
        for index, row in self.distribution.iterrows():
            if self.redistribution:
                # w_disposable for inpriced p_e
                w_disposable_i = [w_i* row['productivity']-p_i*self.F_s  + (1-self.beta-self.gamma) * p_i*k_f**self.beta * l_f**self.gamma * e**(1-self.beta-self.gamma) for w_i,p_i,k_f,l_f,e in zip(w,p,K_f,L_f,E)]
            else:
                # w_disposable for outpriced p_e
                w_disposable_i = [w_i* row['productivity']-p_i*self.F_s  for w_i,p_i in zip(w,p)] 
            
            w_disposable_list_i = np.array(w_disposable_i)
            
            w_disposablelist_func_i = interp1d(t_list, w_disposable_list_i, axis=0,
                    bounds_error=False,
                    kind='cubic',
                    fill_value=(w_ss* row['productivity']-p_ss*self.F_s))
            
            def func_w_bar_at_0_i (tau):
                return w_disposablelist_func_i(tau) * r_discount_func(tau)

            w_disposablelist_bar_i,error = quad(func_w_bar_at_0_i, 0, integrating_t)
            
            total_wealth_i = w_disposablelist_bar_i + K_0 * row['initial_wealth']
            self.distribution.at[index, 'total_wealth'] = total_wealth_i
            
            if total_wealth_i < 0:
                print("not enough wealth")
                     
        return timeseries,eta,self.distribution


# identical parameters between economies
sigma = 2
E_0 = 1
alpha=0.36
beta = 0.36*0.9
gamma= (1-alpha)*0.9
rho=0.0195
phi=0.65
delta=0.05
L=1



distribution = pd.read_csv("")

# exmaple distribution to solve the model quickly
# distribution = pd.DataFrame({'productivity': [1.25, 0.75, 0.75, 1, 1, 1, 1.25, 1.25, 0.75,1],
#                               'initial_wealth': [1, 1, 1, 1, 1, 1, 1, 1, 1,1]})

# scenario based inputs
periods = 100
redistribution = False
F_s=0.2 #0.2 as basic, 0.3 as scenario

reduction_factor = 0.01 # 0.01 as basic, 0.001 as scenario
reduction_speed = 0.5 # 0.5 as basic, 0.3 as scenario
E_ss_A = E_0*reduction_factor

def E_funcA(t):
    return E_ss_A + (1-reduction_factor) * E_0 * math.e**(-reduction_speed*t)

def E_funcB(t):
    return E_0
    

model_B = model(K_0 = 0, alpha=alpha, beta = beta, gamma= gamma ,rho=rho, sigma=sigma, phi=phi, 
                      delta=delta, F_s=F_s, L=L, periods=periods,
                      E_func= E_funcB, E_0 = E_0, E_ss= E_0,
                      redistribution=redistribution, distribution=distribution)

timeseries_B, eta_B, distribution_B = model_B.simulate()
series_B = pd.Series(distribution_B["total_wealth"]) # auxiliary due to df handover

print("--------- First model simulated ---------")

model_A = model(K_0 = timeseries_B['K'][0], alpha=alpha, beta = beta, gamma= gamma ,rho=rho, sigma=sigma, phi=phi, 
                      delta=delta, F_s=F_s, L=L, periods=periods,
                      E_func= E_funcA, E_0 = E_0, E_ss= E_ss_A,
                      redistribution=redistribution, distribution=distribution)

timeseries_A, eta_A, distribution_A = model_A.simulate()
series_A = pd.Series(distribution_A["total_wealth"]) # auxiliary due to df handover

print("--------- Second model simulated ---------")


# =============================================================================
# Calculate psi
# =============================================================================

del distribution["total_wealth"] # auxiliary due to df handover
distribution["total_wealth_A"] = series_A
distribution["total_wealth_B"] = series_B

def CE(W_i_A,W_i_B):
    return (W_i_B/W_i_A) * (eta_B/eta_A)**(sigma/(sigma-1)) -1

distribution["psi"] = None
for index, row in distribution.iterrows():
    distribution.at[index, 'psi'] = CE(row['total_wealth_A'],row['total_wealth_B'])
    
print("--------- Individual psi calculated -----------")


# =============================================================================
# Average psi and gain or loss per quintile
# =============================================================================

num_rows = len(distribution)
quintiles_size = num_rows // 5

distribution = distribution.sort_values(by=['productivity'])
distribution.reset_index(drop=True, inplace=True)
distribution['productivity_quintile'] = np.repeat(np.arange(1, 6), quintiles_size)

# Calculate terciles for 'wealth'
distribution = distribution.sort_values(by=['initial_wealth'])
distribution.reset_index(drop=True, inplace=True)
distribution['initial_wealth_quintile'] = np.repeat(np.arange(1, 6), quintiles_size)

# Save distribution to csv for manual check
#distribution.to_csv('distributions/distribution_df_last_run_backup.csv', index=False)

psi_ra = CE(distribution["total_wealth_A"].mean(), distribution["total_wealth_B"].mean())

ct_psi_mean = pd.crosstab(distribution['initial_wealth_quintile'], distribution['productivity_quintile'], values=distribution['psi'], aggfunc='mean')
ct_psi_mean = ct_psi_mean.fillna(psi_ra)

psi_mean = np.array(ct_psi_mean)
psi_mean = psi_mean[:, [4,3,2, 1, 0]]  # Reorder the y columns


psi_mean = 1-psi_mean/psi_ra # relative welfare difference
psi_mean = np.round(psi_mean,3)

print(psi_ra)
print(psi_mean)

# save welfare matrice
#np.savetxt("path", psi_mean, delimiter=',')

# =============================================================================
# 3d plot of psi gain or loss
# =============================================================================

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xlabels = np.array(['5th', '4th', '3rd', '2nd', '1st']) # productivity labels
ylabels = np.array(['1st', '2nd', '3rd', '4th', '5th']) # wealth labels

xpos, ypos = np.arange(xlabels.shape[0]), np.arange(ylabels.shape[0])
xposM, yposM = np.meshgrid(xpos, ypos)

zpos = 0

dx = dy = 0.5 * np.ones_like(zpos)
dz = psi_mean.ravel()

colors = ['black' if val == 0 else 'blue' if val > 0 else 'red' for val in dz]

ax.bar3d(xposM.ravel(), yposM.ravel(), zpos, dx, dy, dz, color=colors, zsort='average')
ax.set_xticks(xpos)
ax.set_xticklabels(xlabels)
ax.set_yticks(ypos)
ax.set_yticklabels(ylabels)
ax.set_xlabel('Productivity')
ax.set_ylabel('Initial wealth')
ax.set_zlabel('Relative welfare change')
ax.set_zlim(-2, 1)


#plt.savefig('bar_chart_psi_basic.png', dpi = 500)
plt.show()

# =============================================================================
# Plotting timeseries
# =============================================================================

plot_dataframe = pd.DataFrame(timeseries_A)

# Create subplots
fig, axes = plt.subplots(nrows=len(plot_dataframe.columns) , ncols=1, figsize=(7, 10), sharex=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(plot_dataframe.columns)))

# Plot each time series on a separate subplot   
for i, col in enumerate(plot_dataframe.columns):
    
    axes[i].plot(range(len(plot_dataframe[col])), plot_dataframe[col], label=col, color=colors[i])
    axes[i].set_xlabel('t')
    axes[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[i].set_ylabel(plot_dataframe.columns[i])
    axes[i].set_xlim(left=0, right=len(plot_dataframe[col])-1)
    

# Show the plot
fig.align_ylabels()
plt.savefig('time_series.png', dpi = 200)
plt.show()

