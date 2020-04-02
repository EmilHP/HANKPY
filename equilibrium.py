import numpy as np
import time
import solve

from consav.misc import elapsed 

def find_ra_mon_comp(model,Pi0=None,KN0=None,do_print=True,step_size=0.1,tol=1e-8):

    # unpack    
    par = model.par

    # initial
    m = (par.vareps-1)/par.vareps     
    Pi = Pi0
    KN = KN0
    it = 1
    
    # relaxation loop
    t0_outer = time.time()
    while True:

        t0 = time.time()

        if do_print: print(f'{it:3d}: KN = {KN:.8f}, Pi = {Pi:.8f}')

        # a. factor prices   
        par.ra = par.alpha*m/(KN**(1-par.alpha)) - par.delta
        par.w = (1-par.alpha)*m*KN**par.alpha
        par.Pi = Pi

        if do_print: 
            print(f'    implied ra = {par.ra:.4f}')
            print(f'    implied w = {par.w:.4f}')

        # b. solve model
        model.create_grids()
        model.solve(do_print=False)
        model.calculate_moments()
        moms = model.moms

        # c. implied KN and Pi0
        if do_print: 
            print(f'    implied KN = {moms["KN"]:.4f} [{moms["KN_discrepancy"]:.8f}]')
            print(f'    implied Pi = {moms["Pi"]:.4f} [{moms["Pi_discrepancy"]:.8f}]')
            print(f'    time: {elapsed(t0)}')

        # d. check for convergence
        if abs(moms['KN_discrepancy']) < tol and abs(moms['Pi_discrepancy']) < tol:
            break

        # e. updates
        KN = step_size*moms["KN"] + (1-step_size)*KN
        Pi = step_size*moms["Pi"] + (1-step_size)*Pi

        it += 1    

    model.calculate_moments(do_MPC=True)

    if do_print:

        print('')
        print(f'equilibrium found in {elapsed(t0_outer)}')
        print(f' ra = {par.ra:.8f}')
        print(f' w = {par.w:.8f}')
        print(f' Pi = {par.Pi:.8f}')


def find_ra_perf_comp(model,KN0=None,do_print=True,step_size=0.1,tol=1e-8,fix_zeta=False):

    # unpack    
    par = model.par

    # initial
    par.vareps = np.inf
    Pi = 0
    par.Pi = Pi
    KN = KN0
    it = 1
    
    # relaxation loop
    t0_outer = time.time()
    while True:

        t0 = time.time()

        if do_print: print(f'{it:3d}: KN = {KN:.8f}')

        # a. factor prices, KY and zeta
        par.ra = par.alpha/(KN**(1-par.alpha)) - par.delta
        par.w = (1-par.alpha)*KN**par.alpha

        if not fix_zeta:
            KY = par.alpha/(1-par.alpha)*par.w/(par.ra+par.delta)
            par.zeta = (KY/KN)*3 # not same KY and KN as in mon comp, so have to update (do with and wthout)

        if do_print: 
            print(f'    implied ra = {par.ra:.4f}')
            print(f'    implied w = {par.w:.4f}')

        # b. solve model
        model.create_grids()
        model.solve(do_print=False)
        model.calculate_moments()
        moms = model.moms

        # c. implied KN
        if do_print: 
            print(f'    implied KN = {moms["KN"]:.4f} [{moms["KN_discrepancy"]:.8f}]')
            print(f'    time: {elapsed(t0)}')

        # d. check for convergence
        if abs(moms['KN_discrepancy']) < tol:
            break

        # e. updates
        KN = step_size*moms["KN"] + (1-step_size)*KN

        it += 1    

    model.calculate_moments(do_MPC=True)

    if do_print:

        print('')
        print(f'equilibrium found in {elapsed(t0_outer)}')
        print(f' ra = {par.ra:.8f}')
        print(f' w = {par.w:.8f}')

def vary_param_GE_perf_comp(model,parname,vals,KN0,step_size=0.1,tol=1e-8,fix_zeta=False):

    # a. allocate solution containers
    sim_data = {}
    sim_data['ra'] = np.zeros(len(vals))
    sim_data['rb'] = np.zeros(len(vals))
    sim_data['KY'] = np.zeros(len(vals))
    sim_data['BY'] = np.zeros(len(vals))
    sim_data['a_margcum'] = {}
    sim_data['b_margcum'] = {}
    sim_data['a_gini'] = np.zeros(len(vals))
    sim_data['b_gini'] = np.zeros(len(vals))
    sim_data['c'] = np.zeros(len(vals))
    sim_data['c_gini'] = np.zeros(len(vals))
    sim_data['v'] = np.zeros(len(vals))
    sim_data['MPC'] = np.zeros(len(vals))
    sim_data['v_planner'] = np.zeros(len(vals))

    # b. solve model for different values of r_b
    for i,val in enumerate(vals):
        # i. update variable value
        setattr(model.par,parname,val)
        print('Currently solving for ' + parname + f' = {getattr(model.par,parname,val):.4f}')
        # ii. solve for general equilibrium 
        find_ra_perf_comp(model,KN0,step_size=step_size,tol=tol,fix_zeta=fix_zeta,do_print=False)
        # iii. save outcomes
        moms = model.moms
        par = model.par

        sim_data['ra'][i] = par.ra
        sim_data['rb'][i] = val
        sim_data['KY'][i] = moms['KY']
        sim_data['BY'][i] = moms['BY']
        sim_data['a_margcum'][i] = moms['a_margcum']
        sim_data['b_margcum'][i] = moms['b_margcum']
        sim_data['a_gini'][i] = moms['a_gini']
        sim_data['b_gini'][i] = moms['b_gini']
        sim_data['c'][i] = moms['c']
        sim_data['c_gini'][i] = moms['c_gini'] 
        sim_data['MPC'][i] =  moms['MPC']
        sim_data['v'][i] = moms['v']
        # percentiles
        # iv. calculate social planner value function
        # o.
        solve.create_diags_HJB(model.par,model.sol,social_planner=True)
         
        # oo. construct Q
        solve.create_Q(model.par,model.sol,model.ast,'UMFPACK')
 
        # oo. solve equation system
        solve.solve_eq_sys_HJB(model.par,model.sol,model.ast,'UMFPACK',model.cppfile)

        # ooo. solve KF
        index = par.Nb_neg*par.Na + 0
        model.sol.g[:] = 0
        model.sol.g[:,index] = par.z_dist/par.dab_tilde[par.Nb_neg,0]
        solve.solve_KFE(model,solmethod='UMFPACK',do_print=False)

        # oooo. add social planner value function
        _g = model.sol.g.reshape(par.Nz,par.Na,par.Nb,order='F')
        margdist = _g*par.dab_tilde.T
        sim_data['v_planner'][i] = np.sum(margdist*model.sol.v)

    return sim_data


# def find_ra_perf_comp(model,ra0=None,eps=None,do_print=True,tol=1e-8,N_ini=0.57866907):

#     # unpack    
#     par = model.par

#     # initialize
#     # i. ensure perfect competition
#     par.vareps = np.inf
#     par.Pi = 0

#     # ii. modified bisection loop variables
#     ra_min = ra0 - eps
#     ra_max = ra0 + eps
#     err_max = 100
#     err_min = 100
#     K_error = 100
#     N = N_ini # initial N
#     it = 1
    
#     # fast bisection loop
#     t0_outer = time.time()
#     while True:

#         t0 = time.time()

#         if err_max == 100 or err_min == 100:
#             par.ra = (ra_max + ra_min)/2
#         else:
#             weight_max = abs(err_min)/(abs(err_min) + abs(err_max))
#             weight_max = np.fmin(np.fmax(weight_max,0.1), 0.6)
#             par.ra = weight_max*ra_max + (1 - weight_max)*ra_min
        
#         if par.ra < 0: # for stability while converging
#             par.ra = 0.0005
        
#         if do_print: print(f'{it:3d}: ra = {par.ra:.8f}')

#         # a. K demand and wage
#         K_demand = ((par.ra+par.delta)/(par.alpha*N**(1-par.alpha)))**(1/(par.alpha-1))
#         par.w = (1-par.alpha)*(par.alpha/(par.ra+par.delta))**(par.alpha/(1-par.alpha))
#         KY = par.alpha/(1-par.alpha)*par.w/(par.ra+par.delta)
#         KN = KY**(1/(1-par.alpha))
#         par.zeta = (KY/KN)*3

        # # b. solve model
#         model.create_grids()
#         model.solve(do_print=False)
#         model.calculate_moments()
#         moms = model.moms

#         N = moms['N_supply']

#         # c. demand-supply discrepancy for capital
#         K_error = moms['A_supply'] - K_demand

#         # d. update mean labor efficiency and N supply since implied K/N is different to HANK
#         # e. check for convergence
#         if abs(K_error) < tol:
#             if do_print: print(f'   Convergence achieved, final K error = {K_error:.8f}')
#             break

#         # f. update interest errors
#         if K_error > tol: 
#             # too high supply, decrease interest rate
#             ra_max = par.ra
#             err_max = K_error
#         else:
#             # too low supply, increase interest rate
#             ra_min = par.ra
#             err_min = K_error

#         if do_print: 
#             print(f'    implied K demand = {K_demand:.4f} [{K_error:.8f}]')
#             print(f'    implied w = {par.w:.4f}')
#             print(f'    time: {elapsed(t0)}')

#         it += 1    

#     model.calculate_moments(do_MPC=True)

#     if do_print:

#         print('')
#         print(f'equilibrium found in {elapsed(t0_outer)}')
#         print(f' ra = {par.ra:.8f}')
#         print(f' w = {par.w:.8f}')
