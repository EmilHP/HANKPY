import numpy as np
import time
import solve
import pickle

from consav.misc import elapsed 

def transition(model,parname,val,KN0=33.22430956,KNST=31.45765379,do_print=True,step_size=0.1,tol=1e-5,load=False,maxiter=100):

    par = model.par

    if not load:

        data = {} # empty dict for steady-state output

        # a. find initial equilibrium
        if do_print:
            print('Solving initial s.s.')
        model.find_ra_perf_comp(KN0=KN0,tol=1e-5)
        # ii. save output
        KN_initial = model.moms['KN']
        data['g0'] = model.sol.g

        # b. find terminal equilibrium
        if do_print:
            print('Solving terminal s.s.')
        # i. update shocked parameter
        setattr(par,parname,val)
        # ii. find new equilibrium given new parameter value
        model.find_ra_perf_comp(KN0=KNST,tol=1e-5)
        # iii. save output
        KN_terminal = model.moms['KN']
        data['v_st'] = model.sol.v

        # c. create initial KN path guess
        data['KN_path'] = np.linspace(KN_initial,KN_terminal,model.par.N_trans)

        # d. save data ERROR
        #outfile = 'data/' + parname + '_ss_data.pkl' 
        #pickle.dump(data,open(outfile,'wb'))

    else:
        print('Not implemented correctly yet')
        # a. load saved data ERROR
        #setattr(par,parname,val)
        #loadfile = 'data/' + parname + '_ss_data.pkl'
        #data = pickle.load(open(loadfile,'rb'))

    # e. update HJB + KFE time-steps for transition path
    par.DeltaHJB = par.dt_trans
    par.DeltaKFE = par.dt_trans

    # f. solve transition path
    if do_print:
        print('Solving transition path')

    for it in range(10):

        # i. calculate transition given current KN path guess
        model.solve_trans(model,data=data)
        # ii. calculate max difference
        max_diff = np.max(np.abs(model.sol.KN_path_new-data['KN_path']))
        # iii. check convergence critera and update KN path
        if max_diff < tol:
            if do_print:
                print(f'Convergence achieved in iteration = {it+1}')
            break
        else: # update transition path
            data['KN_path'] = (1-step_size)*data['KN_path'] + step_size*model.sol.KN_path_new
            if do_print:
                print(f'Current iteration = {it+1}, max difference = {max_diff:.8f}')
    
    return data['KN_path']