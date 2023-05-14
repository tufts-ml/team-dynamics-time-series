"""
An `event` takes an ordinary sampled group time series of shape (T,J,:) and interprets it as (T_grand,J,:),
where T_grand is the sum of the number of timesteps across i.i.d "events".  An event might induce a large
time gap between timesteps, and a discontinuity in the continuous states x.

If there are E events, then along with the observations, we store 
    end_times=[-1, t_1, …, t_E], where t_e is the timestep at which the e-th event ended.  
So to get the timesteps for the e-th event, you can index from 1,…,T_grand by doing
        [end_times[e-1]+1 : end_times[e]].
"""
