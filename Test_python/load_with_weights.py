# -*- coding: utf-8 -*
import os
from params import *
import numpy as np
import matplotlib.pyplot as plt
from human_robot_weights import human_robot
from robot_robot_weights import robot_robot
from pDMP_functions import pDMP
from scipy.fft import rfft, rfftfreq

plt.close('all')

compute_data = False

runs = 15
running = np.array([[True, False, False],[False, True, False], [False, False, True]])

N_, e_th_ , h_ = N, e_th, h

for idx2 in range(running.shape[0]):
    v_N, v_e, v_h  = running[idx2]
    
    N_all = [5, 25, 50, 100]
    e_th_all = [0.01, 0.02, 0.05]
    h_all = [0.5, 1, 2.5,  5]
    
    
    if [v_N==True, v_e == True, v_h == True].count(True) != 1 :
        raise NameError("Number of variables is incorrect")
    
    if v_N:
        name = 'N'
        e_th_all = [e_th_]*len(N_all)
        h_all = [h_]*len(N_all)
    elif v_e:
        name = 'e_th'
        N_all = [N_]*len(e_th_all)
        h_all = [h_]*len(e_th_all)
    elif v_h:
        name = 'h'
        N = [N_]*len(h_all)
        e_th_all = [e_th_]*len(h_all)
    else:
        name = 'std'
        N_all = [N_]
        e_th_all = [e_th_]
        h_all = [h_]
        
        
    if len(N_all) != len(e_th_all) or len(N_all) != len(h_all) or len(e_th_all) != len(h_all):
        raise NameError("Shapes is not correct!")
        
        
    fig1, ax1 = plt.subplots(2,len(N_all), figsize=(20.0, 10.0), sharex=True)
    fig2, ax2 = plt.subplots(runs,1, figsize=(20.0, 10.0), sharex=True)
    fig3, ax3 = plt.subplots(runs,1, figsize=(20.0, 10.0), sharex=True)
    fig4, ax4 = plt.subplots(len(N_all),1, figsize=(20.0, 10.0), sharex=True)

        
    
    for i in range(len(N_all)):
        
        N = N_all[i]
        e_th = e_th_all[i]
        h = h_all[i]
        
        if v_N:
            v = N
        elif v_e:
            v = e_th
        elif v_h:
            v = h
        else:
            v = 'N/A'
           
    
        if compute_data:
        
            human_robot(N= N, e_th = e_th, h = h, v_N = v_N, v_eth = v_e, v_h = v_h)
            robot_robot(runs,N= N, e_th = e_th, h = h, v_N = v_N, v_eth = v_e, v_h = v_h)

        
        path = 'save_data/'+name
        
        files = os.listdir(path)
        
        files_w_traject = []
        files_w_stiff = []
        
        if v_N or v_e or v_h:
            name_temp = name+"_"+str(v)
        else:
            name_temp = name
        
        for f in files:
            if v_N or v_e or v_h:
                if "data_"+name+"_"+str(v)+"_w_traject_robot" in f: files_w_traject.append(f)
                elif "data_"+name+"_"+str(v)+"_w_stiff_robot" in f: files_w_stiff.append(f)
            else:
                if "data_"+name+"_w_traject_robot" in f: files_w_traject.append(f)
                elif "data_"+name+"_w_stiff_robot" in f: files_w_stiff.append(f)
    
        if v_N or v_e or v_h:
            files_w_traject.sort(key= lambda x: float(x.strip('data_'+name+'_').lstrip(str(v)).strip('_w_traject_robot_').strip('.csv')))
            files_w_stiff.sort(key= lambda x: float(x.strip('data_'+name+'_').lstrip(str(v)).strip('_w_stiff_robot_').strip('.csv')))
        else:
            files_w_traject.sort(key= lambda x: float(x.strip('data_'+name+'_').strip('_w_traject_robot_').strip('.csv')))
            files_w_stiff.sort(key= lambda x: float(x.strip('data_'+name+'_').strip('_w_stiff_robot_').strip('.csv')))
            
        files_w_traject = files_w_traject[:runs]
        files_w_stiff = files_w_stiff[:runs]
        
            
        if len(files_w_traject) != len(files_w_stiff):
            raise NameError('Data was saved incorrectly')
        
        data_w_traject = []
        data_w_stiff = []
            
       
        
        t = np.array([dt*i for i in range(samples)])
        phase = [2*np.pi*(dt/tau)*i for i in range(samples*2)] 
        
        ax1[0,i].set_title(name +str(v))
        
        for j in range(len(files_w_traject)):
            
            w_traject = np.loadtxt('save_data/'+name+'/'+files_w_traject[j], unpack=True)
            w_stiff = np.loadtxt('save_data/'+name+'/'+files_w_stiff[j], unpack=True)
            
            DMP_traject = pDMP(DOF, N, alpha, beta, lambd, dt, h)
            DMP_stiff = pDMP(DOF, N, alpha, beta, lambd, dt, h)
            
            if DOF == 1:
                DMP_traject.set_weights(0, w_traject)
                DMP_stiff.set_weights(0, w_stiff)
            else:
                for i in range(DOF):
                    DMP_traject.set_weights(i, w_traject[i,:])
                    DMP_stiff.set_weights(i, w_stiff[i,:])
            
            
            DMP_traject.set_state(np.array([L]))
            
            data = []
            
            for idx in range(int(samples)):
                # generate phase
                phi = phase[idx]
                
                # Get trajectory      
                DMP_traject.set_phase( np.array([phi]) )
                DMP_traject.set_period( np.array([tau]) )
                
                DMP_stiff.set_phase( np.array([phi]) )
                DMP_stiff.set_period( np.array([tau]) )
                
                DMP_traject.repeat()
                DMP_stiff.repeat()
                
                DMP_traject.integration()
                DMP_stiff.integration()
                
                x, _, _ , _ = DMP_traject.get_state()
                k, _, _, _ = DMP_stiff.get_state()
                
                data.append([x[0], k[0]])
                
            data = np.asarray(data)
           
            if j % 2 == 0:
                ax1[0,i].plot(t, data[:,0], label = "run "+str(j+1))
                ax1[1,i].plot(t, data[:,1], label = "run "+str(j+1))
                
    
            
            ax2[j].plot(t, data[:,0],label = name +' = ' + str(v))
            ax2[j].set_xticklabels([])
            ax2[j].set_ylabel('x'+str(j+1), fontsize='13')  
            
            ax3[j].plot(t, data[:,1],label = name +' = ' + str(v))
            ax3[j].set_xticklabels([])
            ax3[j].set_ylabel('K'+str(j+1), fontsize='13')  
            
            ax4[i].plot(rfftfreq(samples, dt), np.abs(rfft(data[:,0])), label = 'run '+str(j+1))
            ax4[i].set_ylabel('Power')
        
               
       
        ax1[0,i].set_xticklabels([])
        ax1[1,i].set_xticklabels([])
        
        if i == 0:
            ax1[0,i].set_ylabel('x[m]', fontsize='13')
            ax1[1,i].set_ylabel('K [N/m]', fontsize='13')
            
            ax2[-1].set_xlabel('time [s]', fontsize='13')
            ax3[-1].set_xlabel('time [s]', fontsize='13')
            
        ax1[1,i].set_xlabel('time [s]', fontsize='13')
        
        if i is not len(N_all):
            ax4[i].set_xticklabels([])
        
          
    ax1[0,-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax4[-1].set_xlabel('frequency')
    ax4[2].legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
    
      
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
        
        
    fig1.savefig('images/'+name+'/ST_'+name+'_total.png')
    fig2.savefig('images/'+name+'/ST_'+name+'_total_x.png')
    fig3.savefig('images/'+name+'/ST_'+name+'_total_k.png')
    fig4.savefig('images/'+name+'/St_'+name+'_frequency')

    fig1.savefig('images/final/ST_'+name+'_total.png')
    fig2.savefig('images/final/ST_'+name+'_total_x.png')
    fig3.savefig('images/final/ST_'+name+'_total_k.png')
    fig4.savefig('images/final/St_'+name+'_frequency')
    
    