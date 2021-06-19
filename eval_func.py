import numpy as np
seed = 7 
np.random.seed(seed)

from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, auc


def evaluate_new(keras_model, npz_file, epoch):
    name_String="Eval_epoch"+str(epoch)+"_"
    print "\n"
    thefile=np.load(npz_file)
    sc = StandardScaler().fit(thefile["track"])
    print sc.mean_
    sc2 = StandardScaler().fit(thefile["track1"])
    print sc2.mean_
    print sc2.scale_
    
    len2=len(thefile["track"])
    range1=range(0,len2,2)
    range2=range(1,len2,2)
    
    model_train=keras_model.predict([sc.transform(thefile["track"][range2]), sc2.transform(thefile["track1"][range2]), thefile["image"][range2]])
    
    #variables to be plotted
    
    hit_pos=thefile["hit_pos"][range2,1]
    delta=thefile["delta_TH"][range2,1]
    track_pos=thefile["track_pos"][range2,1]
    clucenter=thefile["local_cluster"][range2,1]
    cluBroken=thefile["local_brokencluster"][range2,1]
    broken=thefile["Flag"][range2,0]
    
    #for i in range(100):
     #   print model_train[i], "vs", hit_pos[i]
        #print thefile["truth1"][i,1]
        

    plt.hist(model_train[broken==0][:,0] - hit_pos[broken==0], bins=50, range=[-3,3], alpha=0.5,label='hit position estimate')
    plt.hist(clucenter[broken==0] - hit_pos[broken==0], bins=50, range=[-3,3], alpha=0.5,label='cluster center')
    plt.legend(loc=1)
    plt.xlabel("estimate of hit_position_Y - hit_position_Y [local units]")
    plt.ylabel("# clusters")
    plt.savefig(name_String+"hit_cluster_full.png")
    plt.clf()

    plt.hist(model_train[broken==1][:,0] - hit_pos[broken==1], bins=50, range=[-3,3], alpha=0.5,label='hit position estimate')
    plt.hist(clucenter[broken==1] - hit_pos[broken==1], bins=50, range=[-3,3], alpha=0.5,label='cluster center')
    plt.legend(loc=1)
    plt.xlabel("estimate of hit_position_Y - hit_position_Y [local units]")
    plt.ylabel("# clusters")
    plt.savefig(name_String+"hit_cluster_broken.png")
    plt.clf()

    plt.hist(model_train[broken==1][:,0]-hit_pos[broken==1], bins=50, range=[-3,3], alpha=0.5, color='red', edgecolor='red', label="hit pos estimate")
    plt.hist(track_pos[broken==1]-hit_pos[broken==1], bins=50, range=[-3,3], alpha=0.5, edgecolor='none',label="track-hit position")
    plt.legend(loc=1)
    plt.xlabel("estimate of hit_position_Y - hit_position_Y [local units]")
    plt.ylabel("# clusters")
    plt.savefig(name_String+"hitPosReso_track_onlyBroken.png")
    plt.clf()

    
'''

    plt.hist(model_train[broken==1][:,0]-hit_pos[broken==1], bins=100, range=[-10,10], alpha=0.5, color='red', edgecolor='red', label="hit pos estimate")
    plt.hist(clucenter[broken==1], bins=100, range=[-2,2], alpha=0.5, color='gold',edgecolor='none', label="cluster center")
    plt.legend(loc=1)
    plt.xlabel("estimate of hit_position_Y - hit_position_Y [local units]")
    plt.ylabel("# clusters")
    plt.savefig(name_String+"hitPosReso_onlyBroken.png")
    plt.clf()

    plt.hist(model_train[:,0]-hit_pos, bins=100, range=[-10,10], alpha=0.5, color='red', edgecolor='red', label="hit pos estimate")
    plt.hist(clucenter, bins=100, range=[-2,2], alpha=0.5, color='gold',edgecolor='none', label="cluster center")
    plt.legend(loc=1)
    plt.xlabel("estimate of hit_position_Y - hit_position_Y [local units]")
    plt.ylabel("# clusters")
    plt.savefig(name_String+"hitPosReso_Full.png")
    plt.clf()

    plt.hist(model_train[broken==1][:,0]-hit_pos[broken==1], bins=100, range=[-2,2], alpha=0.5, color='red', edgecolor='red', label="hit pos estimate")
    plt.hist(delta, bins=100, range=[-2,2], alpha=0.5, edgecolor='none',label="track position")
    plt.legend(loc=1)
    plt.xlabel("estimate of hit_position_Y - hit_position_Y [local units]")
    plt.ylabel("# clusters")
    plt.savefig(name_String+"hitPosReso_delta_onlyBroken.png")
    plt.clf()

    plt.hist(model_train[broken==1][:,0]-hit_pos[broken==1], bins=100, range=[-2,2], alpha=0.5, color='red', edgecolor='red', label="hit pos estimate")
    plt.hist(track_pos, bins=100, range=[-2,2], alpha=0.5, edgecolor='none',label="track-hit position")
    plt.legend(loc=1)
    plt.xlabel("estimate of hit_position_Y - hit_position_Y [local units]")
    plt.ylabel("# clusters")
    plt.savefig(name_String+"hitPosReso_track_onlyBroken.png")
    plt.clf()

    plt.hist(model_train[:,0]-track_pos, bins=100, range=[-2,2], alpha=0.5, edgecolor='none',label="hit pos estimate-track_pos")
    plt.hist(delta, bins=100, range=[-2,2], alpha=0.5, edgecolor='none',label="track-hit position")
    plt.legend(loc=1)
    plt.xlabel("estimate of hit_position_Y - track_position_Y [local units]")
    plt.ylabel("# clusters")
    plt.savefig(name_String+"plot_resolutionTrack.png")
    plt.clf() 
  

    plt.hist(model_train[:,0], bins=100, range=[-10,10], alpha=0.5,label='training')
    plt.hist(hit_pos, bins=100, range=[-10,10], alpha=0.5,label='hit position')
    plt.hist(clucenter, bins=100, range=[-10,10], alpha=0.5,label='cluster center')
    plt.legend(loc=1)
    plt.savefig(name_String+"Result_fullclusters.png")
    plt.clf()

    plt.hist(hit_pos, bins=100, range=[-10,10], alpha=0.5,label='hit_pos - cluster_center')
    plt.hist(clucenter, bins=100, range=[-10,10], alpha=0.5,label='cluster center')
    plt.legend(loc=1)
    plt.savefig(name_String+"Result_brokencluster.png")
    plt.clf()
 
    plt.hist(model_train[:,0]-hit_pos, bins=100, range=[-2,2], alpha=0.5, edgecolor='none')
    plt.savefig(name_String+"plot_EstimateOK.png")
    plt.clf() 


    plt.hist(model_train[:,0]-track_pos, bins=100, range=[-2,2], alpha=0.5, edgecolor='none')
    #plt.hist(delta, bins=100, range=[-2,2], alpha=0.5)
    plt.savefig(name_String+"plot_resolutionTrack1.png")
    plt.clf() 

   

    plt.hist2d(hit_pos, model_train[:,0],bins=100, norm=LogNorm())
    cb=plt.colorbar()
    plt.savefig(name_String+"plot_corrRainbowLog.png")
    cb.remove()
    plt.clf()


    plt.hist(model_train[:,0]-track_pos, bins=100, range=[-2,2], alpha=0.5, edgecolor='none')
    plt.hist(model_train[:,0]-clucenter, bins=100, range=[-2,2], alpha=0.5, edgecolor='none')
    plt.hist(model_train[:,0]-cluBroken, bins=100, range=[-2,2], alpha=0.5, edgecolor='none')
    plt.savefig(name_String+"plot_resolutionCluster.png")
    plt.clf()


    plt.hist(model_train[:,0]-hit_pos, bins=100, range=[-2,2], alpha=0.5, color='red', edgecolor='red', label="hit pos estimate")
    plt.hist(track_pos-hit_pos, bins=100, range=[-2,2], alpha=0.5, color='lime',edgecolor='none', label="track pos")
    plt.hist(clucenter-hit_pos, bins=100, range=[-2,2], alpha=0.5, color='gold',edgecolor='none', label="cluster center")
    plt.hist(cluBroken*(broken==1)+clucenter*(broken==0)-hit_pos, bins=100, range=[-2,2], alpha=0.5, color='aqua',edgecolor='none', label="center w bias")
    plt.legend(loc=1)
    plt.savefig(name_String+"plot_resolution_hitPos.png")
    plt.clf()

    plt.hist(model_train[broken==1][:,0]-hit_pos[broken==1], bins=100, range=[-2,2], alpha=0.5, color='red', edgecolor='red', label="hit pos estimate")
    plt.hist(track_pos[broken==1]-hit_pos[broken==1], bins=100, range=[-2,2], alpha=0.5, color='lime',edgecolor='none', label="track pos")
    plt.hist(clucenter[broken==1]-hit_pos[broken==1], bins=100, range=[-2,2], alpha=0.5, color='gold',edgecolor='none', label="cluster center")
    plt.hist(cluBroken[broken==1]-hit_pos[broken==1], bins=100, range=[-2,2], alpha=0.5, color='aqua',edgecolor='none', label="center w bias")
    plt.legend(loc=1)
    plt.savefig(name_String+"plot_resolution_hitPos_onlyBroken.png")
    plt.clf()

    plt.hist(model_train[broken==0][:,0]-hit_pos[broken==0], bins=100, range=[-2,2], alpha=0.5, color='red', edgecolor='red', label="hit pos estimate")
    plt.hist(track_pos[broken==0]-hit_pos[broken==0], bins=100, range=[-2,2], alpha=0.5, color='lime',edgecolor='none', label="track pos")
    plt.hist(clucenter[broken==0]-hit_pos[broken==0], bins=100, range=[-2,2], alpha=0.5, color='gold',edgecolor='none', label="cluster center")
    plt.hist(clucenter[broken==0]-hit_pos[broken==0], bins=100, range=[-2,2], alpha=0.5, color='aqua',edgecolor='none', label="center w bias")
    plt.legend(loc=1)
    plt.savefig(name_String+"plot_resolution_hitPos_onlyFull.png")
    plt.clf()
    
    
    plt.hist(model_train[:,0][broken==0]-hit_pos[broken==0], bins=100, range=[-2,2], alpha=0.5, edgecolor='none', color='royalblue', label="full clusters")
    plt.hist(model_train[:,0][broken==1]-hit_pos[broken==1], bins=100, range=[-2,2], alpha=0.5, edgecolor='none', color='gold', label="broken clusters")
    plt.xlabel("estimate of hit_position_Y - hit_position_Y [local units]")
    plt.ylabel("# clusters")
    plt.legend(loc=1)
    plt.savefig(name_String+"plot_Estimate_sovr.png")
    plt.clf() 
    
    plt.hist([model_train[:,0][broekn==0]-hit_pos[broekn==0],model_train[:,0][broken==1]-hit_pos[broken==1]], bins=100, range=[-2,2], alpha=0.5, edgecolor='none', color=['royalblue','gold'], label=["full clusters","broken clusters"], stacked=True)   
    plt.xlabel("estimate of hit_position_Y - hit_position_Y [local units]")
    plt.ylabel("# clusters")
    plt.legend(loc=1)
    plt.savefig(name_String+"plot_Estimate_stack.png")
    plt.clf()
    
    plt.hist(model_train[broken==0][:,0]-hit_pos[broken==0], bins=100, range=[-2,2], alpha=0.5, color='royalblue', edgecolor='royalblue', label="hit pos estimate")
    plt.hist(clucenter[broken==0]-hit_pos[broken==0], bins=100, range=[-2,2], alpha=0.5, color='gold',edgecolor='none', label="cluster center")
    plt.xlabel("estimate of hit_position_Y - hit_position_Y [local units]")
    plt.ylabel("# clusters")
    plt.legend(loc=1)
    plt.savefig(name_String+"plot_resolution_hitPos_onlycluster.png")
    plt.clf()
    
    plt.hist(model_train[broken==0][:,0]-hit_pos[broken==0], bins=100, range=[-0.5,0.5], alpha=0.5, color='royalblue', edgecolor='royalblue', label="hit pos estimate")
    plt.hist(clucenter[broken==0]-hit_pos[broken==0], bins=100, range=[-0.5,0.5], alpha=0.5, color='gold',edgecolor='none', label="cluster center")
    plt.xlabel("estimate of hit_position_Y - hit_position_Y [local units]")
    plt.ylabel("# clusters")
    plt.legend(loc=1)
    plt.savefig(name_String+"plot_resolution_hitPos_onlycluster_small.png")
    plt.clf()
    
    plt.hist(model_train[broken==1][:,0]-hit_pos[broken==1], bins=100, range=[-2,2], alpha=0.5, color='royalblue', edgecolor='royalblue', label="hit pos estimate")
    plt.hist(cluBroken[broken==1]-hit_pos[broken==1], bins=100, range=[-2,2], alpha=0.5, color='gold',edgecolor='none', label="cluster center")
    plt.xlabel("estimate of hit_position_Y - hit_position_Y [local units]")
    plt.ylabel("# clusters")
    plt.legend(loc=1)
    plt.savefig(name_String+"plot_resolution_hitPos_onlyclusterBroken.png")
    plt.clf()
    
    plt.hist(model_train[broken==1][:,0]-hit_pos[broken==1], bins=100, range=[-2,2], alpha=0.5, color='royalblue', edgecolor='royalblue', label="hit pos estimate")
    plt.hist(track_pos[broken==1]-hit_pos[broken==1], bins=100, range=[-2,2], alpha=0.5, color='lime',edgecolor='none', label="track position Y")
    plt.xlabel("estimate of hit_position_Y - hit_position_Y [local units]")
    plt.ylabel("# clusters")
    plt.legend(loc=1)
    plt.savefig(name_String+"plot_resolution_hitPos_onlyTraBroken.png")
    plt.clf()
    
'''
