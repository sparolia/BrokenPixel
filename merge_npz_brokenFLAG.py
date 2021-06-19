import numpy as np
import sys


file1=sys.argv[1]
file2=sys.argv[2]

arra1=np.load(file1)
arra2=np.load(file2)

lista=[]

print arra1.files

a=arra1["track"]
b=arra2["track"]
print a.shape, b.shape
track=np.concatenate((a,b))


a=arra1["track1"]
b=arra2["track1"]
print a.shape, b.shape
track1=np.concatenate((a,b))

a=arra1["image"]
b=arra2["image"]
print a.shape, b.shape
image=np.concatenate((a,b))

a=arra1["hit_pos"]
b=arra2["hit_pos"]
print a.shape, b.shape
hit_pos=np.concatenate((a,b))

a=arra1["delta_TH"]
b=arra2["delta_TH"]
print a.shape, b.shape
delta_TH=np.concatenate((a,b))

a=arra1["track_pos"]
b=arra2["track_pos"]
print a.shape, b.shape
track_pos=np.concatenate((a,b))

a=arra1["Truth_image"]
b=arra2["Truth_image"]
print a.shape, b.shape
Truth_image=np.concatenate((a,b))


a=arra1["local_brokencluster"]
b=arra2["local_brokencluster"]
print a.shape, b.shape
Seville (SVQ)local_brokencluster=np.concatenate((a,b))


a=arra1["local_cluster"]
b=arra2["local_cluster"]
print a.shape, b.shape
local_cluster=np.concatenate((a,b))


a=arra1["theCenter"]
b=arra2["theCenter"]
print a.shape, b.shape
theCenter=np.concatenate((a,b))


Broken=np.ones((len(a),1))
Full=np.zeros((len(b),1))
print Broken.shape, Full.shape
Flag=np.concatenate((Broken,Full))

np.savez("merged_wFlag"+sys.argv[1]+sys.argv[2], track=track, track1=track1, image=image, hit_pos=hit_pos, delta_TH=delta_TH, track_pos=track_pos, Truth_image=Truth_image,
local_brokencluster=local_brokencluster, local_cluster=local_cluster, theCenter=theCenter, Flag=Flag)



#for f in arra1.files:
    #print f
    
    #a=arra1[f]
    #b=arra2[f]
    
    #print a.shape, b.shape
    
    #c=np.concatenate((a,b))
    
    #print c.shape
    #lista.append(c)
    
#np.savez("merged", lista[1] )
    
    
    
