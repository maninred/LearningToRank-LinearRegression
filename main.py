import numpy as np
import random
import csv

t=[]
X=[]
# #The following commented is for retriving data for the actual file from the microsoft server
# with open("Querylevelnorm.txt") as data_file:
#     for line in data_file:
#         temparr=[]
#         count=count+1
#         Arr = line.split(" ")
#         t.append(float(Arr[0]))
#         for i in range(2,48):
#             temp = Arr[i].split(":")
#             temparr.append(float(temp[1]))
#         X.append(temparr)
firstfile  = open('Querylevelnorm_X.csv', "rU")
firstreader = csv.reader(firstfile)
for row in firstreader:
    temm=[]
    for col in row:
        temm.append(float(col))
    X.append(temm)

secondfile= open('Querylevelnorm_t.csv', "rU")
secondreader = csv.reader(secondfile)
for row in secondreader:
    t.append(float(row[0]))

npAllX=np.array(X)
Num=len(npAllX)
lenX=int(Num*0.8)
npX=npAllX[0:lenX]
val_npX=npAllX[lenX:lenX+int(0.1*Num)]
tes_npX=npAllX[lenX+int(0.1*Num):Num]
nptAll=np.array(t)
npt=nptAll[0:lenX]
val_npt=nptAll[lenX:lenX+int(0.1*Num)]
tes_npt=nptAll[lenX+int(0.1*Num):Num]
temc=[]
for i in range(0,46):
    temm=[]
    for j in range(0,46):
        if(i==j):
            tvar=np.var(npX[:,i])
            if(tvar!=0):
                temm.append(tvar*(0.1))
            else:
                temm.append(0.000001)
        else:
            temm.append(0)
    temc.append(temm)
npSig=np.array(temc)
npSig_inv=np.linalg.inv(npSig)

M=12
print "The value of M is choosen "+str(M)
rand=random.sample(range(1,len(npX)), M-1)
Mu=[]
for i in range(0,len(rand)):
    Mu.append(X[i])
npMu=np.array(Mu)
Pi=[]
for i in range(0,len(npX)):
    temppi=[1]
    for j in range(0,M-1):
        a=np.exp((-1)*(0.5)*np.dot(np.dot(((npX[i]-npMu[j]).T),npSig_inv),(npX[i]-npMu[j])))
        temppi.append(a)
    Pi.append(temppi)
npPi=np.array(Pi)

#Calculation fro testing set -80%

np_lamda=0.01
print "the value of lambda is choosen "+str(np_lamda)
w=np.dot((np.linalg.inv((np_lamda*np.array(np.identity(M)))+np.dot((npPi.T),npPi))),np.dot(npPi.T,npt))
Eww=(0.5)*np.dot(w.T,w)
tempp=0
for i in range(0,len(npX)):
    tempp=tempp+((npt[i]-np.dot(w.T,npPi[i]))**2)
Edw=(0.5)*tempp
Ew=Edw+(np_lamda*Eww)
Erms=((2*Ew)/len(npX))**(0.5)
print "The actual Erms value is"
print Erms


# Calculating the same for validation set- 20 %

val_Pi=[]
for i in range(0,len(val_npX)):
    temppi=[1]
    for j in range(0,M-1):
        a=np.exp((-1)*(0.5)*np.dot(np.dot(((val_npX[i]-npMu[j]).T),npSig_inv),(val_npX[i]-npMu[j])))
        temppi.append(a)
    val_Pi.append(temppi)
val_npPi=np.array(val_Pi)
val_w=np.dot((np.linalg.inv((np_lamda*np.array(np.identity(M)))+np.dot((val_npPi.T),val_npPi))),np.dot(val_npPi.T,val_npt))
val_Eww=(0.5)*np.dot(val_w.T,val_w)
val_tempp=0
for i in range(0,len(val_npX)):
    val_tempp=val_tempp+((val_npt[i]-np.dot(val_w.T,val_npPi[i]))**2)
val_Edw=(0.5)*val_tempp
val_Ew=val_Edw+(np_lamda*val_Eww)
val_Erms=((2*val_Ew)/len(val_npX))**(0.5)
print "The value of Validation Erms is"
print val_Erms

#Calculation for testing set

tes_Pi=[]
for i in range(0,len(tes_npX)):
    temppi=[1]
    for j in range(0,M-1):
        a=np.exp((-1)*(0.5)*np.dot(np.dot(((tes_npX[i]-npMu[j]).T),npSig_inv),(tes_npX[i]-npMu[j])))
        temppi.append(a)
    tes_Pi.append(temppi)
tes_npPi=np.array(tes_Pi)
tes_w=np.dot((np.linalg.inv((np_lamda*np.array(np.identity(M)))+np.dot((tes_npPi.T),tes_npPi))),np.dot(tes_npPi.T,tes_npt))
tes_Eww=(0.5)*np.dot(tes_w.T,tes_w)
tes_tempp=0
for i in range(0,len(tes_npX)):
    tes_tempp=tes_tempp+((tes_npt[i]-np.dot(tes_w.T,tes_npPi[i]))**2)
tes_Edw=(0.5)*tes_tempp
tes_Ew=val_Edw+(np_lamda*tes_Eww)
tes_Erms=((2*tes_Ew)/len(tes_npX))**(0.5)
print "The value of test Erms is"
print tes_Erms
print ""
print ""

# SGD

#training
SGD_W=[]
etta=0.01
for i in range(M):
    SGD_W.append(random.random())
for j in range(0,len(npX)):
    if(j==0):
        temppp=np.array(SGD_W)
    delEd=(-1)*np.dot((npt[j]-np.dot(temppp.T,npPi[j])),npPi[j])
    delEw=temppp.T
    delE=delEd+np_lamda*delEw
    SGD_W_1=temppp+((-1)*etta*(delE))
    temppp=np.array(SGD_W_1)
# print "The value of W from SGD is"
# print SGD_W_1
SGD_Eww=(0.5)*np.dot(SGD_W_1.T,SGD_W_1)
SGD_tempp=0
for i in range(0,len(npX)):
    SGD_tempp=SGD_tempp+((npt[i]-np.dot(SGD_W_1.T,npPi[i]))**2)
SGD_Edw=(0.5)*SGD_tempp
SGD_Ew=SGD_Edw+(np_lamda*SGD_Eww)
SGD_Erms=((2*SGD_Ew)/len(npX))**(0.5)

print "The actual value of SGD_Erms for training set is"
print SGD_Erms

# SGD- validation set
for j in range(0,len(val_npX)):
    if(j==0):
        temppp=np.array(SGD_W)
    delEd=(-1)*np.dot((val_npt[j]-np.dot(temppp.T,npPi[j])),npPi[j])
    delEw=temppp.T
    delE=delEd+np_lamda*delEw
    SGD_W_1=temppp+((-1)*etta*(delE))
    temppp=np.array(SGD_W_1)
# print "The value of W from SGD is"
# print SGD_W_1
SGD_Eww=(0.5)*np.dot(SGD_W_1.T,SGD_W_1)
SGD_tempp=0
for i in range(0,len(npX)):
    SGD_tempp=SGD_tempp+((npt[i]-np.dot(SGD_W_1.T,npPi[i]))**2)
SGD_Edw=(0.5)*SGD_tempp
SGD_Ew=SGD_Edw+(np_lamda*SGD_Eww)
val_SGD_Erms=((2*SGD_Ew)/len(npX))**(0.5)
print "The value of SGD_Erms for validation set is"
print val_SGD_Erms

#SGD- testing set
for j in range(0,len(tes_npX)):
    if(j==0):
        temppp=np.array(SGD_W)
    delEd=(-1)*np.dot((tes_npt[j]-np.dot(temppp.T,npPi[j])),npPi[j])
    delEw=temppp.T
    delE=delEd+np_lamda*delEw
    SGD_W_1=temppp+((-1)*etta*(delE))
    temppp=np.array(SGD_W_1)
# print "The value of W from SGD is"
# print SGD_W_1
SGD_Eww=(0.5)*np.dot(SGD_W_1.T,SGD_W_1)
SGD_tempp=0
for i in range(0,len(npX)):
    SGD_tempp=SGD_tempp+((npt[i]-np.dot(SGD_W_1.T,npPi[i]))**2)
SGD_Edw=(0.5)*SGD_tempp
SGD_Ew=SGD_Edw+(np_lamda*SGD_Eww)
tes_SGD_Erms=((2*SGD_Ew)/len(npX))**(0.5)
print "The value of SGD_Erms for testing set is"
print tes_SGD_Erms
print ""
print ""

# the following code is for the synthesised data

ifile  = open('input.csv', "rU")
reader = csv.reader(ifile)
Syn_X_all=[]
for row in reader:
    temm=[]
    for col in row:
        temm.append(float(col))
    Syn_X_all.append(temm)
np_Syn_X_all=np.array(Syn_X_all)
np_Syn_X=np_Syn_X_all[0:int(len(np_Syn_X_all)*(0.8))]
val_np_Syn_X=np_Syn_X_all[int(len(np_Syn_X_all)*(0.8)):int(len(np_Syn_X_all)*(0.9))]
tes_np_Syn_X=np_Syn_X_all[int(len(np_Syn_X_all)*(0.9)):len(np_Syn_X_all)]

ifile_o  = open('output.csv', "rU")
reader_o = csv.reader(ifile_o)
Syn_t_all=[]
for row in reader_o:
    Syn_t_all.append(float(row[0]))
np_Syn_t_all=np.array(Syn_t_all)
np_Syn_t=np_Syn_t_all[0:int(len(np_Syn_t_all)*(0.8))]
val_np_Syn_t=np_Syn_t_all[int(len(np_Syn_t_all)*(0.8)):int(len(np_Syn_t_all)*(0.9))]
tes_np_Syn_t=np_Syn_t_all[int(len(np_Syn_t_all)*(0.9)):len(np_Syn_t_all)]

temc=[]
for i in range(0,len(np_Syn_X[0])):
    temm=[]
    for j in range(0,len(np_Syn_X[0])):
        if(i==j):
            tvar=np.var(np_Syn_X[:,i])
            if(tvar!=0):
                temm.append(tvar*(0.1))
            else:
                temm.append(0.000001)
        else:
            temm.append(0)
    temc.append(temm)
Syn_npSig=np.array(temc)
Syn_npSig_inv=np.linalg.inv(Syn_npSig)

Syn_M=5
print "The value of M for the synthetic data is choosen "+str(Syn_M)
rand1=random.sample(range(1,len(np_Syn_X)), Syn_M-1)
Syn_Mu=[]
for i in range(0,len(rand1)):
    Syn_Mu.append(np_Syn_X[i])
Syn_npMu=np.array(Syn_Mu)
Syn_Pi=[]
for i in range(0,len(np_Syn_X)):
    temppi=[1]
    for j in range(0,Syn_M-1):
        a=np.exp((-1)*(0.5)*np.dot(np.dot(((np_Syn_X[i]-Syn_npMu[j]).T),Syn_npSig_inv),(np_Syn_X[i]-Syn_npMu[j])))
        temppi.append(a)
    Syn_Pi.append(temppi)
Syn_npPi=np.array(Syn_Pi)

Syn_np_lamda=0.01
print "The value of lambda for the synthetic data is choosen "+str(Syn_np_lamda)
Syn_w=np.dot((np.linalg.inv((Syn_np_lamda*np.array(np.identity(Syn_M)))+np.dot((Syn_npPi.T),Syn_npPi))),np.dot(Syn_npPi.T,np_Syn_t))

# for the training set
# print "The value of Syn_W is"
# print Syn_w
Syn_Eww=(0.5)*np.dot(Syn_w.T,Syn_w)
tempp=0
for i in range(0,len(np_Syn_X)):
    tempp=tempp+((np_Syn_t[i]-np.dot(Syn_w.T,Syn_npPi[i]))**2)
Syn_Edw=(0.5)*tempp
Syn_Ew=Syn_Edw+(Syn_np_lamda*Syn_Eww)
Syn_Erms=((2*Syn_Ew)/len(np_Syn_X))**(0.5)
print "The actual Syn_Erms value for training set is"
print Syn_Erms


# Calculating the same for validation set
val_Pi=[]
for i in range(0,len(val_np_Syn_X)):
    temppi=[1]
    for j in range(0,Syn_M-1):
        a=np.exp((-1)*(0.5)*np.dot(np.dot(((val_np_Syn_X[i]-Syn_npMu[j]).T),Syn_npSig_inv),(val_np_Syn_X[i]-Syn_npMu[j])))
        temppi.append(a)
    val_Pi.append(temppi)
val_npPi=np.array(val_Pi)
val_w=np.dot((np.linalg.inv((Syn_np_lamda*np.array(np.identity(Syn_M)))+np.dot((val_npPi.T),val_npPi))),np.dot(val_npPi.T,val_np_Syn_t))
val_Eww=(0.5)*np.dot(val_w.T,val_w)
val_tempp=0
for i in range(0,len(val_np_Syn_X)):
    val_tempp=val_tempp+((val_np_Syn_t[i]-np.dot(val_w.T,Syn_npPi[i]))**2)
val_Edw=(0.5)*val_tempp
val_Ew=val_Edw+(Syn_np_lamda*val_Eww)
val_Erms=((2*val_Ew)/len(val_np_Syn_X))**(0.5)
print "The value of Validation Erms for synthetic  data is"
print val_Erms

#Calculation for testing set

tes_Pi=[]
for i in range(0,len(tes_np_Syn_X)):
    temppi=[1]
    for j in range(0,Syn_M-1):
        a=np.exp((-1)*(0.5)*np.dot(np.dot(((tes_np_Syn_X[i]-Syn_npMu[j]).T),Syn_npSig_inv),(tes_np_Syn_X[i]-Syn_npMu[j])))
        temppi.append(a)
    tes_Pi.append(temppi)
tes_npPi=np.array(tes_Pi)
tes_w=np.dot((np.linalg.inv((Syn_np_lamda*np.array(np.identity(Syn_M)))+np.dot((tes_npPi.T),tes_npPi))),np.dot(tes_npPi.T,val_np_Syn_t))
tes_Eww=(0.5)*np.dot(tes_w.T,tes_w)
tes_tempp=0
for i in range(0,len(tes_np_Syn_X)):
    tes_tempp=tes_tempp+((val_np_Syn_t[i]-np.dot(tes_w.T,tes_npPi[i]))**2)
tes_Edw=(0.5)*tes_tempp
tes_Ew=val_Edw+(Syn_np_lamda*tes_Eww)
tes_Erms=((2*tes_Ew)/len(tes_np_Syn_X))**(0.5)
print "The value of test Erms for the synthetic set is"
print tes_Erms
print ""
print ""

# SGD - for training set
Syn_SGD_W=[]
Syn_etta=0.01
for i in range(Syn_M):
    Syn_SGD_W.append(random.random())
for j in range(0,len(np_Syn_X)):
    if(j==0):
        temppp=np.array(Syn_SGD_W)
    delEd=(-1)*np.dot((np_Syn_t[j]-np.dot(temppp.T,Syn_npPi[j])),Syn_npPi[j])
    delEw=temppp.T
    delE=delEd+Syn_np_lamda*delEw
    Syn_SGD_W_1=temppp+((-1)*Syn_etta*(delE))
    temppp=np.array(Syn_SGD_W_1)
Syn_SGD_Eww=(0.5)*np.dot(Syn_SGD_W_1.T,Syn_SGD_W_1)
Syn_SGD_tempp=0
for i in range(0,len(np_Syn_X)):
    Syn_SGD_tempp=Syn_SGD_tempp+((np_Syn_t[i]-np.dot(Syn_SGD_W_1.T,Syn_npPi[i]))**2)
Syn_SGD_Edw=(0.5)*Syn_SGD_tempp
Syn_SGD_Ew=Syn_SGD_Edw+(Syn_np_lamda*Syn_SGD_Eww)
Syn_SGD_Erms=((2*Syn_SGD_Ew)/len(np_Syn_X))**(0.5)

print "The value of Syn_SGD_Erms for training set is"
print Syn_SGD_Erms


# SGD- validation set
for j in range(0,len(val_np_Syn_X)):
    if(j==0):
        temppp=np.array(Syn_SGD_W)
    delEd=(-1)*np.dot((val_np_Syn_t[j]-np.dot(temppp.T,Syn_npPi[j])),Syn_npPi[j])
    delEw=temppp.T
    delE=delEd+Syn_np_lamda*delEw
    SGD_W_1=temppp+((-1)*Syn_etta*(delE))
    temppp=np.array(SGD_W_1)
# print "The value of W from SGD is"
# print SGD_W_1
SGD_Eww=(0.5)*np.dot(SGD_W_1.T,SGD_W_1)
SGD_tempp=0
for i in range(0,len(np_Syn_X)):
    SGD_tempp=SGD_tempp+((np_Syn_t[i]-np.dot(SGD_W_1.T,Syn_npPi[i]))**2)
SGD_Edw=(0.5)*SGD_tempp
SGD_Ew=SGD_Edw+(Syn_np_lamda*SGD_Eww)
val_SGD_Erms=((2*SGD_Ew)/len(np_Syn_X))**(0.5)
print "The value of SGD_Erms for validation set  in synthetic data is"
print val_SGD_Erms

#SGD- testing set
for j in range(0,len(tes_np_Syn_X)):
    if(j==0):
        temppp=np.array(Syn_SGD_W)
    delEd=(-1)*np.dot((tes_np_Syn_t[j]-np.dot(temppp.T,Syn_npPi[j])),Syn_npPi[j])
    delEw=temppp.T
    delE=delEd+Syn_np_lamda*delEw
    SGD_W_1=temppp+((-1)*Syn_etta*(delE))
    temppp=np.array(SGD_W_1)
# print "The value of W from SGD is"
# print SGD_W_1
SGD_Eww=(0.5)*np.dot(SGD_W_1.T,SGD_W_1)
SGD_tempp=0
for i in range(0,len(tes_np_Syn_X)):
    SGD_tempp=SGD_tempp+((tes_np_Syn_t[i]-np.dot(SGD_W_1.T,Syn_npPi[i]))**2)
SGD_Edw=(0.5)*SGD_tempp
SGD_Ew=SGD_Edw+(Syn_np_lamda*SGD_Eww)
tes_SGD_Erms=((2*SGD_Ew)/len(tes_np_Syn_X))**(0.5)
print "The value of SGD_Erms for testing set in synthetic  data is"
print tes_SGD_Erms

print "The end of the program"