import numpy as np
from numpy.linalg import norm as norm
from numba import njit, float64
from mink_control.optimized_functions import T_1F, T_ji
from mink_control.Object_v2 import Cylinder, cyl_get_coords
from mink_control.env_config import *

# shift = np.array([[0,0],[np.pi/2, np.pi/2],[np.pi/2, np.pi/2],[0,0],[-np.pi/2,-np.pi/2]])

# S = [S1, S2, S3, S4, S5] offsets
S = np.array([.135+.202, 0, 0, .2675, 0])
# l = [l12, l23, l34, l45, l56] link lengths
l = np.array([0, .2015, 0, 0, .175+.14])
# a = [a12, a23, a34, a45, a56]
a = np.array([np.pi/2, 0, np.pi/2, -np.pi/2, 0])
r = np.sum(S[1:]) + np.sum(l)
limits = np.array([[-r,r],[-r,r],[0,r+S[0]]])
shift = np.array([0,-np.pi/2,-np.pi/2,0,np.pi/2])

cyl1 = Cylinder(r=.05, L=l[1])
cyl2 = Cylinder(r=.05, L=S[3])
cyl3 = Cylinder(r=.05, L=l[4])
body1 = cyl1.original
body2 = cyl2.original
body3 = cyl3.original

@njit(nogil=True)
def get_transforms(th, S, a, l):
    th = th-shift
    T1F = T_1F(th[0],S[0])
    T_2F = T1F@T_ji(th[1], a[0], l[0], S[1])
    T_3F = T_2F@T_ji(th[2], a[1], l[1], S[2])
    T_4F = T_3F@T_ji(th[3], a[2], l[2], S[3])
    T_5F = T_4F@T_ji(th[4], a[3], l[3], S[4])

    T_arr = np.zeros((5,4,4))
    T_arr[0,:,:] = T1F
    T_arr[1,:,:] = T_2F[:,:]
    T_arr[2,:,:] = T_3F[:,:]
    T_arr[3,:,:] = T_4F[:,:]
    T_arr[4,:,:] = T_5F[:,:]

    return T_arr

def make_plot(th, S, a, l):
    th = th - shift
    # T_arr = get_transforms(th, S, a, l)
    O_i = np.array([0,0,0,1])
    Ptool_5 = np.array([l[-1],0,0,1])
    TF = T_1F(th[0], S[0])
    T_21 = T_ji(th[1], a[0], l[0], S[1])
    T_32 = T_ji(th[2], a[1], l[1], S[2])
    T_43 = T_ji(th[3], a[2], l[2], S[3])
    T_54 = T_ji(th[4], a[3], l[3], S[4])

    points = np.zeros((5,4))
    points[0,:] = np.zeros(4)
    points[1,:] = TF@T_21@O_i
    points[2,:] = TF@T_21@T_32@O_i
    points[3,:] = TF@T_21@T_32@T_43@T_54@O_i
    points[4,:] = TF@T_21@T_32@T_43@T_54@Ptool_5

    xx = points[:,0]
    yy = points[:,1]
    zz = points[:,2]
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xx,yy,zz)
    ax.plot3D(xx,yy,zz)
    r = np.sum(S[1:]) + np.sum(l)
    ax.set_xlim3d(left=-r, right=r)
    ax.set_ylim3d(bottom=-r, top=r)
    ax.set_zlim3d(bottom=0, top=r+S[0])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return points 

@njit(nogil=True)
def forward(th, S, a, l): #, make_plot=False):
    th = th - shift
    TF = T_1F(th[0], S[0])
    T_21 = T_ji(th[1], a[0], l[0], S[1])
    T_32 = T_ji(th[2], a[1], l[1], S[2])
    T_43 = T_ji(th[3], a[2], l[2], S[3])
    T_54 = T_ji(th[4], a[3], l[3], S[4])
    Ptool_5 = np.array([l[-1],0,0,1])
    O = np.array([0.0,0.0,0.0,1.0])
    T_5f = TF@T_21@T_32@T_43@T_54
    F5 = T_5f@O
    eef = T_5f@Ptool_5
    u = (eef - F5)/np.linalg.norm(eef - F5)
    return np.vstack((eef[0:3],u[0:3]))

@njit(nogil=True)
def calc_eef_vel(th,w):
    eef1 = forward(th,S,a,l)
    eef1 = eef1[0,:]
    th2 = th + w*.0001
    eef2 = forward(th2,S,a,l)
    eef2 = eef2[0,:]
    eef_vel = (eef2 - eef1)/.0001
    return eef_vel, eef1

@njit((float64[:,:])(float64[:],float64[:],float64[:],float64[:]), nogil=True)
def points(th,S,a,l):
    T_arr = get_transforms(th,S,a,l)
    O_i = np.array([0.0,0.0,0.0,1.0])#,dtype=float)
    Ptool_5 = np.array([l[-1],0.0,0.0,1.0])#,dtype=float)
    points = np.zeros((4,5))
    points[:,0] = np.array([0.0,0.0,0.0,1.0])#,dtype=float)
    points[:,1] = np.array([0.0,0.0,S[0],1.0])#,dtype=float)
    points[:,2] = T_arr[2,:,:]@O_i
    points[:,3] = T_arr[4,:,:]@O_i
    points[:,4] = T_arr[4,:,:]@Ptool_5

    return points 

def reverse(p, u, S, a, l, make_plot=False):
    '''
    inputs:
        p : tool point of the eef asb fixed frame
        u : unit vect describing the orientation of the eef, parallel
            w/ a56
    outputs:
        th_arr : two joint angle solutions satisfying the reverse analysis
    '''
    # th1 
    # two solutions but only interested in the first one
    a56 = l[-1]
    O5_f = p - a56*u
    x,y = O5_f[0],O5_f[1]
    r = np.linalg.norm([x,y])
    if x == 0 and y == 0:
        th1 = 0.0
        case1 = True
    else:
        c1 = x/r
        s1 = y/r
        th1 = np.arctan2(s1,c1)
        case1 = False

    O2_f = np.array([0,0,S[0]])
    # th2 and th3
    # th2: two solutions branchs
    # th3: one solution branch
    v25 = O5_f - O2_f
    x25,y25,z25 = v25[0],v25[1],v25[2]
    xy = np.sqrt(x25**2 + y25**2)
    aph = np.arctan2(z25, xy) 
    # th2
    # apply cosine law                 /B\
    # a^2 = b^2 + c^2 - 2*b*c*cos(A) A/___\C
    # O2_f@A, O3_f@B, O5_f@C
    a_,c = S[3],l[1] 
    b = np.sqrt(xy**2 + z25**2)
    if np.round(b,6) == np.round(a_+c,6):
        # links 2 and 3 are colinear/parallel
        th2 = np.array([aph, aph])
        th3 = np.array([np.pi/2, np.pi/2])
    elif np.round(b,6) == 0.0:
        # arm is foled in on itself
        th1 = np.array([aph,aph])
        th2 = np.array(-np.pi/2, -np.pi/2)
    else: 
        A = np.arccos((b**2+c**2-a_**2)/(2*b*c))
        th2 = np.array([A + aph, aph - A])

        cB = (a_**2+c**2-b**2)/(2*a_*c)
        sB = np.sin(A)*b / a_
        B = np.arctan2(sB,cB)
        th3 = np.array([B-np.pi/2, 3*np.pi/2-B])

    # th4 and th5
    # each have one solution branch
    # T = T_1F @ T_21 @ T_32
    T1 = T_1F(th1, S[0])@T_ji(th2[0], a[0], l[0], S[1])@T_ji(th3[0],a[1],l[1],S[2])
    T2 = T_1F(th1, S[0])@T_ji(th2[1], a[0], l[0], S[1])@T_ji(th3[1],a[1],l[1],S[2])
    # [                  0]
    # [a34, S3xa34, S3,  0]
    # [                  0]
    # [0,        0,   0, 1]
    T_arr = np.array([T1, T2])
    th4 = np.zeros(2)
    th5 = np.zeros(2)
    for i in range(0,2):
        T = T_arr[i,:,:]
        a34_f = T[0:3,0]
        O3_f = T@np.array([0,0,0,1])
        O3_f = O3_f[0:3]
        S4_f = (O5_f - O3_f) / np.linalg.norm(O5_f - O3_f)
        S5_f = np.cross(S4_f,u)
        if np.all(np.round(S5_f,12) == 0.0):
            # S5_f and u are colinear, th4 could be anything
            S5_f = np.cross(S4_f, a34_f)
        S5_f = S5_f/np.linalg.norm(S5_f)
        a45_f = np.cross(S5_f,S4_f)
        a45_f = a45_f/np.linalg.norm(a45_f)

        # th4 
        c4 = np.dot(a34_f, a45_f)
        s4 = np.dot(S4_f,np.cross(a34_f, a45_f))
        th4[i] = np.arctan2(s4,c4)

        # th5
        c5 = np.dot(a45_f, u)
        s5 = np.dot(S5_f, np.cross(a45_f,u))
        th5[i] = np.arctan2(s5,c5)

    temp = np.array([th1,th1])
    th = np.vstack((temp, th2, th3, th4, th5))
    
    # zero position is straight up, must adjust th 
    th = th + np.vstack((shift,shift)).T
    return th.T

def get_coords(th, time_step, make_plot=False):
    res = .01
    # th = th - shift 
    T_arr = get_transforms(th, S, a, l)
    # body 1 
    T1 = T_arr[1,:,:]
    Rot_y = np.array([[np.cos(np.pi/2), 0, np.sin(np.pi/2), 0],
                [0, 1, 0, 0],
                [-np.sin(np.pi/2), 0, np.cos(np.pi/2), 0],
                [0,0,0,1]])
    T1 = T1@Rot_y
    coords1, feat1 = cyl_get_coords(body1.T, time_step, T1, res, limits)
    # body 2
    T2 = T_arr[2,:,:]
    Rot_x = np.array([[1,0,0,0],
                  [0, np.cos(np.pi/2), -np.sin(np.pi/2), 0],
                  [0, np.sin(np.pi/2), np.cos(np.pi/2), 0],
                  [0,0,0,1]])
    T2 = T2@Rot_x
    coords2, feat2 = cyl_get_coords(body2.T, time_step, T2, res, limits)
    # body 3
    T3 = T_arr[4,:,:]
    T3 = T3@Rot_y
    coords3, feat3 = cyl_get_coords(body3.T, time_step, T3, res, limits)

    coords = np.vstack((coords1,coords2,coords3))
    feats = np.vstack((feat1, feat2, feat3))

    if make_plot:
        xx = coords[:,1]
        yy = coords[:,2]
        zz = coords[:,3]
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(xx,yy,zz)
        ax.plot3D(xx,yy,zz)
        r = np.sum(S[1:])/res + np.sum(l)/res
        ax.set_xlim3d(left=0, right=2*r)
        ax.set_ylim3d(bottom=0, top=2*r)
        ax.set_zlim3d(bottom=0, top=r+S[0]/res)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    return coords, feats

    



    
# th = np.zeros(5, dtype=float)
# th[0] = -np.pi/2
# th[1] = -np.pi/4
# print(th)
# temp = forward(th,S,a,l)
# p = temp[0,:]
# u = temp[1,:]
# print(np.round(forward(th,S,a,l),3))
# print(np.round(reverse(p, u, S, a, l),3))

# phi = -np.pi/3
# p = np.array([.3,.3,.3])
# u = np.array([0,1.0,0.0])
# print('p',np.round(p,3))
# print('u',np.round(u,3))
# u = u / np.linalg.norm(u)
# th = reverse(p, u, S, a, l)
# print(np.round(180*th/np.pi,3))
# # th[-1,0] = th[-1,0]*-1
# temp = forward(th[0,:],S, a, l)
# print(np.round(temp,3))

# print(th[0,:]*180/np.pi)
# points = make_plot(th[0,:],S,a,l)
# print(points)