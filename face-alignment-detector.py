import face_alignment
import cv2
import math
from math import atan,sqrt

frame = cv2.imread('Test_images/image-1.jpeg')
#frame = cv2.flip(frame,1)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='sfd')
det = fa.get_landmarks_from_image(frame)
copy = frame.copy()

for i in range(len(det[0])):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    color = (0, 0, 255)
    thickness = 2
    # print((int(det[0][i][0]), int(det[0][i][1])))
    cv2.putText(copy, str(i), (int(det[0][i][0]), int(det[0][i][1])), font, fontScale,
                color, thickness, None, False)


def slope(x1,y1,x2,y2):

    if x2 != x1:
        return((y2-y1)/(x2-x1))
    else:
        return 'NA'

def per_dist(x1, y1, x2, y2, x3, y3):
    x12, y12 = x2 - x1, y2 - y1
    x13, y13 = x3 - x1, y3 - y1
    mod = sqrt(x12 * x12 + y12 * y12)
    dist = abs(x12 * y13 - y12 * x13) / mod
    return dist

def section(x1, x2, y1, y2, m, n):
    x = ((n * x1) + (m * x2)) / (m + n)
    y = ((n * y1) + (m * y2)) / (m + n)
    return int(x), int(y)

def drawLine(image,x1,y1,x2,y2,color):

    m = slope(x1,y1,x2,y2)
    h,w = image.shape[:2]
    if m != 'NA':

        px = 0
        py = -(x1-0)*m+y1
        qx = w
        qy = -(x2-w)*m+y2
    else:
        px,py = x1,0
        qx,qy = x1,h
    cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), color , 2)


def shortest_angle(M1, M2):
    PI = 3.14159265
    angle = abs((M2 - M1) / (1 + M1 * M2))
    ret = atan(angle)
    val = (ret * 180) / PI
    return val

def find_angle(m1, m2):
    pi = 3.141
    angle = ((m2-m1)/(1+m1*m2))
    ret = atan(angle)
    val = (ret*180)/pi
    if val < 0:
        val += 180
        return round(val, 2)
    else:
        return round(val, 2)

def acute_obtuse(i):
    return 180-i

def points(h):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    color = (0, 0, 255)
    thickness = 2
    cv2.putText(copy, str(h), (int(det[0][h][0]), int(det[0][h][1])), font, fontScale,
                 color, thickness,None, False)

def angle_lines(x1,y1,x2,y2,x3,y3,x4,y4):
    p1 = slope(x1,y1,x2,y2)
    p2 = slope(x3,y3,x4,y4)

    if p1 and p2 != 'NA':
        return find_angle(p1,p2)
    elif p1 == 'NA':
        return find_angle(p2,0)
    elif p2 == 'NA':
        return find_angle(p1,0)



chin_ratio2 = []
def chin_ratio1(a, b, c):
    if (round(abs(a[0] - b[0]) / abs(a[0] - c[0]), 1) == 0.50):
        chin_ratio2.append(
            "Tip of chin is at 1/2 times the distance between pupil perpendicular and head perpendicular line")
    if (round(abs(a[0] - b[0]) / abs(a[0] - c[0]), 1) < 0.50):
        f = "Tip of chin lies at the left of midpoint of pupil perpendicular and head perpendicular line by- " + str(
            round(0.50 - round(abs(a[0] - b[0]) / abs(a[0] - c[0]), 2), 2)) + " times the distance between the lines"

        chin_ratio2.append(f)

    if (round(abs(a[0] - b[0]) / abs(a[0] - c[0]), 1) > 0.50):
        g = "Tip of chin lies at the right of midpoint of pupil perpendicular and head perpendicular line by- " + str(
            round(round(abs(a[0] - b[0]) / abs(a[0] - c[0]), 2) - 0.50, 2)) + " times the distance between the lines"
        chin_ratio2.append(g)

def lip_ratio1(a, b, c):
    if (round(abs(a[0] - b[0]) / abs(a[0] - c[0]), 2) == 0.33):
        chin_ratio2.append(
            "-Lower lip line is at 1/3 times the distance between pupil perpendicular and head perpendicular line")
    if (round(abs(a[0] - b[0]) / abs(a[0] - c[0]), 2) < 0.33):
        h = "Lower lip line lies at the left of 1/3rd of pupil perpendicular and head perpendicular line point by- " + str(
            0.33 - round(abs(a[0] - b[0]) / abs(a[0] - c[0]), 2)) + " times the distance between the lines"
        chin_ratio2.append(h)

    if (round(abs(a[0] - b[0]) / abs(a[0] - c[0]), 2) > 0.33):
        i = "Lower lip line lies at the right of 1/3rd of pupil perpendicular and head perpendicular line point by- " + str(
            round(abs(a[0] - b[0]) / abs(a[0] - c[0]), 2) - 0.33) + " times the distance between the lines"
        chin_ratio2.append(i)

Nx,Ny = int(det[0][27][0]),int(det[0][27][1]) # Nasal Point
SNx,SNy = int(det[0][33][0]),int(det[0][33][1]) # Sub-nasal point
Pgx,Pgy = int(det[0][10][0]),int(det[0][10][1]) # Chin point
Po1x,Po1y = int(det[0][1][0]),int(det[0][1][1]) # ear point-1
Po2x,Po2y = int(det[0][2][0]),int(det[0][2][1]) # ear point-2
Prnx,Prny = int(det[0][30][0]),int(det[0][30][1]) # nose-tip point
Or1x,Or1y = int(det[0][40][0]),int(det[0][40][1]) # eye-point 1
Or2x,Or2y = int(det[0][41][0]),int(det[0][41][1]) # eye-point 2
GI1x,GI1y = int(det[0][21][0]),int(det[0][21][1])  #centre-point 1
GI2x,GI2y = int(det[0][22][0]),int(det[0][22][1])  #centre-point 2
PlipTx,PlipTy = int(det[0][50][0]),int(det[0][50][1]) # Protrusive upper lip
PlipBx,PlipBy = int(det[0][58][0]),int(det[0][58][1]) # Protrusive lower lip
Pox,Poy = section(Po1x,Po2x,Po1y,Po2y,1,1)            # Ear point
Orx,Ory = section(Or1x,Or2x,Or1y,Or2y,1,1)      # Eye point
GIx,GIy = section(GI1x,GI2x,GI1y,GI2y,1,1)            #Centre point
one3rdx,one3rdy = section(Pox,PlipBx,Poy,PlipBy,1,3) # point of 1-3rd of line connecting eye-point and bottom lip point

#N-SN line
drawLine(copy,Nx,Ny,SNx,SNy,(128,0,0))
cv2.putText(copy,'N',(Nx,Ny),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),3)
cv2.putText(copy,'SN',(SNx,SNy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),3)

#Po-Or line
drawLine(copy,Pox,Poy,Orx,Ory,(0,100,0))
cv2.putText(copy,'Po',(Pox,Poy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),3)
cv2.putText(copy,'Or',(Orx,Ory),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),3)

#S-line  Pg-SN
drawLine(copy,Pgx,Pgy,SNx,SNy,(0,0,128))
cv2.putText(copy,'Pg',(Pgx,Pgy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),3)

#E-line -  Prn-Pg
drawLine(copy,Pgx,Pgy,Prnx,Prny,(128,0,128))
cv2.putText(copy,'Prn',(Prnx,Prny),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),3)

#GI-SN line
drawLine(copy,GIx,GIy,SNx,SNy,(139,69,19))
cv2.putText(copy,'GI',(GIx,GIy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),3)

#sub-luxation line (Po- one-third of (Por-bottomlip line))
drawLine(copy,Pox,Poy,one3rdx,one3rdy,(139,69,19))

print('The angle between N-Sn line and Frankfort horizontal line is(Po=1)',angle_lines(Nx,Ny,SNx,SNy,Po1x,Po1y,Orx,Ory),'degrees')
print('The angle between N-Sn line and Frankfort horizontal line is(Po=1+2/2)',angle_lines(Nx,Ny,SNx,SNy,Pox,Poy,Orx,Ory),'degrees')
print('The facial angle is ',acute_obtuse(angle_lines(Nx,Ny,SNx,SNy,SNx,SNy,Pgx,Pgy)),'degrees')
print('The z-angle is(Po=1) ',acute_obtuse(angle_lines(Po1x,Po1y,Orx,Ory,Prnx,Prny,Pgx,Pgy)),'degrees')
print('The z-angle is(Po=1+2/2) ',acute_obtuse(angle_lines(Pox,Poy,Orx,Ory,Prnx,Prny,Pgx,Pgy)),'degrees')
print('The Horizontal angle of atlas subluxation line is(po=1)',find_angle(slope(Po1x,Po1y,one3rdx,one3rdy),0))
print('The Horizontal angle of atlas subluxation line is(po=1+2/2)',find_angle(slope(Pox,Poy,one3rdx,one3rdy),0))
#print('E-line distances')

chin_ratio1([det[0][41][0],det[0][9][1]],[det[0][9][0],det[0][9][1]],[det[0][27][0],det[0][9][1]])
lip_ratio1([det[0][41][0],det[0][9][1]],[det[0][9][0],det[0][9][1]],[det[0][27][0],det[0][9][1]])
print(chin_ratio2)


cv2.imshow('Final-Image',copy)

#cv2.imwrite('result-10-fhl.jpg',copy)