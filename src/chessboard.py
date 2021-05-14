import cv2
import numpy as np
from collections import defaultdict

def segment_by_angle_kmeans(lines, k=2, **kwargs):

    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    angles = np.array([line[0][1] for line in lines])
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)

    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

def segmented_intersections(lines):
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections

def clust(intersections):
    result = []
    for point in intersections:
        if (len(result)==0):
            result.append(point)
            continue
        flag = True
        for p in result:
            if (pow(p[0][0]-point[0][0],2)+pow(p[0][1]-point[0][1],2)<50):
                flag = False
        if (flag):
            result.append(point)
    return result

img = cv2.imread('../img/4.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3),0)
cv2.imwrite('../result/gray_blur.png', blur)
edges = cv2.Canny(blur,50,200,L2gradient=False)
cv2.imwrite('../result/canny.png', edges)
lines = cv2.HoughLines(edges,1,np.pi/180,60)
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)

cv2.imwrite('../result/hough.png',img)

segmented = segment_by_angle_kmeans(lines)
intersections = segmented_intersections(segmented)
pts = clust(intersections)
for point in pts:
    cv2.circle(img,tuple(point[0]),2,(255,255,0))

cv2.imwrite('../result/intersection.png',img)