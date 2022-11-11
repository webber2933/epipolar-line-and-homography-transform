# HW2

**1. Fundamental Matrix Estimation from Point Correspondences:**

(1) read_point(file): 

input 2D座標的文字檔，轉成homogenous coordinate儲存

(2) linear_list_square(point1,point2):

input 兩兩對應的座標點，每一組對應座標點可以提供一個式子
```
for i in range(num):
    u1 = point1[i][0]
    v1 = point1[i][1]
    u2 = point2[i][0]
    v2 = point2[i][1]
    A[i] = np.array([u1*u2, u1*v2, u1, v1*u2, v1*v2, v1, u2, v2, 1])
```

對A做SVD後，取VT的最後一個row作為f
```
# compute the SVD of A
U, D, VT = np.linalg.svd(A, full_matrices=True)
# take the rightmost column of V which corresponds to the smallest singular value
f = VT[-1, :]
# reshape f into a 3*3 matrix
f = np.reshape(f, (3, 3))
```

再對f做SVD，並讓D只保留前2個singular value，以滿足rank two constraint
```
# enforce the rank-two constraint
U, D, VT = np.linalg.svd(f, full_matrices=True)
D_rank2 = np.zeros((3, 3))
D_rank2[0][0] = D[0]
D_rank2[1][1] = D[1]
fundamental = np.dot(U, np.dot(D_rank2, VT))
```

(3) normalized_eight_point_algorithm(point1,point2):

在linear least square前先normalize點座標，讓座標在x,y的平均都為0
```
mean1 = np.mean(point1_euclidean, axis=0)
mean2 = np.mean(point2_euclidean, axis=0)

point1_zero_mean = point1_euclidean - mean1
point2_zero_mean = point2_euclidean - mean2
```
計算每個點到中心點的平均距離來設scale，透過mean和scale來設定transformation matrix
```
T1 = np.array([[scale1, 0, -mean1[0] * scale1],
               [0, scale1, -mean1[1] * scale1],
               [0, 0, 1]])
```
2組對應點分別用不同的matrix normalize後做linear least square，最後再recover fundamental matrix
```
normalized_pt1 = T1.dot(point1.T).T
normalized_pt2 = T2.dot(point2.T).T
fundamental = linear_least_square(normalized_pt1, normalized_pt2)
fundamental = np.dot(np.dot(T1.T,fundamental),T2)
```

(4) draw_epipolar_line_and_cal_avg_dist(point1,point2,img1,img2,F):

input 2組對應點和圖片，以及要使用哪一種fundamental matrix，透過Fp'取得p'點在另一張圖對應的epipolar line(ax+by+c = 0)，其中line[0][i]為第i點的對應的a，line[1][i]=b，line[2][i]=c
```
line1 = F.dot(point2.T)
```
取通過線上的兩個點(0,-c/b),(W,-(aW + c)/b)畫線
```
x1,y1,x2,y2 = 0,int(-line1[2][i]/line1[1][i]),img1.shape[1],int(-(line1[2][i]+line1[0][i]*img1.shape[1])/line1[1][i])
cv2.line(img1,(x1,y1),(x2,y2),(0, 0, 255),1)
```

**2. Homography transform :**

(1) estimate_homography(point1,point2):

input 兩兩對應的座標點，每一組座標點提供2個式子
```
for i in range(num):
    x1 = point1[i][0]
    y1 = point1[i][1]
    x2 = point2[i][0]
    y2 = point2[i][1]
    A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
    A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
```
和上題一樣對A做SVD得到AH=0的linear least squares solution

(2) rectify:

透過backward warping找到每個target座標對應到的source座標，因為是homogeneous coordinate所以將x,y除以z

```
pixel = H_inv.dot(np.array([j,i,1]))
pixel /= pixel[2]
x = pixel[0]
y = pixel[1]
```
得到source座標後，透過bilinear interpolation對這個座標的周圍4個整點取值，並用內插法算出這個target點的RGB值
```
a = x - int(x)
b = int(y) + 1 - y
value = (1-a)*(1-b)*original_img[int(y) + 1][int(x)]
value += a*(1-b)*original_img[int(y) + 1][int(x) + 1]
value += a*b*original_img[int(y)][int(x) + 1]
value += (1-a)*b*original_img[int(y)][int(x)] 

rectified_img[i][j] = value
```