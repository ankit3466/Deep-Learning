import os
import sys
import cv2

path = 'C://Users/user/OneDrive/Desktop/rps/'

#print(os.path.exists(path))

label = sys.argv[1]
num_sample = int(sys.argv[2])

os.mkdir(os.path.join(path,label))

store_path = os.path.join(path,label)

start = False
count = 0

cap = cv2.VideoCapture(0)

while True:

    sat , frame = cap.read()

    if count == num_sample:
        break
    
    cv2.rectangle(frame , (100,100),(500,500),(255,255,255),2)

    

    if start:
        img = frame[100:500,100:500]
        save_path = os.path.join(store_path,'{}.jpg'.format(count+1))
        cv2.imwrite(save_path,img)
        count += 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
            (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start

    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

    
