import numpy as np
import cv2
import pickle
import scipy.signal

cap = cv2.VideoCapture(0)
N=2
g=open('videorecord.txt', 'wb')
o=open('original.txt', 'wb')
filt1=np.ones((N,N))/N
filt2=scipy.signal.convolve2d(filt1,filt1)/N

#Get size of frame:
retval, frame = cap.read()
rows,columns,d = frame.shape
print(rows,columns)

#Prevous Y frame:
Yprev=np.zeros((rows,columns))
#Vectors for current frame as graphic:
framevectors=np.zeros((rows,columns,3))
#motion vectors, for each block a 2-d vector:
mv=np.zeros((int(rows/8),int(columns/8),2))

#Process 25 frames:
for n in range(25):
    print("Frame no ",n)
    ret, frame = cap.read()
    [rows,columns,c]=frame.shape

    if ret==True:

        #Here goes the processing to reduce data...
        reduced = np.zeros((rows,columns,c))
        Y=(0.114*frame[:,:,0]+0.587*frame[:,:,1]+0.299*frame[:,:,2]);

        Cb=(0.4997*frame[:,:,0]-0.33107*frame[:,:,1]-0.16864*frame[:,:,2]);

        Cr=(-0.081282*frame[:,:,0]-0.418531*frame[:,:,1]+0.499813*frame[:,:,2]);
        reduced[:,:,0]=Y
        reduced[:,:,1]=Cb
        reduced[:,:,2]=Cr

        #print(grid3.shape)
        #print(framevectors.shape)

        cv2.imshow('Original',frame/255.0+framevectors)#

        #Two color components are filtered first
        Crfilt=scipy.signal.convolve2d(Cr,filt2,mode='same')
        Cbfilt=scipy.signal.convolve2d(Cb,filt2,mode='same')

        # Downsampling
        DCr=Crfilt[0::N,::N];
        DCb=Cbfilt[0::N,::N];
        block = [8,8]
        framevectors=np.zeros((rows,columns,3))

        for yblock in range(int((rows-8)/8)):#(480-8)/8
            #print("yblock=",yblock)
            block[0]=yblock*8+8;#200
            for xblock in range(int((columns-8)/8)):#(640-8)/8
                #print("xblock=",xblock)
                block[1]=xblock*8+8;#300
                #current block:
                Yc=Y[block[0]:block[0]+8 ,block[1]:block[1]+8]                      #Current frame
                #previous block:
                Yp=Yprev[block[0]-8 : block[0]+8, block[1]-8 : block[1]+8]          #Previous frame
                #Some high value for MAE for initialization:
                bestmae=100.0;

                Ycorr=scipy.signal.fftconvolve(Yp, Yc[::-1,::-1], mode='valid')
                #print Ycorr

                index2d=np.unravel_index(np.argmax(Ycorr),(Ycorr.shape))            #Index2d will have the pixel index where the block has maximum correlation
#########################################Maximum correlation value will lead to the position where there is maximum similarity ##########################################################
############################################## between the current and the previous block hence motion estimation #######################################################################
                secarg = np.add(block,index2d)
                cv2.line(framevectors, tuple(block)[::-1], tuple(secarg)[::-1],(0.0,1.0,0.0),1)
                #cv2.line(framevectors, (block[1], block[0]),(block[1] + mv[yblock, yblock, 1].astype(int), block[0] + mv[yblock, yblock, 0].astype(int)),(1.0, 1.0, 1.0));
        Yprev=Y.copy()
        #converting  images back to integer:
        frame=np.array(frame,dtype='uint8')
        Y=np.array(Y, dtype='uint8')
        DCr=np.array(DCr, dtype='int8')
        DCb=np.array(DCb, dtype='int8')
        #"Serialize" the captured video frame (convert it to a string)
        #using pickle, and write/append it to file g:
#       
        pickle.dump(frame,o)
          
        pickle.dump(Y,g)
        pickle.dump(DCr,g)
        pickle.dump(DCb,g)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
print ("Uncompressed file size")
print (o.tell()/(1024*1024) , "MBs")
print ("Compressed file size")
print (g.tell()/(1024*1024) , "MBs")

# Release everything if job is finished
cap.release()
g.close()
o.close()
cv2.destroyAllWindows()
