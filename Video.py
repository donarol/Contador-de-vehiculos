import cv2
import numpy as np

class Video:
    def __init__(self):
        self.video = {"direccion":"","nombre":"","width":0,"height":0}
        self.puntos_src_H = []
        self.puntos_dst_H = []
        self.img_out = {"nombre":"","width":0,"height":0,"img":None}
        self.kernel_dilatacion = 3
        self.contorno = {"min":0,"max":0,"height":0,"herencia":False}

    def confVideo(self,video,nombre,width,height):
        self.video['direccion'] = video
        self.video['nombre'] = nombre
        self.video['width'] = width
        self.video['height'] = height

    def setImgOut(self,nombre,width,height):
        self.img_out['nombre'] = nombre
        self.img_out['width'] = width
        self.img_out['height'] = height

    def setHomografia(self,src,dst):
        self.puntos_src_H = np.array(src)
        self.puntos_dst_H = np.array(dst)

    def setContorno(self,min,max,height,herencia):
        self.contorno["min"] = min
        self.contorno["max"] = max
        self.contorno["height"] = height
        self.contorno["herencia"] = herencia 
        
    def getSovel(self,img):
        frame_float=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        frame_float=frame_float.astype(float)
        Hsx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        Hsy=np.transpose(Hsx)
        bordex=cv2.filter2D(frame_float,-1,Hsx)
        bordey=cv2.filter2D(frame_float,-1,Hsy)
        Mxy=bordex**2+bordey**2 
        Mxy=np.sqrt(Mxy)
        Mxy=Mxy/np.max(Mxy)
        mask=np.where(Mxy>0.1,255,0)
        mask=np.uint8(mask)
        return mask

    def getContorno(self,img):
        contorno = img.copy()
        contornos = []
        contours, hierarchy = cv2.findContours(contorno,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        con = 0
        for cont in contours:
            area = cv2.contourArea(cont)
            x,y,w,h = cv2.boundingRect(cont)
            if self.contorno["herencia"]:
                if area > self.contorno["min"] and area < self.contorno["max"] and h > self.contorno["height"]:    
                    M = cv2.moments(cont)
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    contornos.append({"x":x,"y":y,"w":w,"h":h,"cx":cx,"cy":cy})
            else:
                if area > self.contorno["min"] and area < self.contorno["max"] and h > self.contorno["height"] and hierarchy[0][con][3]==-1:                
                    M = cv2.moments(cont)
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    contornos.append({"x":x,"y":y,"w":w,"h":h,"cx":cx,"cy":cy})
            con = con + 1
        return contornos

    def dibujaContorno(self,img,contornos):
        for cont in contornos:
            x,y,w,h,cx,cy = cont['x'],cont['y'],cont['w'],cont['h'],cont['cx'],cont['cy']
            cv2.circle(img,(cx,cy),5,(0,255,0),-1)
            area = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]]) 
            cv2.drawContours(img,[area],-1,(255,0,0),2)
        cv2.putText(img,"numero de autos: {}".format(len(contornos)),(10,10),1,1,(0,255,0),1)

    def iniciarVideo(self):
        cap = cv2.VideoCapture(self.video['direccion'])   
        
        cv2.namedWindow(self.video['nombre'])
        cv2.moveWindow(self.video['nombre'], 0,0)     
        kernel_dilatacion = np.ones((self.kernel_dilatacion,self.kernel_dilatacion),np.uint8) 

        while True:
            ret, frame = cap.read()
            if ret == False: 
                print("error al reproducir el video")
                break
            frame = cv2.resize(frame, (self.video['width'],self.video['height']), interpolation = cv2.INTER_AREA)
            #homografia
            ho,status = cv2.findHomography(self.puntos_src_H,self.puntos_dst_H)
            self.img_out["img"] = cv2.warpPerspective(frame,ho,(self.img_out["width"],self.img_out["height"]))
            #bordes
            mask = self.getSovel(self.img_out["img"])
            #dilatacion
            dilatacion = cv2.dilate(mask,kernel_dilatacion,iterations=1)
            #contornos
            contornos = self.getContorno(dilatacion)

            self.dibujaContorno(self.img_out["img"],contornos)
            
            cv2.imshow(self.video["nombre"],frame)
            cv2.imshow("bordes",mask)
            cv2.imshow("dilatacion",dilatacion)
            cv2.imshow(self.img_out["nombre"],self.img_out["img"])

            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
            
w_out = 800
h_out = 450
video = Video()
video.confVideo('video/video_negro.MP4', "nombre",1500,900)
video.setImgOut('res',w_out,h_out)
video.kernel_dilatacion = 9
video.setHomografia([[36,496],[355,496],[355,661],[36,661]],[[0,0],[w_out,0],[w_out,h_out],[0,h_out]])
video.setContorno(600,60000,30,False)
video.iniciarVideo()



