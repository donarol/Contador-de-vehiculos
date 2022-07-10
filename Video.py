import cv2
import numpy as np

class Video:
    def __init__(self):
        self.frame_name = ["frame","mascara","dilatacion","salida"]
        self.video = {"direccion":"","nombre":"","width":0,"height":0}
        self.puntos_src_H = []
        self.puntos_dst_H = []
        self.img_out = {"nombre":"","width":1500,"height":500,"img":None}

    def confVideo(self,video,nombre,width=1500,height=900):
        self.video['direccion'] = video
        self.video['nombre'] = nombre
        self.video['width'] = width
        self.video['height'] = height

    def setHomografia(self,src,dst):
        self.puntos_src_H = np.array(src)
        self.puntos_dst_H = np.array(dst)

    def iniciarVideo(self):
        cap = cv2.VideoCapture(self.video['direccion'])        
        while True:
            ret, frame = cap.read()
            if ret == False: 
                print("error al reproducir el video")
                break
            cv2.namedWindow(self.video['nombre'])
            cv2.moveWindow(self.video['nombre'], 0,0)
            frame = cv2.resize(frame, (self.video['width'],self.video['height']), interpolation = cv2.INTER_AREA)
            #homografia
            ho,status = cv2.findHomography(self.puntos_src_H,self.puntos_dst_H)
            self.img_out["img"] = cv2.warpPerspective(frame,ho,(self.img_out["width"],self.img_out["height"]))
           
            
           
            cv2.imshow(self.video["nombre"],frame)
            cv2.imshow(self.img_out["nombre"],self.img_out["img"])

            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
            
video = Video()
video.confVideo('video/video_negro.MP4', "nombre")
video.setHomografia([[36,496],[355,496],[355,661],[36,661]],[[0,0],[1500,0],[1500,500],[0,500]])
video.iniciarVideo()





