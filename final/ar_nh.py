# -*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
from PIL import Image
import numpy as np
from webcam import Webcam
import camera
from Find_Keypoint import find_point as fp

def my_calibration(sz):
    row, col = sz
    fx = 983
    fy = 983
    K = np.diag([fx, fy, 1])
    K[0, 2] = 331
    K[1, 2] = 232
    #with np.load('calib.npz') as X:
    #    mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
    #print("mtx : ", mtx, "dist : ", dist, "rvecs : ", rvecs, "tvecs : ", tvecs)
    return K
##############################
class ObjLoader(object):
    def __init__(self, fileName):
        self.vertices = []
        self.faces = []
        ##
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                    self.vertices.append(vertex)

                elif line[0] == "f":
                    string = line.replace("//", "/")
                    ##
                    i = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            face.append(string[i:-1])
                            break
                        face.append(string[i:string.find(" ", i)])
                        i = string.find(" ", i) + 1
                    ##
                    self.faces.append(tuple(face))

            f.close()
        except IOError:
            print(".obj file not found.")

    def render_scene(self):
        if len(self.faces) > 0:
            glMatrixMode(GL_MODELVIEW)
            glBegin(GL_TRIANGLES)
            for face in self.faces:
                for f in face:
                    vertexDraw = self.vertices[int(f) - 1]
                    glVertex3fv(vertexDraw)
            glEnd()
##############################
class OpenGLGlyphs:
    ##############################################################초기화
    def __init__(self):
        # initialise webcam and start thread
        self.webcam = Webcam()
        self.webcam.start()
        self.find = fp()
        self.find.set_img('book.jpg')

        self.hei, self.wid = self.webcam.get_frame_shape()[:2]
        # initialise cube
        self.d_obj = None
        self.img = None
        # initialise texture
        self.texture_background = None
        self.K = None
        self.mark_kp = None
        self.mark_des = None
        self.set_keypoint()
        self.new_kp = None

        self.mat_kp = None
        self.mat_des = None
        self.H = None

        # self.Rt=None

    ##############################################################카메라 세팅
    def _init_gl(self, Width, Height):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        self.K = my_calibration((Height, Width))
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        fovy = 2 * np.arctan(0.5 * Height / fy) * 180 / np.pi
        aspect = (float)(Width * fy) / (Height * fx)
        # define the near and far clipping planes
        near = 0.1
        far = 100.0
        # set perspective
        gluPerspective(fovy, aspect, near, far)

        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)

    ##############################################################marker의 kp, des저장
    def set_keypoint(self):

        self.find.start()
        self.mark_kp, self.mark_des = self.find.get_point()

    ##############################################################K값 구하기
    def _draw_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # get image from webcam
        image = self.webcam.get_current_frame()

        Rt = self._my_cal(image)

        # convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = Image.fromarray(bg_image)

        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)

        # create background texture
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.wid, self.hei, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)

        # draw background
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glPushMatrix()

        # glTranslatef(0.0,0.0,0.0)
        gluLookAt(0.0, 0.0, 12, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        self._draw_background()
        glPopMatrix()
        ################Rt를 구해서 매칭되는 이미지가 있는지 판단

        if Rt is not None:
            self._set_modelview_from_camera(Rt)
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_NORMALIZE)
            glClear(GL_DEPTH_BUFFER_BIT)
            ObjLoader("jnu.obj").render_scene()

        glutSwapBuffers()

    ##############################################################OpenGL용 Rt변환
    def _set_modelview_from_camera(self, Rt):

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        Rx = np.array([[0.2, 0, 0], [0, 0, 0.2], [0, 0.2, 0]])

        # set rotation to best approximation
        R = Rt[:, :3]

        # change sign of x-axis
        R[0, :] = -R[0, :]
        # set translation
        t = Rt[:, 3]
        t[0] = -t[0]

        # setup 4*4 model view matrix
        M = np.eye(4)
        M[:3, :3] = np.dot(R, Rx)
        M[:3, 3] = t
        M[3, :3] = t

        # transpose and flatten to get column order
        M = M.T

        m = M.flatten()
        # replace model view with the new matrix
        glLoadMatrixf(m)

    ##############################################################Rt반환
    def _my_cal(self, image):
        find_H = fp()
        find_H.set_cv_img(image)
        find_H.start()
        kp, des = find_H.get_point()

        self.H = self.match_images(self.mark_kp, self.mark_des, kp, des)
        if self.H is not None:
            cam1 = camera.Camera(np.hstack((self.K, np.dot(self.K, np.array([[0], [0], [-1]])))))
            # Rt1=dot(linalg.inv(self.K),cam1.P)
            cam2 = camera.Camera(np.dot(self.H, cam1.P))

            A = np.dot(np.linalg.inv(self.K), cam2.P[:, :3])
            A = np.array([A[:, 0], A[:, 1], np.cross(A[:, 0], A[:, 1])]).T
            cam2.P[:, :3] = np.dot(self.K, A)
            Rt = np.dot(np.linalg.inv(self.K), cam2.P)

            return Rt
        else:
            return None

    ##############################################################match image
    def match_images(self, kp1, des1, kp2, des2):
        matcher = cv2.BFMatcher()
        match_des = matcher.knnMatch(des1, des2, k=2)
        matches = []
        matA, matB = [], []
        matC = []

        for m in match_des:
            if m[0].distance < 0.8 * m[1].distance:
                matA.append(kp1[m[0].queryIdx])
                matB.append(kp2[m[0].trainIdx])
                matC.append(des1[m[0].queryIdx])

        if len(matA) > 50:
            ptsA = np.float32([m.pt for m in matA])
            ptsB = np.float32([n.pt for n in matB])
            H1 = []
            H1, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)
            H1 = self.homo_check(H1)
            self.mat_kp = np.array([matB[i] for i in range(status.shape[0]) if status[i] == 1])
            self.mat_des = np.array([matC[i] for i in range(status.shape[0]) if status[i] == 1])

            return H1
        else:
            return None

    ##############################################################homography check
    def homo_check(self, H1):
        if self.H is None:
            return H1
        else:
            if cv2.norm(H1, self.H) > 1.0:
                return H1
            else:
                return self.H

    def _draw_background(self):
        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0);
        glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0);
        glVertex3f(4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0);
        glVertex3f(4.0, 3.0, 0.0)
        glTexCoord2f(0.0, 0.0);
        glVertex3f(-4.0, 3.0, 0.0)
        glEnd()
        glDeleteTextures(1)

    def keyboard(self, *args):
        if args[0] == GLUT_KEY_UP:
            glutDestroyWindow(self.window_id)
            self.webcam.finish()
            sys.exit()

    ##############################################################OpenGL창 초기

    def main(self):
        # setup and run OpenGL
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.wid, self.hei)
        glutInitWindowPosition(200, 200)
        self.window_id = glutCreateWindow(b"OpenGL Glyphs")
        self._init_gl(self.wid, self.hei)
        glutDisplayFunc(self._draw_scene)
        glutIdleFunc(self._draw_scene)
        glutMainLoop()


# run an instance of OpenGL Glyphs
openGLGlyphs = OpenGLGlyphs()
openGLGlyphs.main()
sys.exit()