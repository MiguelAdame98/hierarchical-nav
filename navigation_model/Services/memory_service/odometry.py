# Update 2022
# =============================================================================
# Ghent University 
# IDLAB of IMEC
# Daria de Tinguy - daria.detinguy at ugent.be
# =============================================================================

# Original Source
# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2013 Renato de Pontes Pereira, renato.ppontes at gmail dot com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import numpy as np
from .view_cells import compare_image_templates
from .modules import clip_rad_360
from pyquaternion import Quaternion
# from robosalad.runtime.datastructs import Pose, transform, inverse


class Odometry(object):
    def __init__(self):
        self.odometry = [0., 0., np.pi / 2]

    def __call__(self, observations, dt=0.1):
        odom = observations["odom"]
        vtrans = odom[-6] * dt
        vrot = odom[-1] * dt

        odom_angle = Quaternion(odom[6], odom[3], odom[4], odom[5])
        self.odometry[0] = odom[0]
        self.odometry[1] = odom[1]
        self.odometry[2] = odom_angle.angle
        return vtrans, vrot, self.odometry


class ActionOdometry(object):
    ''' Use cmd_vel for Odometry '''

    def __init__(self):
        self.odometry = [0., 0., np.pi / 2]

    def __call__(self, observations, dt=0.1):
        action = observations["action"]

        vtrans = action[0] * dt * 20
        vrot = action[-1] * dt * 0.85

        self.odometry[2] += vrot
        self.odometry[0] += vtrans * np.cos(self.odometry[2])
        self.odometry[1] += vtrans * np.sin(self.odometry[2])
        return vtrans, vrot, self.odometry

class HotEncodedActionOdometry(object):
    ''' Use HotEncoded Action for Odometry, gridworld adapted'''

    def __init__(self):
        self.odometry = [0., 0., np.pi/2]

    def __call__(self, action:list, dt:float=1)-> tuple[float, float]:
        #action = observations["HEaction"] #hot encoded action : [F,R,L]
        # print('action', action)
        vtrans = action[0] 
        vrot = 0
        if action[1] != 0: #right
            vrot = np.pi / 2 
        elif action[2] != 0: #left
            vrot = -np.pi / 2 

        self.odometry[2] += vrot
        self.odometry[2] = clip_rad_360(self.odometry[2])
        self.odometry[0] += vtrans * round(np.cos(self.odometry[2]),4) 
        self.odometry[1] += vtrans * round(np.sin(self.odometry[2]),4) # the round is there to correct the error on pi that would accumulate sin(2pi) != sin(0)
        #print('in action, vtrans, vrot, odom:', vtrans, vrot, self.odometry)
        return vtrans, vrot

    def position_applying_motion(self,odometry:list,action:list,dt:int=1)-> list:
        #hot encoded action : [F,R,L]
        vtrans = action[0] 
        vrot = 0
        if action[1] != 0: #right
            vrot = np.pi / 2 
        elif action[2] != 0: #left
            vrot = -np.pi / 2 

        odometry[2] += vrot
        odometry[2] = clip_rad_360(odometry[2])
        odometry[0] += vtrans * round(np.cos(odometry[2]),4) 
        odometry[1] += vtrans * round(np.sin(odometry[2]),4) 
        return odometry

class PoseOdometry(object):
    ''' Use pose for Odometry '''

    def __init__(self):
        self.latest = None
        self.odometry = None
        self.odometry = [0., 0., np.pi/2]
    def clip_rad_360(self,angle): #easierto debug
        while angle < 0:
            angle += 2 * np.pi
        while angle >= 2 * np.pi:
            angle -= 2 * np.pi
        return angle

    def __call__(self, odometry, observations, dt=0.1):
        p = observations["pose"]
        if len(p) == 3: #(xyth)
            print('pose', p)
            if 0 < p[2] < 4 and p[2].is_integer:
                p[2] = np.pi * p[2]
            
            if self.latest is None:
                self.latest = p
                diff = p- odometry
                #NOTE: the following only works when translation on only 1 axe
                vtrans,vrot = np.sum(np.sign(diff[:2])) * np.sqrt(pow(diff[0],2) + pow(diff[1],2)), p[2]
            else:
                vtrans = np.sqrt(pow(p[0] - odometry[0],2) + pow(p[1] - odometry[1],2))
                vrot = self.clip_rad_360(p[2] - odometry[2])                
                
            odometry = p
            print('in pose odom, vtrans, vrot, odom:', vtrans, vrot, odometry)
            return vtrans , vrot, odometry

        if len(p) == 7: #(xyzquat)
            pose = Pose.from_numpy(p)
            yaw, _, _ = pose.orientation.pyquaternion().yaw_pitch_roll
            # drone has Z pointing down
            yaw = -yaw
            self.odometry = [pose.position.x, pose.position.y, yaw]

            if self.latest is None:
                self.latest = pose
                return 0, 0
            else:
                diff = transform(pose, inverse(self.latest))

                vtrans = diff.position.x
                yaw, _, _ = diff.orientation.pyquaternion().yaw_pitch_roll
                vrot = -yaw
                self.latest = pose
                return vtrans / 100, vrot, self.odometry

        print('ERROR in memory_graph PoseOdometry')
class PoseOdometry(object):
    ''' Use pose for Odometry '''

    def __init__(self):
        self.latest = None
        self.odometry = None
        self.odometry = [0., 0., np.pi/2]
    def clip_rad_360(self,angle): #easierto debug
        while angle < 0:
            angle += 2 * np.pi
        while angle >= 2 * np.pi:
            angle -= 2 * np.pi
        return angle

    def __call__(self, observations, dt=0.1):
        p = observations["pose"]
        if len(p) == 3: #(xyth)
            print('pose', p)
            if 0 < p[2] < 4 and p[2].is_integer:
                p[2] = np.pi * p[2]
            
            if self.latest is None:
                self.latest = p
                diff = p- self.odometry
                #NOTE: the following only works when translation on only 1 axe
                vtrans,vrot = np.sum(np.sign(diff[:2])) * np.sqrt(pow(diff[0],2) + pow(diff[1],2)), p[2]
            else:
                vtrans = np.sqrt(pow(p[0] - self.odometry[0],2) + pow(p[1] - self.odometry[1],2))
                vrot = self.clip_rad_360(p[2] - self.odometry[2])                
                
            self.odometry = p
            print('in pose odom, vtrans, vrot, odom:', vtrans, vrot, self.odometry)
            return vtrans , vrot

        if len(p) == 7: #(xyzquat)
            pose = Pose.from_numpy(p)
            yaw, _, _ = pose.orientation.pyquaternion().yaw_pitch_roll
            # drone has Z pointing down
            yaw = -yaw
            self.odometry = [pose.position.x, pose.position.y, yaw]

            if self.latest is None:
                self.latest = pose
                return 0, 0
            else:
                diff = transform(pose, inverse(self.latest))

                vtrans = diff.position.x
                yaw, _, _ = diff.orientation.pyquaternion().yaw_pitch_roll
                vrot = -yaw
                self.latest = pose
                return vtrans / 100, vrot

        print('ERROR in memory_graph PoseOdometry')
        
            

        


class VisualOdometry(object):
    '''Visual Odometry Module.'''

    def __init__(self, key, width=160, height=120):
        self.key = key
        '''Initializes the visual odometry module.'''

        # TODO this is pretty hard-coded for 640x480 image input
        self.IMAGE_Y_SIZE = width
        self.IMAGE_X_SIZE = height
        # self.IMAGE_VTRANS_Y_RANGE = slice(270, 430)
        self.IMAGE_VTRANS_Y_RANGE = slice(height // 2, height)
        # self.IMAGE_VROT_Y_RANGE = slice(75, 240)
        self.IMAGE_VROT_Y_RANGE = slice(height // 6, height // 2)
        # self.IMAGE_ODO_X_RANGE = slice(180 + 15, 460 + 15)
        self.IMAGE_ODO_X_RANGE = slice(width // 4, width - width // 4)
        # TODO what with these numbers?
        self.VTRANS_SCALE = 100
        self.VISUAL_ODO_SHIFT_MATCH = width // 4
        self.ODO_ROT_SCALING = np.pi / 180. / 7.

        template_size = (self.IMAGE_ODO_X_RANGE.stop -
                         self.IMAGE_ODO_X_RANGE.start)
        self.old_vtrans_template = np.zeros(template_size)
        self.old_vrot_template = np.zeros(template_size)

        self.odometry = [0., 0., np.pi / 2]

    def _create_template(self, subimg):
        '''Compute the sum of columns in subimg and normalize it.

        :param subimg: a sub-image as a 2D numpy array.
        :return: the view template as a 1D numpy array.
        '''
        x_sums = np.sum(subimg, 0)
        avint = np.sum(x_sums, dtype=np.float32) / x_sums.size
        return x_sums / avint

    def __call__(self, observations, dt=None):
        '''Execute an interation of visual odometry.

        :param img: the full gray-scaled image as a 2D numpy array.
        :return: the deslocation and rotation of the image from the previous 
                 frame as a 2D tuple of floats.
        '''
        img = observations[self.key]
        if len(img.shape) == 3:
            # convert to numpy image format
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            # make grayscale
            img = np.average(img, axis=2)

        subimg = img[self.IMAGE_VTRANS_Y_RANGE, self.IMAGE_ODO_X_RANGE]
        template = self._create_template(subimg)

        # VTRANS
        offset, diff = compare_image_templates(
            template,
            self.old_vtrans_template,
            self.VISUAL_ODO_SHIFT_MATCH
        )
        vtrans = diff * self.VTRANS_SCALE

        if vtrans > 10:
            vtrans = 0

        self.old_vtrans_template = template

        # VROT
        subimg = img[self.IMAGE_VROT_Y_RANGE, self.IMAGE_ODO_X_RANGE]
        template = self._create_template(subimg)

        offset, diff = compare_image_templates(
            template,
            self.old_vrot_template,
            self.VISUAL_ODO_SHIFT_MATCH
        )
        vrot = offset * (50. / img.shape[1]) * np.pi / 180
        self.old_vrot_template = template

        # Update raw odometry
        self.odometry[2] += vrot
        self.odometry[0] += vtrans * np.cos(self.odometry[2])
        self.odometry[1] += vtrans * np.sin(self.odometry[2])

        return vtrans, vrot
