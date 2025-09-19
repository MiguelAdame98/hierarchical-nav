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
import numpy as np

def _wrap_rad(a: float) -> float:
    # [-pi, pi]
    a = (a + np.pi) % (2*np.pi) - np.pi
    return a

def _encode_heading_to_rad(h):
    """
    Accepts:
      - discrete headings {0,1,2,3}  (MiniGrid: N,E,S,W or your convention)
      - radians already (float)
    Returns radians in [0, 2π).
    """
    try:
        # If it's a numpy scalar or python number:
        hv = float(h)
        # If it's one of the discrete quarter turns, map k -> k*(π/2)
        if hv in (0.0, 1.0, 2.0, 3.0) and float(hv).is_integer():
            return (np.pi / 2.0) * int(hv)
        # Else assume it's already radians; normalize to [0, 2π)
        return hv % (2*np.pi)
    except Exception:
        # If pose carries already-radian np.float64 etc., just try modulo
        return h % (2*np.pi)

class MinigridActionPoseOdometry:
    """
    Hybrid odometry for 3-action MiniGrid (F,L,R) that uses the *observed pose*
    as ground-truth to guard action-based odometry.

    Interface:
      __call__(action_ob, dt=1.0, pose_ob=None) -> (vtrans, vrot)
      .odometry -> last observed (x, y, th_rad)
    """

    def __init__(self, cell_step: float = 1.0, left_is_ccw: bool = True):
        self.cell_step = float(cell_step)
        self.left_is_ccw = bool(left_is_ccw)
        self.odometry = [0.0, 0.0, 0.0]   # (x, y, th_rad)
        self._prev_pose = None            # last observed (x, y, th_rad)

    def _parse_action(self, a):
        """
        Accept either:
          - int in {0,1,2}  (e.g., 0=F, 1=L, 2=R)  <-- adapt if your mapping differs
          - one-hot [F,R,L] or [F,L,R]
        Returns ('F' | 'L' | 'R', one_hot_tuple)
        """
        # Heuristics:
        if isinstance(a, (list, tuple, np.ndarray)):
            arr = np.asarray(a).astype(float).ravel()
            # try [F, R, L]
            if arr.size >= 3:
                f, r, l = arr[0], arr[1], arr[2]
                if f > 0.5 and r < 0.5 and l < 0.5: return 'F', (1,0,0)
                if r > 0.5 and f < 0.5 and l < 0.5: return 'R', (0,1,0)
                if l > 0.5 and f < 0.5 and r < 0.5: return 'L', (0,0,1)
        else:
            # integer coding
            code = int(a)
            if code == 0: return 'F', (1,0,0)
            if code == 1: return 'L', (0,0,1)
            if code == 2: return 'R', (0,1,0)
        # fallback
        return 'F', (1,0,0)

    def __call__(self, action_ob, dt: float = 1.0, pose_ob=None):
        """
        Produce outcome-aware (vtrans, vrot).
        If pose_ob is provided as (x, y, th) with th either {0..3} or radians,
        we use it to decide whether forward actually happened and to measure Δθ.
        """
        # Parse action intention
        act, _ = self._parse_action(action_ob)

        # If we don't have a pose observation, fall back to intent-only
        if pose_ob is None:
            if act == 'F':
                return self.cell_step, 0.0
            elif act == 'L':
                return 0.0, ( np.pi/2 if self.left_is_ccw else -np.pi/2 )
            else:  # 'R'
                return 0.0, ( -np.pi/2 if self.left_is_ccw else  np.pi/2 )

        # Normalize observed pose to floats
        x_obs = float(pose_ob[0]); y_obs = float(pose_ob[1]); th_obs = _encode_heading_to_rad(pose_ob[2])

        if self._prev_pose is None:
            # Bootstrap
            self._prev_pose = (x_obs, y_obs, th_obs)
            self.odometry = [x_obs, y_obs, th_obs]
            return 0.0, 0.0

        x_prev, y_prev, th_prev = self._prev_pose
        dx = x_obs - x_prev
        dy = y_obs - y_prev
        # "Did we move a grid cell?"
        moved_ok = (abs(dx) + abs(dy)) >= (0.5 * self.cell_step)  # robust to tiny noise

        # Observed rotation
        dth = _wrap_rad(th_obs - th_prev)

        # Outcome-aware outputs
        if act == 'F':
            vtrans = self.cell_step if moved_ok else 0.0
            vrot   = 0.0  # forward shouldn't rotate in MiniGrid; ignore small dth noise
        elif act == 'L':
            vtrans = 0.0
            # prefer observed ±π/2 if clean; otherwise enforce the nominal quarter-turn
            vrot   = dth if abs(abs(dth) - (np.pi/2)) < np.deg2rad(10) else ( np.pi/2 if self.left_is_ccw else -np.pi/2 )
        else:  # 'R'
            vtrans = 0.0
            vrot   = dth if abs(abs(dth) - (np.pi/2)) < np.deg2rad(10) else ( -np.pi/2 if self.left_is_ccw else  np.pi/2 )

        # Update internal odom pose with *observed* pose (not integrated)
        self._prev_pose = (x_obs, y_obs, th_obs)
        self.odometry = [x_obs, y_obs, th_obs]

        return float(vtrans), float(vrot)

    # Optional utility you might already rely on elsewhere:
    def position_applying_motion(self, pose_xyz, action_one_hot):
        """
        Apply a commanded motion on a hypothetical pose (used e.g. for ghost nodes).
        This is intent-based (not outcome guarded) by design.
        """
        x, y, th = float(pose_xyz[0]), float(pose_xyz[1]), _encode_heading_to_rad(pose_xyz[2])
        F, R, L = int(action_one_hot[0]), int(action_one_hot[1]), int(action_one_hot[2])
        if L: th = (th + np.pi/2) % (2*np.pi)
        if R: th = (th - np.pi/2) % (2*np.pi)
        if F:
            x += self.cell_step * np.cos(th)
            y += self.cell_step * np.sin(th)
        return [x, y, th]

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
            print('we are inside the odometry pose', p)
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
