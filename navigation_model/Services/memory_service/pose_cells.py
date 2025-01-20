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

import itertools
import numpy as np
import torch

def create_pc_weights(dim, var):
    dim_center = int(np.floor(dim / 2.))

    weight = np.zeros([dim, dim, dim])
    for x, y, z in itertools.product(range(dim), range(dim), range(dim)):
        dx = -(x - dim_center)**2
        dy = -(y - dim_center)**2
        dz = -(z - dim_center)**2
        weight[x, y, z] = 1.0 / (var * np.sqrt(2 * np.pi)) * \
            np.exp((dx + dy + dz) / (2. * var**2))

    weight = weight / np.sum(weight)
    return weight


class PoseCells(object):
    '''Pose Cell module.'''

    def __init__(self, dim_xy=61, dim_th=36, inject_energy=0.1,
                 global_inhibition=0.00002, torched=False, device="cpu", posecell_vtrans_scalling = 1./10. , **kwargs):
        '''Initializes the Pose Cell module.'''

        self.DIM_XY = dim_xy
        self.DIM_TH = dim_th
        self.W_E_VAR = 1
        self.W_E_DIM = 7
        self.W_I_VAR = 2
        self.W_I_DIM = 5
        self.INJECT_ENERGY = inject_energy
        self.GLOBAL_INHIB = global_inhibition
        self.CELLS_TO_AVG = 3
        self.POSECELL_VTRANS_SCALING = posecell_vtrans_scalling

        self.W_EXCITE = create_pc_weights(self.W_E_DIM, self.W_E_VAR)
        self.W_INHIB = create_pc_weights(self.W_I_DIM, self.W_I_VAR)

        self.W_E_DIM_HALF = int(np.floor(self.W_E_DIM / 2.))
        self.W_I_DIM_HALF = int(np.floor(self.W_I_DIM / 2.))

        self.torched = False
        self.device = device
        if torched:
            self.torched = True
            self.W_EXCITE = torch.from_numpy(self.W_EXCITE).unsqueeze(0).unsqueeze(0).float()
            self.W_EXCITE = torch.nn.Parameter(self.W_EXCITE, requires_grad=False)
            self.W_INHIB = torch.from_numpy(self.W_INHIB).unsqueeze(0).unsqueeze(0).float()
            self.W_INHIB = torch.nn.Parameter(self.W_INHIB, requires_grad=False)
            self.excite_op = torch.nn.Conv3d(1, 1, self.W_E_DIM, 1, self.W_E_DIM_HALF, 1, 1, False, "circular")
            self.excite_op.weight = self.W_EXCITE
            self.excite_op = self.excite_op.to(device)
            self.inhibit_op = torch.nn.Conv3d(1, 1, self.W_I_DIM, 1, self.W_I_DIM_HALF, 1, 1, False, "circular")
            self.inhibit_op.weight = self.W_INHIB
            self.inhibit_op = self.inhibit_op.to(device)

        self.C_SIZE_TH = (2. * np.pi) / self.DIM_TH
        self.E_XY_WRAP = list(range(self.DIM_XY - self.W_E_DIM_HALF, self.DIM_XY)) + \
            list(range(self.DIM_XY)) + list(range(self.W_E_DIM_HALF))
        self.E_TH_WRAP = list(range(self.DIM_TH - self.W_E_DIM_HALF, self.DIM_TH)) + \
            list(range(self.DIM_TH)) + list(range(self.W_E_DIM_HALF))
        self.I_XY_WRAP = list(range(self.DIM_XY - self.W_I_DIM_HALF, self.DIM_XY)) + \
            list(range(self.DIM_XY)) + list(range(self.W_I_DIM_HALF))
        self.I_TH_WRAP = list(range(self.DIM_TH - self.W_I_DIM_HALF, self.DIM_TH)) + \
            list(range(self.DIM_TH)) + list(range(self.W_I_DIM_HALF))
        self.XY_SUM_SIN_LOOKUP = np.sin(np.multiply(
            list(range(1, self.DIM_XY + 1)), (2 * np.pi) / self.DIM_XY))
        self.XY_SUM_COS_LOOKUP = np.cos(np.multiply(
            list(range(1, self.DIM_XY + 1)), (2 * np.pi) / self.DIM_XY))
        self.TH_SUM_SIN_LOOKUP = np.sin(np.multiply(
            list(range(1, self.DIM_TH + 1)), (2 * np.pi) / self.DIM_TH))
        self.TH_SUM_COS_LOOKUP = np.cos(np.multiply(
            list(range(1, self.DIM_TH + 1)), (2 * np.pi) / self.DIM_TH))
        self.AVG_XY_WRAP = list(range(self.DIM_XY - self.CELLS_TO_AVG, self.DIM_XY)) + \
            list(range(self.DIM_XY)) + list(range(self.CELLS_TO_AVG))
        self.AVG_TH_WRAP = list(range(self.DIM_TH - self.CELLS_TO_AVG, self.DIM_TH)) + \
            list(range(self.DIM_TH)) + list(range(self.CELLS_TO_AVG))

        self.cells = np.zeros([self.DIM_XY, self.DIM_XY, self.DIM_TH])
        self.active = a, b, c = [self.DIM_XY //
                                 2, self.DIM_XY // 2, self.DIM_TH // 2]
        self.cells[a, b, c] = 1

    def reset(self, x_pc, y_pc, th_pc):
        a = np.min([np.max([int(np.floor(x_pc)), 1]), self.DIM_XY])
        b = np.min([np.max([int(np.floor(y_pc)), 1]), self.DIM_XY])
        c = np.min([np.max([int(np.floor(th_pc)), 1]), self.DIM_TH])

        self.cells = np.zeros([self.DIM_XY, self.DIM_XY, self.DIM_TH])
        self.cells[a, b, c] = 1
        self.active = a, b, c

    def compute_activity_matrix(self, xywrap, thwrap, wdim, pcw, excite=True):
        '''Compute the activation of pose cells.'''
        if self.torched:
            cell_tensor = torch.from_numpy(self.cells).unsqueeze(0).unsqueeze(0).float()
            cell_tensor = cell_tensor.to(self.device)
            with torch.no_grad():
                ret = None
                if excite:
                    ret = self.excite_op(cell_tensor)
                else:
                    ret = self.inhibit_op(cell_tensor)
            return ret.cpu().numpy()[0,0]
        else:
             # The goal is to return an update matrix that can be added/subtracted
            # from the posecell matrix
            pca_new = np.zeros([self.DIM_XY, self.DIM_XY, self.DIM_TH])
            # for nonzero posecell values
            indices = np.nonzero(self.cells)
            for i, j, k in zip(*indices):
                pca_new[np.ix_(xywrap[i:i + wdim],
                            xywrap[j:j + wdim],
                            thwrap[k:k + wdim])] += self.cells[i, j, k] * pcw
            return pca_new

    def get_pc_max(self, xywrap, thwrap):
        '''Find the x, y, th center of the activity in the network.'''

        x, y, z = np.unravel_index(np.argmax(self.cells), self.cells.shape)
        if self.torched:
            return x, y, z
        z_posecells = np.zeros([self.DIM_XY, self.DIM_XY, self.DIM_TH])

        zval = self.cells[np.ix_(
            xywrap[x:x + self.CELLS_TO_AVG * 2],
            xywrap[y:y + self.CELLS_TO_AVG * 2],
            thwrap[z:z + self.CELLS_TO_AVG * 2]
        )]
        z_posecells[np.ix_(
            self.AVG_XY_WRAP[x:x + self.CELLS_TO_AVG * 2],
            self.AVG_XY_WRAP[y:y + self.CELLS_TO_AVG * 2],
            self.AVG_TH_WRAP[z:z + self.CELLS_TO_AVG * 2]
        )] = zval

        # get the sums for each axis
        x_sums = np.sum(np.sum(z_posecells, 2), 1)
        y_sums = np.sum(np.sum(z_posecells, 2), 0)
        th_sums = np.sum(np.sum(z_posecells, 1), 0)
        th_sums = th_sums[:]

        # now find the (x, y, th) using population vector decoding to handle
        # the wrap around
        x = (np.arctan2(np.sum(self.XY_SUM_SIN_LOOKUP * x_sums),
                        np.sum(self.XY_SUM_COS_LOOKUP * x_sums)) *
             self.DIM_XY / (2 * np.pi)) % (self.DIM_XY)

        y = (np.arctan2(np.sum(self.XY_SUM_SIN_LOOKUP * y_sums),
                        np.sum(self.XY_SUM_COS_LOOKUP * y_sums)) *
             self.DIM_XY / (2 * np.pi)) % (self.DIM_XY)

        th = (np.arctan2(np.sum(self.TH_SUM_SIN_LOOKUP * th_sums),
                         np.sum(self.TH_SUM_COS_LOOKUP * th_sums)) *
              self.DIM_TH / (2 * np.pi)) % (self.DIM_TH)

        # print x, y, th
        return (x, y, th)

    def __call__(self, view_cell, vtrans, vrot):
        '''Execute an interation of pose cells.

        :param view_cell: the last most activated view cell.
        :param vtrans: the translation of the robot given by odometry.
        :param vrot: the rotation of the robot given by odometry.
        :return: a 3D-tuple with the (x, y, th) index of most active pose cell.
        '''
        #print('in posecell,trans and rot', vtrans, vrot)
        vtrans = vtrans * self.POSECELL_VTRANS_SCALING
        #print('in posecell,trans scalled', vtrans)

        # if this isn't a new vt then add the energy at its associated posecell
        # location
        if not view_cell.first:
            act_x = np.min(
                [np.max([int(np.floor(view_cell.x_pc)), 1]), self.DIM_XY])
            act_y = np.min(
                [np.max([int(np.floor(view_cell.y_pc)), 1]), self.DIM_XY])
            act_th = np.min(
                [np.max([int(np.floor(view_cell.th_pc)), 1]), self.DIM_TH])

            # print [act_x, act_y, act_th]
            # this decays the amount of energy that is injected at the vt's
            # posecell location
            # this is important as the posecell Posecells will errounously snap
            # for bad vt matches that occur over long periods (eg a bad matches that
            # occur while the agent is stationary). This means that multiple vt's
            # need to be recognised for a snap to happen
            energy = self.INJECT_ENERGY * \
                (1. / 30.) * (30 - np.exp(1.2 * np.min([view_cell.decay, 100])))
            if energy > 0:
                self.cells[act_x, act_y, act_th] += energy
        #===============================

        # local excitation - self.le = PC elements * PC weights
        self.cells = self.compute_activity_matrix(self.E_XY_WRAP,
                                                  self.E_TH_WRAP,
                                                  self.W_E_DIM,
                                                  self.W_EXCITE,
                                                  excite=True)
        # print np.max(self.cells)
        # raw_input()

        # local inhibition - self.li = self.le - self.le elements * PC weights
        self.cells = self.cells - self.compute_activity_matrix(self.I_XY_WRAP,
                                                               self.I_TH_WRAP,
                                                               self.W_I_DIM,
                                                               self.W_INHIB,
                                                               excite=False)

        # local global inhibition - self.gi = self.li elements - inhibition
        self.cells[self.cells < self.GLOBAL_INHIB] = 0
        self.cells[self.cells >= self.GLOBAL_INHIB] -= self.GLOBAL_INHIB

        # normalization
        total = np.sum(self.cells)
        self.cells = self.cells / total

        # Path Integration
        # vtrans affects xy direction
        # shift in each th given by the th
        for dir_pc in range(self.DIM_TH):
            direction = np.float64(dir_pc - 1) * self.C_SIZE_TH
            # N,E,S,W are straightforward
            if (direction == 0):
                self.cells[:, :, dir_pc] = \
                    self.cells[:, :, dir_pc] * (1.0 - vtrans) + \
                    np.roll(self.cells[:, :, dir_pc], 1, 1) * vtrans

            elif direction == np.pi / 2:
                self.cells[:, :, dir_pc] = \
                    self.cells[:, :, dir_pc] * (1.0 - vtrans) + \
                    np.roll(self.cells[:, :, dir_pc], 1, 0) * vtrans

            elif direction == np.pi:
                self.cells[:, :, dir_pc] = \
                    self.cells[:, :, dir_pc] * (1.0 - vtrans) + \
                    np.roll(self.cells[:, :, dir_pc], -1, 1) * vtrans

            elif direction == 3 * np.pi / 2:
                self.cells[:, :, dir_pc] = \
                    self.cells[:, :, dir_pc] * (1.0 - vtrans) + \
                    np.roll(self.cells[:, :, dir_pc], -1, 0) * vtrans

            else:
                pca90 = np.rot90(self.cells[:, :, dir_pc],
                                 int(np.floor(direction * 2 / np.pi)))
                dir90 = direction - \
                    int(np.floor(direction * 2 / np.pi)) * np.pi / 2

                # extend the Posecells one unit in each direction (max supported at the moment)
                # work out the weight contribution to the NE cell from the SW, NW, SE cells
                # given vtrans and the direction
                # weight_sw = v * cos(th) * v * sin(th)
                # weight_se = (1 - v * cos(th)) * v * sin(th)
                # weight_nw = (1 - v * sin(th)) * v * sin(th)
                # weight_ne = 1 - weight_sw - weight_se - weight_nw
                # think in terms of NE divided into 4 rectangles with the sides
                # given by vtrans and the angle
                pca_new = np.zeros([self.DIM_XY + 2, self.DIM_XY + 2])
                pca_new[1:-1, 1:-1] = pca90

                weight_sw = (vtrans**2) * np.cos(dir90) * np.sin(dir90)
                weight_se = vtrans * np.sin(dir90) - \
                    (vtrans**2) * np.cos(dir90) * np.sin(dir90)
                weight_nw = vtrans * np.cos(dir90) - \
                    (vtrans**2) * np.cos(dir90) * np.sin(dir90)
                weight_ne = 1.0 - weight_sw - weight_se - weight_nw

                pca_new = pca_new * weight_ne + \
                    np.roll(pca_new, 1, 1) * weight_nw + \
                    np.roll(pca_new, 1, 0) * weight_se + \
                    np.roll(np.roll(pca_new, 1, 1), 1, 0) * weight_sw

                pca90 = pca_new[1:-1, 1:-1]
                pca90[1:, 0] = pca90[1:, 0] + pca_new[2:-1, -1]
                pca90[1, 1:] = pca90[1, 1:] + pca_new[-1, 2:-1]
                pca90[0, 0] = pca90[0, 0] + pca_new[-1, -1]

                # unrotate the pose cell xy layer
                self.cells[:, :, dir_pc] = np.rot90(pca90,
                                                    4 - int(np.floor(direction * 2 / np.pi)))

        # Path Integration - Theta
        # Shift the pose cells +/- theta given by vrot
        if vrot != 0:
            weight = (np.abs(vrot) / self.C_SIZE_TH) % 1
            if weight == 0:
                weight = 1.0

            shift1 = int(np.sign(vrot) *
                         int(np.floor(abs(vrot) / self.C_SIZE_TH)))
            shift2 = int(np.sign(vrot) *
                         int(np.ceil(abs(vrot) / self.C_SIZE_TH)))
            self.cells = np.roll(self.cells, shift1, 2) * (1.0 - weight) + \
                np.roll(self.cells, shift2, 2) * (weight)

        self.active = self.get_pc_max(self.AVG_XY_WRAP, self.AVG_TH_WRAP)

        print('in posecell,active cell',self.active)
        return self.active
