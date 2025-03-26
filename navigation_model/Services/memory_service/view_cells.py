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
try:
    import torch
finally:
    pass


import numpy as np
# import time

#from dommel_library.distributions.multivariate_normal import MultivariateNormal
# import re
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')

class TorchedViewCell(object):

    _ID = 0

    def __init__(self, cells, x_pc, y_pc, th_pc, init_decay=1.0):
        self.id = TorchedViewCell._ID
        self.cells = cells  #why???!!!!
        self.x_pc = x_pc
        self.y_pc = y_pc
        self.th_pc = th_pc
        self.decay = init_decay
        self.first = True
        self.exps = []#contains exp id
        self.to_update = False
        #self.init_local_position = local_position
        self.relevant_poses = []
        #self.template_info = {}

        TorchedViewCell._ID += 1

    @property
    def template(self):
        #return self.cells.templates[self.id, :].cpu().numpy()
        return self.cells.templates[self.id, :].detach().cpu().numpy()

    @property
    def score(self):
        #return self.cells.scores[self.id, :].cpu().numpy()
        return self.cells.scores[self.id, :].detach().cpu().numpy()

    


class TorchedViewCells(object):
    '''View Cell module.'''

    def __init__(self, key, global_decay=0.1, active_decay=1.0,
                 match_threshold=0.09, **kwargs):
        '''Initializes the View Cell module.'''
        self.cells = []
        self.templates = None
        #self.cumulated_cell_score = None
        #self.scores = None
        self.prev_cell = None
       

        self.GLOBAL_DECAY = global_decay
        self.ACTIVE_DECAY = active_decay
        self.MATCH_THRESHOLD = match_threshold
        #self.activated_cells = []

        self.key = key
        
    def load_memorised_templates(self):
        initial_size = 200

        self.templates = torch.zeros(
            (initial_size, self.cells[0].template.shape[-1]), dtype=torch.from_numpy(self.cells[0].template).dtype).to(device)
        
        for cell in self.cells:
            temp = torch.from_numpy(cell.template).to(device)
            if self.templates.shape[0] == cell.id:
                
                new_templates = torch.zeros(int(self.templates.shape[0] * 1.5), temp.shape[-1],
                                            dtype=temp.dtype).to(device)
                new_templates[:self.templates.shape[0], :] = self.templates
                self.templates = new_templates
    
            self.templates[cell.id, :] = temp
        

    def create_cell(self, template, x_pc, y_pc, th_pc):
        cell = TorchedViewCell(self, x_pc,
                               y_pc, th_pc, self.ACTIVE_DECAY)
        self.cells.append(cell)
        # add template to the array  and grow the array if required
        if self.templates is None:
            # intialize templates TODO: NOT SET A FINITE AMOUNT OF VIEWS
            initial_size = 200
            
            self.templates = torch.zeros(
                (initial_size, template.shape[-1]), dtype=template.dtype).to(device)
           # self.cumulated_cell_score = [0]*initial_size

        elif self.templates.shape[0] == cell.id:
            # we need to expand
            
                
            new_templates = torch.zeros(int(self.templates.shape[0] * 1.5), template.shape[-1],
                                        dtype=template.dtype).to(device)
            new_templates[:self.templates.shape[0], :] = self.templates
            self.templates = new_templates

            # cumulated_cell_score = [0]* len(self.cumulated_cell_score) * 1.5
            # cumulated_cell_score[:len(self.cumulated_cell_score)] = self.cumulated_cell_score.copy()
            # self.cumulated_cell_score = cumulated_cell_score
        
    
        self.templates[cell.id, :] = template
        
        return cell
    def update_prev_cell(self,exp_view_cell):
        self.prev_cell = exp_view_cell

    def _score(self, template):
        # TODO this only implements state cosine distance
    
        a = torch.matmul(self.templates, template.unsqueeze(-1))
        score = 1 - torch.abs(a / (
            torch.norm(torch.Tensor(template), dim=-1) * torch.norm(torch.Tensor(self.templates), dim=-1).unsqueeze(-1)))

        s = [[x,i] for i,x in enumerate(score)]
        print('show me the cos similiarity score', s[:TorchedViewCell._ID])
        return score

    def _compare_templates_kl_alternative_fct(self, t1, t2):
        split = t1.shape[-1] // 2
        mu1 = t1[..., 0:split]
        sigma1 = t1[..., split:]
        mu2 = t2[..., 0:split]
        sigma2 = t2[..., split:]

        # s = - KL
        sigma_ratio_squared = (sigma1 / (sigma2 + 1e-12)) ** 2
        kl = 0.5 * (
            ((mu1 - mu2) / (mu2 + 1e-12)) ** 2
            + sigma_ratio_squared
            - np.log(sigma_ratio_squared)
            - 1
        )
        return torch.mean(kl)
    
    def kl_alternative_fct(self, template):
        kl_list = []
        for id in range(TorchedViewCell._ID):
            view_cell_id_template = self.templates[id]
            kl = self._compare_templates_kl_alternative_fct(template,view_cell_id_template )
            kl_list.append([kl, id])
        print('alternative fct KL, MUST be the same as other fct', kl_list)

    def _kl_divergence_score(self,template):
        templates = self.templates[:TorchedViewCell._ID]
        templates = MultivariateNormal(templates[:,:int(templates.shape[-1]/2)], templates[:,int(templates.shape[-1]/2):])
        
        if not type(template) == type(MultivariateNormal(torch.zeros(1),torch.ones(1))):  
            template =  MultivariateNormal(template[:,:int(template.shape[-1]/2)], template[:,int(template.shape[-1]/2):])
        temp= template.unsqueeze(0)
        #print('show me the current template and templates new shapes:',temp.shape, templates.shape)
        kl = torch.distributions.kl_divergence(temp,templates)
        kl = kl / temp.shape[-1]
        test_kl = [[x,i] for i,x in enumerate(kl)]
        print('show me the kl divergence',test_kl)
        return kl

    #//def remove_views_without_exp(self):
        cells_no_exp = []
        print('How many view cells:', len(self.cells))
        for cell in self.cells:
            if len(cell.exps) == 0:
                print(cell.id,' has no exp')
                cells_no_exp.append(cell.id)
        if len(cells_no_exp)>0:
            for id in cells_no_exp:
                self.cells.pop(id)
                
                TorchedViewCell._ID-=1

            for id, cell in enumerate(self.cells):
                print('cell id', cell.id, 'will become', id)
                cell_template = self.templates[cell.id,:]
                cell.id = id
                self.templates[cell.id,:] = cell_template
    def remove_views_without_exp(self):
        print('How many view cells:', len(self.cells))
        
        # Identify indices of view cells that have no experiences.
        indices_to_remove = []
        for i, cell in enumerate(self.cells):
            if len(cell.exps) == 0:
                print(f'{cell.id} has no exp')
                indices_to_remove.append(i)
        
        # Remove cells in descending order to avoid index shifts.
        if indices_to_remove:
            for idx in sorted(indices_to_remove, reverse=True):
                self.cells.pop(idx)
                TorchedViewCell._ID -= 1
            
            # After removal, reassign IDs and update the templates accordingly.
            for new_id, cell in enumerate(self.cells):
                print(f'cell id {cell.id} will become {new_id}')
                # Use clone() to create a copy of the tensor row.
                cell_template = self.templates[cell.id, :].clone()
                cell.id = new_id
                self.templates[new_id, :] = cell_template

            
               
    def get_closest_template(self,template):
        scores = self._score(template)
               
        value, indice = torch.min(
            scores[:TorchedViewCell._ID, :], dim=0)
        #NOTE: there should be only 1 min value, unless there is been a copy of a view_cell, then we want the newly created one to avoid imp issue
        indices = (scores[:TorchedViewCell._ID, :] <= value + 0.001).nonzero().squeeze()
        
        min_score = value[-1].item()
        if len(indices.shape) == 1:
            indices = [indices]
        i = indices[-1][0].item()
        return min_score, i
    
    def __call__(self, observations, x_pc, y_pc, th_pc, **kwargs):
        '''Execute an iteration of visual template.

        :param observation: the observation as a numpy array.
        :param x_pc: index x of the current pose cell.
        :param y_pc: index y of the current pose cell.
        :param th_pc: index th of the current pose cell.
        :return: the active view cell.
        '''
        cell_bis = None
        delta_exp_above_thresold = kwargs.get('delta_exp_above_thresold', True)
        #local_position = kwargs.get('local_position', None)
        current_exp_id = kwargs.get('current_exp_id', None)

        self.remove_views_without_exp()
        # TODO vectorize decay as well?
        for cell in self.cells:
            cell.decay -= self.GLOBAL_DECAY
            if cell.decay < 0:
                cell.decay = 0

        
        #The agent holds a single belief, else Multiple models running in parallel, agent lost
        if observations is not None:
            if isinstance(observations, np.ndarray):
                template = torch.from_numpy(observations).to(device)
            else:
                template = observations
            
            if TorchedViewCell._ID :
                with torch.no_grad():

                    min_score, i = self.get_closest_template(template)
                    # kl_score = self._kl_divergence_score(template)
                    
                    print('min scores and indices', min_score, i)
                    print(' NO KL: closest looking exp ' + str(i)+ ' view match score and th ' + str(min_score) +' '+ str(self.MATCH_THRESHOLD))
                    
            if not TorchedViewCell._ID or min_score > self.MATCH_THRESHOLD:
                
                #TODO: REMOVE THIS IF, THIS IS A SIMPLIFICATION NOT ADAPTED FOR REAL WORLD
                if delta_exp_above_thresold:
                    print('creating view cell')
                    cell = self.create_cell(template, x_pc, y_pc, th_pc)
                    self.prev_cell = cell
                    return cell, None
                
                #if not far enough from prev view we replace prev cell ob by the new cell ob
                else: 
                    print(' replacing view cell', self.prev_cell.id,'template of exp', current_exp_id,', LP and GP to update')
                    self.templates[self.prev_cell.id, :] = template
                    self.prev_cell.to_update = True
                # for exp in self.prev_cell.exps:
                #     if exp.id == current_exp_id:
                #         exp.init_local_position = local_position
                #         break
                    return self.prev_cell, None
                

            elif self.prev_cell is not None and self.prev_cell.id == i :
                print('update view cell', i)
                #if we are still at the same cell, update content 
                self.templates[self.prev_cell.id, :] = template
                 
            else :
                print('selecting old view cell', i)

            cell = self.cells[i]
            cell.decay += self.ACTIVE_DECAY

            if self.prev_cell != cell:
                cell.first = False
                #We copy the cell in case we are not close looping but the view is the same.
                cell_bis = self.create_cell(self.templates[cell.id, :], cell.x_pc, cell.y_pc, cell.th_pc)
                

            self.prev_cell = cell
        elif self.prev_cell is not None: 
            # in the specific situation if we close loop observations, but dist to other exps > th, 
            # so we create a new exp with a copy of view cell
            # In such a case we need to update to the view_cell copy template (exact same template but higher id) 
            min_score, i = self.get_closest_template(self.templates[self.prev_cell.id, :])
            if i != self.prev_cell.id :
                self.prev_cell = self.cells[i]

        return self.prev_cell, cell_bis


def compare_image_templates(t1, t2, slen):
    cwl = t1.size

    mindiff = 1e10
    minoffset = 0

    for offset in range(slen + 1):
        e = (cwl - offset)

        cdiff = np.abs(t1[offset:cwl] - t2[:e])
        cdiff = np.sum(cdiff) / e

        if cdiff < mindiff:
            mindiff = cdiff
            minoffset = offset

        cdiff = np.abs(t1[:e] - t2[offset:cwl])
        cdiff = np.sum(cdiff) / e

        if cdiff < mindiff:
            mindiff = cdiff
            minoffset = -offset

    return minoffset, mindiff


'''
class GaussianLatentViewCells(ViewCells):
    #View Cell module that uses latent gaussian distributions as template

    def __init__(self, key, global_decay=0.1, active_decay=1.0,
                 match_threshold=1.0, **kwargs):
        ViewCells.__init__(self, global_decay, active_decay,
                           match_threshold, **kwargs)
        self.key = key

    def _create_template(self, observations):
        return observations[self.key]

    def _compare_templates(self, t1, t2):
        split = t1.shape[-1] // 2
        mu1 = t1[..., 0:split]
        sigma1 = t1[..., split:]
        mu2 = t2[..., 0:split]
        sigma2 = t2[..., split:]

        # s = - KL
        sigma_ratio_squared = (sigma1 / (sigma2 + 1e-12)) ** 2
        kl = 0.5 * (
            ((mu1 - mu2) / (mu2 + 1e-12)) ** 2
            + sigma_ratio_squared
            - np.log(sigma_ratio_squared)
            - 1
        )
        return min(np.mean(kl), 10.0)
'''