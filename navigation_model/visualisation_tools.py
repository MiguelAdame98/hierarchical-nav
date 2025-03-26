import matplotlib.pyplot as plt
import numpy as np
import torch
from dommel_library.datastructs import TensorDict
from dommel_library.modules.visualize import (vis_image, vis_images)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

#=============================== IMAGES METHODS ===========================================

def visualise_image(images: torch.Tensor, title: str = '', fig_id: int = np.random.randint(1)) -> np.ndarray:
    images_list= convert_tensor_to_matplolib_list(images)
    plt.figure(fig_id)
    plt.imshow(images_list)
    plt.title(title)
    plt.axis('off')
    return images_list

def convert_tensor_to_matplolib_list(images: torch.Tensor) -> list:
    images_list = vis_images(images,show=False, fmt="numpy").tolist()#.cpu().detach().numpy()
    #image_array = np.transpose(image_array, (1, 2, 0)).tolist()
    return images_list

def convert_tensor_image_to_array(image:torch.Tensor)->np.ndarray:
    img = vis_image(image, show=False, fmt="torch").cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img*255
    return img

#=============================== POLICY VISU =============================================

def transform_policy_from_hot_encoded_to_str(policy:list) -> list:
        ''' given a policy of hot encoded actions transform it 
        to the corresponding string direction
        '''
        str_policy = []
        for action in policy:
            if action[0] == 1:
                str_policy.append('F')
            elif action[1] == 1:
                str_policy.append('R')
            elif action[2] == 1:
                str_policy.append('L')
            else:
                str_policy.append('X')
        return str_policy

#=============================== VIDEO METHODS ===========================================

def record_video_frames(data:dict, env_definition:dict, agent_lost:bool, visited_rooms:list, memory_map_data:dict, step_count:int) -> np.ndarray:
    ''' we create a full frame ready to be added to a video or displayed'''
    fig = plt.figure(3, figsize = (12, 6))
    plt.clf()
    #s = plt.GridSpec(5, 6, wspace =0.3, hspace = 0.3) #
    s = fig.add_gridspec(4, 6, width_ratios = [3, 3, 3, 3, 3,3], height_ratios = [3, 3, 3 , 3],wspace =0.8, hspace = 0.4)

    #-- WORLD--#
    ax1 = fig.add_subplot(s[:2, :2])
    ax1 = get_image_plot(ax1, data['env_image'], 'World')

    #-- GT OBSERVATION--#
    ax2 = fig.add_subplot(s[:1, 2:3])
    ax2 = get_image_plot(ax2, data['ground_truth_ob'], 'GT OB')
   
     #-- PREDICTED OBSERVATIONS--#
    if 'image_predicted' in data and data['image_predicted']is not None:
        ax3 = fig.add_subplot(s[:2, 3])
        ax3 = get_image_plot(ax3, data['image_predicted'], 'Predicted ob')

   #-- INFO GAIN (NOT IMP) ---#
    ax2_bis = fig.add_subplot(s[1:2, 2:3])
    if "info_gain" in data and data["info_gain"] is not None:
        info_gain_value = data["info_gain"]
        # Check if it's already a 3D array:
        if hasattr(info_gain_value, "shape") and len(info_gain_value.shape) == 3:
            img = vis_image(info_gain_value, show=False, fmt="numpy")
            ax2_bis = get_image_plot(ax2_bis, img, "Place_info_gain")
        else:
            # info_gain is likely a float (or something not a 3D image),
            # so either skip plotting or convert to a mock image, e.g.:
            
            # Skip plotting:
            pass
            
            # OR turn it into a small image:
            import numpy as np
            mock_img = np.full((50, 50, 3), info_gain_value, dtype=np.float32)
            img = vis_image(mock_img, show=False, fmt="numpy")
            ax2_bis = get_image_plot(ax2_bis, img, "Place_info_gain")


    #-- MAPS OF VISITED ROOMS --#
    ax4 = fig.add_subplot(s[2:4, :2])
    ax4 = plot_visited_rooms(ax4,visited_rooms, env_definition)

    #-- DESIRED OBJECTIVE --#
    if 'goal' in data:
        ax5 = fig.add_subplot(s[2:4, 2:4])
        if len(data['goal'].shape)>3:
            data['goal'] = torch.mean(data['goal'], dim=list(range(len(data['goal'].shape)-3)))
        print("data['goal'] shape", data['goal'].shape)
        img =  vis_image(data['goal'], show=False, fmt="numpy")
        ax5 = get_image_plot(ax5, img, 'Imagined desired objective')

    #-- PRED/OB MSE ---#
    ax6 = fig.add_subplot(s[:2, 4])
    ax6 = plot_MSE_bar(ax6, data['mse'], agent_lost)

    ax7 = fig.add_subplot(s[2:4, 4:])
    ax7= plot_memory_map(ax7, memory_map_data)

    ax8 = fig.add_subplot(s[:2, 5])
    ax8 = print_positions(ax8, data['GP'], data['pose'], step_count)
    

    fig = plt.gcf()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    #print(f"Canvas size: {canvas_width}x{canvas_height}")
    s, (width, height) = fig.canvas.print_to_buffer()
    #print(f"Buffer size: {len(s)} Expected size: {width, height, 4}")
    width, height = 1200,600
    buf = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    # we don't need the alpha channel
    buf = buf[:, :, 0:3]

    #plt.show()
    # if self.close_plot:
    plt.close(fig)

    
    return buf

def get_image_plot(ax, data,title):
    ax.imshow(data)
    ax.set_title(title)
    plt.axis('off')
    return ax

def plot_MSE_bar(ax, mse_err, agent_lost):
    #-- PRED/OB MSE ---#
    try:
        #print('mse data length', len(data['mse']), 'last mse', mse_err)
        if mse_err <0.5:
            color = 'blue'
        else:
            color = 'red'
        
        plt.bar('MSE error', mse_err , color = color, width = 0.1)
    except KeyError as e:
        ax.text(-0.3,0.9, e, fontsize=10, color='black')

    ax.set_ylim(0,1)
    if agent_lost:
        back_color = '#ff8970'
    else:
        back_color = '0.8'
    ax.set_facecolor(color=back_color)
    plt.grid(axis='y')
    plt.title('mse ob/expectation')

def plot_visited_rooms(ax, visited_rooms, env_definition):
    n_row = env_definition['n_row']
    n_col = env_definition['n_col']

    # Create a grid of the specified size with all white tiles
    grid = np.ones((n_row, n_col))  

    # Assign colors based on visited rooms
    for i in range(n_col):
        for j in range(n_row):
            if (i, j) in visited_rooms:
                position_in_list = visited_rooms.index((i, j))
                color_value = position_in_list / len(visited_rooms)
                grid[j][i] = color_value

    # Create the plot
    im = ax.imshow(grid, cmap='gist_heat', vmin=0, vmax=1)
    ax.set_xticks(np.arange(n_col))
    ax.set_yticks(np.arange(n_row))
    # Create a divider for existing axes instance
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Create a colorbar
    cbar = plt.colorbar(im, cax=cax, orientation='vertical', label='Visited Rooms')

    # Set custom tick labels
    cbar.set_ticks([0, 0.9, 1])
    cbar.set_ticklabels(['oldest', 'newest', 'unknown'])

    # Set title for the subplot
    ax.set_title('Rooms ordered by discovery')

    return ax

def plot_memory_map(ax, memory_map_data):
    plt.title('Exp Map')
    plt.grid()
    
    #if nothing to report in map, just pass
    if memory_map_data['current_exp_id'] < 0:
        return ax
    
    exps_positions = np.vstack(memory_map_data['exps_GP'])
    ax.scatter(exps_positions[:,0], exps_positions[:,1], c=memory_map_data['exps_decay'], cmap="viridis")
    
    if len(memory_map_data['ghost_exps_GP']) > 0:
        ghost_exps_positions = np.vstack(memory_map_data['ghost_exps_GP'])
        ax.scatter(ghost_exps_positions[:,0], ghost_exps_positions[:,1], c='m', linewidths=0.5, alpha=0.5)
                
        if len(memory_map_data['ghost_exps_link']) > 0:
            ghost_exps_links = np.vstack(memory_map_data['ghost_exps_link'])
            ax.plot(ghost_exps_links[:,0], ghost_exps_links[:,1], c='m', linestyle='dashed', alpha=0.2)
    
    if len(memory_map_data['exps_links']) > 0:
        exps_links = np.vstack(memory_map_data['exps_links'])
        ax.plot(exps_links[:,0], exps_links[:,1], 'grey')

    ax.plot(memory_map_data['current_exp_GP'][0],
            memory_map_data['current_exp_GP'][1], 'r', marker='x')

    s0 = str(memory_map_data['current_exp_id'])
    ax.text(memory_map_data['current_exp_GP'][0], memory_map_data['current_exp_GP'][1], s0, fontsize=10, color='red')

    return ax

def print_positions(ax, GP, pose, step_count):
    current_gp = [float(x) for x in np.around(np.array(GP), 2)]
    s1 = 'Global Position [x,y,th]: ' 
    ax.text(-0.3, 0.9, s1, fontsize=10, color='black')
    ax.text(0, 0.8, str(current_gp), fontsize=10, color='black')
    
    s3 = 'Local Position [x y th]: ' 
    ax.text(-0.3,0.6, s3, fontsize=10, color='black')
    ax.text(0, 0.5, str(pose), fontsize=10, color='black')
    ax.set_axis_off()
    s0 = 'Step: ' + str(step_count) 
    ax.text(-0.3,1.1, s0, fontsize=10, color='black')
    return ax

#=============================== OTHER METHODS ===========================================
def plot_pose_cube(memory_graph:object):
    """ plot CAN pose cell gradual motion"""
    plt.figure(3)
    plt.title('Pose cube')
    
    ax = plt.axes(projection="3d")
    x, y, th = memory_graph.pc
    dim_xy = memory_graph.pose_cells.DIM_XY
    dim_th = memory_graph.pose_cells.DIM_TH
    ax.plot3D(x, y, th,'x')
    ax.plot3D([0, dim_xy], [y[-1], y[-1]], [th[-1], th[-1]], 'k')
    ax.plot3D([x[-1], x[-1]], [0, dim_xy], [th[-1], th[-1]], 'k')
    ax.plot3D([x[-1], x[-1]], [y[-1], y[-1]], [0, dim_th], 'k')
    ax.plot3D([x[-1]], [y[-1]], [th[-1]], 'mo')
    ax.grid()
    ax.axis([0, dim_xy, 0, dim_xy])
    ax.set_zlim(0, dim_th)
    plt.show()

def plot_room(allo_model:object, door_poses:list, step:int= None):
    '''
    plot the imagined room of the allocentric model
    '''
    # print(exps, len(exps))
    # print(type(exps[0]['observation']), type(exps[0]['observation'][0]))
    #place = place.unsqueeze(0).unsqueeze(0).repeat(5,1,1) #MultivariateNormal(torch.from_numpy(exps[rat_exps_id]['observation'])).unsqueeze(0).unsqueeze(0).repeat(5,1,1)
    #print('observation door poses 0', exps[rat_exps_id]['observation_door_poses'][0], type(exps[rat_exps_id]['observation_door_poses'][0]))
    dps = door_poses.copy()
        
    poses = []
    for door_p in dps:
        if   door_p[2] == 2:
            door_p[0] += 4
            door_p[1] += 1
        elif door_p[2] == 0:
            door_p[0] -= 2
        elif door_p[2] == 1:
            door_p[1] -= 2
        elif door_p[2] == 3:
            door_p[1] += 2

        for angle in range(4):
            door_p[2]= angle
            poses.append(door_p.copy())
    
    pred_pose = TensorDict({'pose_query': torch.Tensor(poses).unsqueeze(0).repeat(5,1,1)})
    #print('pred poses', pred_pose, pred_pose['pose_query'].shape)
    pred_step = allo_model.model.forward(pred_pose, place=None, reconstruct=True)

    #--- Show 1 mean image
    img_pred = torch.mean(pred_step['image_predicted'], dim=0)

    view_points = []
    for view in range(img_pred.shape[0]):
        image_predicted_data = vis_image(img_pred[view],show=False, fmt="torch").cpu().detach().numpy()
        image_predicted_data = np.transpose(image_predicted_data, (1, 2, 0)) * 255  # from normalized data 0 - 1 to 255 img
        image_predicted_data = image_predicted_data.astype(np.uint8)
        
        image_predicted_data = Image.fromarray(image_predicted_data, "RGB")
        view_points.append(image_predicted_data)
    
    background = Image.new("RGB", (180, 180), (255, 255, 255))
    middle_background = round(background.size[0]/2)

    for i in range(0,len(view_points), 4):
        if i == 0:
            shift = [0,0]
        else:
            #start = [middle_background,middle_background] #it's the middle of first image in background (so 3.5tiles*8px forward agent)
            shift = np.array([poses[0][0] - poses[i][0], poses[0][1] + poses[i][1]])*8
            print(poses[0],poses[i], shift)
            
        view_points[i].putalpha(180)
        img_look_forward = view_points[i]
        img_look_right = view_points[i+1].rotate(-90)
        img_look_behind = view_points[i+2].rotate(180)
        img_look_left = view_points[i+3].rotate(90)

        middle_img = round(img_look_forward.size[0]/2)
        H_agent_in_img = middle_img - (3*8)
        
        
        # starting at coordinates (x,y) upper/left agle is (0,0)
        background.paste(img_look_forward, (middle_background+shift[1]-middle_img, middle_background+shift[0]-middle_img), mask=img_look_forward)
        background.paste(img_look_right, (middle_background+shift[1]-H_agent_in_img, middle_background+shift[0]-H_agent_in_img), mask=img_look_forward)
        background.paste(img_look_left, (middle_background+shift[1]-img_look_left.size[0]+4, middle_background+shift[0]-H_agent_in_img), mask=img_look_forward)
        background.paste(img_look_behind, (middle_background+shift[1]-middle_img, middle_background+shift[0]+middle_img-8), mask=img_look_forward)
                
    background.show()
    background.save('imagination_results/imagined_room/imagined_room_step_'+str(step) +'.png')
    print('room saved as: imagination_results/imagined_room/imagined_room_step_'+str(step)+'.png')
