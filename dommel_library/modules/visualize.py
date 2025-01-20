import os
import torch
import imageio
import math
import numpy as np
from pyquaternion import Quaternion

from matplotlib import cm
import matplotlib as mpl

from dommel_library.datastructs import TensorDict
from dommel_library.distributions.multivariate_normal import InnerMultivariateNormal

if os.environ.get("MPL_BACKEND", "") != "":
    mpl.use(os.environ.get("MPL_BACKEND", ""))
elif os.environ.get("DISPLAY", "") == "":
    print("No display found. Using non-interactive Agg backend.")
    mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402


default_vis_mapping = {
    "image": ["image", "camera"],
    "intensity": ["intensity", "depth", "range_doppler", "range_azimuth"],
    "lidar": ["lidar", "laserscan"],
    "twist": ["twist", "odom"],
    "vector": ["vector"],
    "pose": ["pose"],
    "position": ["position"],
    "radar": ["radar"],
    "rpyt": ["rpyt"]
}


def visualize(observation, keys=None, fmt="torch", show=False, **kwargs):
    """ Visualize a dictionary of observations.

    Observations are provided as a dictionary with string keys
    and tensor (or numpy array) values.

    :param keys: the keys of the observation to visualize (default all)
    :param fmt: the output format of the image ("torch" or "numpy")
    (default "torch")
    :param show: whether to immediately show the image with matplotlib
    (default False)
    :param **kwargs: additional parameters
    :returns: an image tensor or numpy array.
    """
    if keys is not None:
        to_visualize = TensorDict({key: observation[key] for key in keys})
    else:
        to_visualize = observation

    vis_mapping = kwargs.get("vis_mapping", {})
    for k, v in default_vis_mapping.items():
        if k in vis_mapping:
            vis_mapping[k] = list(vis_mapping[k]) + v
        else:
            vis_mapping[k] = v

    result = {}
    for key, value in to_visualize.items():
        if "timestamp" in key:
            continue
        elif "parameters" in key:
            continue
        elif any([key.startswith(x) for x in vis_mapping["image"]]):
            result[key] = vis_image(value, fmt=fmt, **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["lidar"]]):
            result[key] = vis_lidar(value, fmt=fmt, **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["intensity"]]):
            result[key] = vis_intensity(value, fmt=fmt, **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["twist"]]):
            result[key] = vis_twist(value, fmt=fmt, **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["vector"]]):
            result[key] = vis_vector(value, fmt=fmt, **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["pose"]]):
            result[key] = vis_poses(value, fmt=fmt, **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["radar"]]):
            result[key] = vis_radar(value, fmt=fmt, **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["rpyt"]]):
            result[key] = vis_rpyt(value, fmt=fmt, **kwargs)

    if show:
        show_grid(result, show=True, **kwargs)

    return TensorDict(result)


def export(observation, path, keys=None, **kwargs):
    """ Export observation visualization to file.
    """
    imgs = visualize(observation, keys, fmt="numpy", show=False, **kwargs)
    if len(imgs.keys()) == 1:
        frame = imgs[next(iter(imgs))]
    else:
        frame = show_grid(imgs, fmt="numpy")
    imageio.imwrite(path, frame)


def visualize_sequence(
    observations, keys=None, max_length=8, show=False, **kwargs
):
    """ Visualize a sequence of observations.

    Observations are provided as a dictionary with string keys and tensor
    (or numpy array) values. The tensors have a batch and time dimension.

    :param keys: the keys of the observation to visualize (default all)
    :param max_length: the max width of the grid (default None)
    :param show: whether to immediately show the image with matplotlib
    (default False)
    :param **kwargs: additional parameters
    :returns: an image tensor or numpy array. Time and batch dimensions are
    shown as consecutive frames in a grid.
    """
    if keys is not None:
        to_visualize = TensorDict({key: observations[key] for key in keys})
    else:
        to_visualize = observations

    vis_mapping = kwargs.get("vis_mapping", {})
    for k, v in default_vis_mapping.items():
        if k in vis_mapping:
            vis_mapping[k] = list(vis_mapping[k]) + v
        else:
            vis_mapping[k] = v

    result = {}
    for key, seq in to_visualize.items():
        if "timestamp" in key:
            continue
        elif "parameters" in key:
            continue
        elif any([key.startswith(x) for x in vis_mapping["image"]]):
            result[key] = vis_images(seq, max_length=max_length, **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["lidar"]]):
            result[key] = vis_lidars(seq, max_length=max_length, **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["intensity"]]):
            result[key] = vis_intensities(seq, max_length=max_length,
                                          **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["twist"]]):
            result[key] = vis_twists(seq, max_length=max_length, **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["vector"]]):
            if isinstance(seq, InnerMultivariateNormal):
                mu = seq.mean
                sigma = seq.stdev
                result[key] = vis_vectors(mu, sigma,
                                          max_length=max_length, **kwargs)
            else:
                result[key] = vis_vectors(seq, max_length=max_length,
                                          **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["pose"]]):
            result[key] = vis_poses(seq, **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["position"]]):
            result[key] = vis_positions(seq, **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["radar"]]):
            result[key] = vis_radars(seq, **kwargs)
        elif any([key.startswith(x) for x in vis_mapping["rpyt"]]):
            result[key] = vis_rpyts(seq, **kwargs)

    if show:
        show_grid(result, show=True, **kwargs)

    return TensorDict(result)


def export_sequence(observations, path, keys=None, **kwargs):
    """ Export observation sequence visualization to file.
    """
    result = []
    if path[-3:] == "mp4":
        for i in range(observations.shape[1]):
            if observations.shape[0] == 1:
                imgs = visualize(observations[0, i, ...],
                                 keys, fmt="numpy", show=False, **kwargs)
            else:
                raise Exception("Only sequences with batch_size 1"
                                "can currently be exported as mp4")
            if len(imgs.keys()) == 1:
                frame = imgs[next(iter(imgs))]
            else:
                frame = show_grid(imgs, fmt="numpy")
            result.append(frame)
        imageio.mimwrite(path, result)
    else:
        imgs = visualize_sequence(observations, keys, show=False, **kwargs)
        frame = show_grid(imgs, fmt="numpy")
        imageio.imwrite(path, frame)


def vis_images(images, max_length=None, fmt="torch", show=False,
               padding=2, pad_value=0, **kwargs):
    """ Visualize a sequence of image data.

    Plots both numpy (HWC) and torch (CHW) formatted data.
    """
    images = numpyify(images)

    if len(images.shape) == 4:
        # add batch dimension
        images = np.expand_dims(images, 0)

    if images.shape[2] != 3 and images.shape[2] != 1:
        # numpy HWC format... permute dimensions to torch CHW format
        images = np.transpose(images, (0, 1, 4, 2, 3))

    if images.dtype == np.uint8:
        # convert to float, divide by 255
        images = images.astype(np.float32)
        images = images / 255

    # calculate number of images per row
    nrow = images.shape[1]
    if max_length is not None:
        if images.shape[1] > max_length:
            nrow = max_length

    # make grid
    shape = [images.shape[0] * images.shape[1]] + list(images.shape[2:])
    images = images.reshape(shape)
    img = make_grid(images.clip(0.0, 1.0), nrow=nrow,
                    padding=padding, pad_value=pad_value)
    # render using matplotlib if requested
    if show:
        to_show = np.transpose(img, (1, 2, 0))
        plt.imshow(to_show)
        plt.axis("off")
        plt.show()

    if fmt == "numpy":
        img = np.transpose(img, (1, 2, 0))
        img *= 255
        img = img.astype(np.uint8)
    elif fmt == "torch":
        img = torch.as_tensor(img)

    return img


def vis_image(image, fmt="torch", show=False,
              hflip=False, vflip=False, **kwargs):
    """ Visualize a single image.
    """
    img = numpyify(image)

    assert len(img.shape) == 3

    if img.shape[0] != 3 and img.shape[0] != 1:
        # numpy HWC format... permute dimensions to torch CHW format
        img = np.transpose(img, (2, 0, 1))

    if img.dtype == np.uint8:
        # convert to float, divide by 255
        img = img.astype(np.float32)
        img = img / 255

    if vflip:
        img = np.flip(img, 1)

    if hflip:
        img = np.flip(img, 2)

    img = np.clip(img, 0, 1)

    if show:
        to_show = np.transpose(img, (1, 2, 0))
        plt.imshow(to_show)
        plt.axis("off")
        plt.show()

    if fmt == "numpy":
        img = np.transpose(img, (1, 2, 0))
        img *= 255
        img = img.astype(np.uint8)
    elif fmt == "torch":
        img = torch.as_tensor(img)

    return img


def vis_intensities(maps, max_length=None, fmt="torch",
                    show=False, **kwargs):
    """ Visualize a sequence of 2D maps as intensities.
    """
    maps = numpyify(maps)
    # sometimes 2D maps have extra 1 dimension
    # to enable processing by CNN
    # this 1 dimension can be last in numpy format (HWC, C=1)
    if maps.shape[-1] == 1:
        maps = np.squeeze(maps, -1)

    # or can be the -3rd in torch format (CHW, C=1)
    # but be aware that it might also be a missing batch dimension
    # and a sequence lenght of 1
    if maps.shape[-3] == 1 and len(maps.shape) > 3:
        maps = np.squeeze(maps, -3)

    if len(maps.shape) == 3:
        # add batch dimension
        maps = np.expand_dims(maps, 0)

    # this is a slow loop... do something more clever
    # to plot everything in one go?
    batch = []
    for i in range(maps.shape[0]):
        sequence = []
        for j in range(maps.shape[1]):
            colormap = maps[i][j]
            img = vis_intensity(colormap)
            sequence.append(img)
        s = np.stack(sequence, axis=0)
        batch.append(s)
    b = np.stack(batch, axis=0)
    return vis_images(b, max_length, fmt, show)


def vis_intensity(colormap, fmt="torch", show=False,
                  hflip=False, vflip=False,
                  min_intensity=None, max_intensity=None,
                  **kwargs):
    """ Visualize a intensity map.
    """
    colormap = numpyify(colormap)
    # sometimes 2D maps have extra 1 dimension
    # to enable processing by CNN
    if colormap.shape[-1] == 1:
        colormap = np.squeeze(colormap, -1)

    assert len(colormap.shape) == 2

    if vflip:
        colormap = np.flip(colormap, 0)

    if hflip:
        colormap = np.flip(colormap, 1)

    if min_intensity is None:
        colormap = colormap - np.min(colormap)
    else:
        colormap = colormap - min_intensity

    if max_intensity is None:
        colormap = colormap / np.max(colormap)
    else:
        colormap = colormap / max_intensity

    viridis = cm.get_cmap("viridis", 256)
    colormap = viridis(colormap)[:, :, 0:3]
    return vis_image(colormap, fmt, show)


def vis_lidars(
    scans,
    angle_min=-np.pi / 2,
    angle_incr=None,
    max_range=6,
    render_lines=False,
    axis=False,
    max_length=None,
    fmt="torch",
    show=False,
    **kwargs
):
    """ Visualize lidar scans.

    :param angle_min: the angle of the first lidar ray (default -pi/2)
    :param angle_incr: the shift in angle in between rays
    (default None, the increments are calculated by assuming a scan range
    of angle_min, -angle_min)
    :param max_range: the maximal range to plot
    :param render_lines: render ray lines or not (default False)
    :param axis: whether to plot axis (default False)
    """
    scans = numpyify(scans)

    if len(scans.shape) == 2:
        # add batch dimension
        scans = np.expand_dims(scans, 0)

    batch = []
    for i in range(scans.shape[0]):
        sequence = []
        for j in range(scans.shape[1]):
            scan = scans[i][j]
            img = vis_lidar(scan, angle_min, angle_incr,
                            max_range, render_lines, axis)
            sequence.append(img)
        s = np.stack(sequence, axis=0)
        batch.append(s)
    b = np.stack(batch, axis=0)
    return vis_images(b, max_length, fmt, show, pad_value=1)


def vis_lidar(
    scan,
    angle_min=-1.5585244894,
    angle_incr=None,
    max_range=6,
    render_lines=False,
    axis=False,
    fmt="torch",
    show=False,
    **kwargs
):
    """ Visualize a lidar scan.

    :param angle_min: the angle of the first lidar ray (default -pi/2)
    :param angle_incr: the shift in angle in between rays
    (default None, the increments are calculated by assuming a scan range
    of angle_min, -angle_min)
    :param max_range: the maximal range to plot
    :param render_lines: render ray lines or not (default False)
    :param axis: whether to plot axis (default False)
    """
    scan = numpyify(scan)
    assert len(scan.shape) == 1
    fig = plt.figure(figsize=(4, 4), dpi=100)
    x = []
    y = []
    angle = angle_min
    if angle_incr is None:
        angle_incr = np.abs(angle_min) * 2 / (len(scan) + 1)
    for i in range(len(scan)):
        r = scan[i].item()
        x.append(-r * np.sin(angle))
        y.append(r * np.cos(angle))
        angle += angle_incr
    start_point = (0.0, 0.0)
    ax = plt.gca()
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(0, max_range)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.axis(axis)
    if render_lines:
        for i in range(len(x)):
            line = mpl.lines.Line2D(
                [start_point[0], x[i]], [start_point[1], y[i]], lw=0.05
            )
            ax.add_line(line)
    plt.scatter(x, y, s=[1] * len(x))

    if show:
        plt.show()
    img = fig2img(fig, fmt)
    plt.close(fig)
    return img


def vis_twists(
    twists,
    max_linear=0.3,
    max_angular=2,
    max_length=None,
    fmt="torch",
    show=False,
    **kwargs
):
    """ Visualize Twist commands.

    :param max_linear: the maximal linear velocity to render
    :param max_angular: the maximal angular velocity to render
    """
    twists = numpyify(twists)

    if len(twists.shape) == 2:
        # add batch dimension
        twists = np.expand_dims(twists, 0)

    batch = []
    for i in range(twists.shape[0]):
        sequence = []
        for j in range(twists.shape[1]):
            twist = twists[i][j]
            img = vis_twist(twist, max_linear, max_angular)
            sequence.append(img)
        s = np.stack(sequence, axis=0)
        batch.append(s)
    b = np.stack(batch, axis=0)
    return vis_images(b, max_length, fmt, show, pad_value=1)


def vis_twist(
    twist, max_linear=0.5, max_angular=2, fmt="torch", show=False, **kwargs
):
    """ Visualize a Twist command.

    :param max_linear: the maximal linear velocity to render
    :param max_angular: the maximal angular velocity to render
    """
    fig, ax = plt.subplots()
    arrow_width = 0.1
    plt.xlim(-max_angular, max_angular)
    plt.ylim(-max_linear, max_linear)
    plt.axis("off")
    vx = twist[-5]
    vy = twist[-6]
    ax.annotate(
        "",
        xy=(vx, vy),
        xytext=(0, 0),
        arrowprops=dict(facecolor="black", shrink=arrow_width),
    )
    va = twist[-1]
    if va != 0:
        ax.annotate(
            "",
            xy=(-va, 0),
            xytext=(0, 0),
            arrowprops=dict(facecolor="black", shrink=arrow_width),
        )

    if show:
        plt.show()
    img = fig2img(fig, fmt)
    plt.close(fig)
    return img


def vis_rpyts(
    rpyts, max_length=None, fmt="torch", show=False, **kwargs
):
    """ Visualize a ActuatorControl command containig roll, pitch, yaw
    and thrust.
    """
    rpyts = numpyify(rpyts)

    if len(rpyts.shape) == 2:
        # add batch dimension
        rpyts = np.expand_dims(rpyts, 0)

    batch = []
    for i in range(rpyts.shape[0]):
        sequence = []
        for j in range(rpyts.shape[1]):
            rpyt = rpyts[i][j]
            img = vis_rpyt(rpyt, **kwargs)
            sequence.append(img)
        s = np.stack(sequence, axis=0)
        batch.append(s)
    b = np.stack(batch, axis=0)
    return vis_images(b, max_length, fmt, show, pad_value=1)


def vis_rpyt(
    rpyt, max_roll=0.3, max_pitch=0.3, max_yaw=0.2,
    fmt="torch", show=False, **kwargs
):
    """ Visualize a ActuatorControl command containig roll, pitch, yaw
    and thrust.
    """
    fig, ax = plt.subplots()
    arrow_width = 0.1
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis("off")
    roll = rpyt[0] / max_roll
    pitch = rpyt[1] / max_pitch
    yaw = rpyt[2] / max_yaw
    thrust = rpyt[3]
    if roll != 0 or pitch != 0:
        ax.annotate(
            "",
            xy=(roll, pitch),
            xytext=(0, 0),
            arrowprops=dict(facecolor="black", shrink=arrow_width),
        )

    if yaw != 0:
        ax.annotate(
            "",
            xy=(-yaw, 0),
            xytext=(0, 0),
            arrowprops=dict(facecolor="black", shrink=arrow_width),
        )

    if thrust != 0:
        ax.annotate(
            "",
            xy=(0.5, thrust),
            xytext=(0.5, 0),
            arrowprops=dict(facecolor="red", shrink=arrow_width),
        )

    if show:
        plt.show()
    img = fig2img(fig, fmt)
    plt.close(fig)
    return img


def vis_vectors(vectors, sigmas=None, max_length=None,
                fmt="torch", show=False, **kwargs):
    """ Visualize a batch of vectors.
    """
    vectors = numpyify(vectors)
    if sigmas is not None:
        sigmas = numpyify(sigmas)

    if len(vectors.shape) == 2:
        # add batch dimension
        vectors = np.expand_dims(vectors, 0)
        if sigmas:
            sigmas = np.expand_dims(sigmas, 0)

    batch = []  # batch can be a pre allocated numpy array
    for i in range(vectors.shape[0]):
        sigma = None
        if sigmas is not None:
            sigma = sigmas[i]

        img = vis_vector(vectors[i], sigma, **kwargs)
        batch.append(img)
    b = np.stack(batch, axis=0)
    return vis_images(b, max_length, fmt, show, pad_value=1)


def vis_vector(vector, sigma=None, ylim=None, fmt="torch",
               show=False, **kwargs):
    """ Visualize a sequence of vectors as line plots
    """
    vector = numpyify(vector)
    # if we still have a sequence of vectors, only plot the last?
    if len(vector.shape) == 3:
        vector = vector[-1, :, :]
    assert len(vector.shape) == 2
    fig, ax = plt.subplots()
    ax.set_xticklabels([])
    if ylim:
        ax.set_ylim(*ylim)
    for i in range(vector.shape[-1]):
        mu = vector[:, i]
        plt.plot(mu)
        if sigma is not None:
            sigma = numpyify(sigma)
            std = sigma[:, i]
            plt.fill_between(
                x=range(len(mu)), y1=mu - std, y2=mu + std, alpha=0.2
            )

    if show:
        plt.show()

    img = fig2img(fig, fmt)
    plt.close(fig)
    return img


def vis_poses(pose_vector, quaternion="XYZW", forward=[1, 0, 0],
              frame="Z_UP", quiver_length=1.0,
              xlim=None, ylim=None, zlim=None,
              fmt="torch", show=False, **kwargs):
    pose_vector = numpyify(pose_vector)
    if len(pose_vector.shape) == 1:
        pose_vector = np.expand_dims(pose_vector, 0)
    if len(pose_vector.shape) == 2:
        pose_vector = np.expand_dims(pose_vector, 0)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.gca(projection="3d")
    for i in range(pose_vector.shape[0]):
        for j in range(pose_vector.shape[1]):
            p = pose_vector[i, j, :]
            x, y, z = p[:3]
            if p.shape[0] == 6 or quaternion == "log":
                qx, qy, qz = p[3:6]
                log_q = Quaternion(x=qx, y=qy, z=qz, w=0.0)
                rotation = Quaternion.exp(log_q)
            elif quaternion == "WXYZ":
                qw, qx, qy, qz = p[3:7]
                rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            else:
                qx, qy, qz, qw = p[3:7]
                rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)

            if frame == "Z_DOWN":
                # z-axes points downwards (i.e. pixhawk drone)
                rotation = rotation * Quaternion(axis=[1, 0, 0], angle=180)
            elif frame == "Z_IN":
                # z-axes points inwards the camera (i.e. t265 camera)
                rotation = rotation * Quaternion(axis=[1, 0, 0], angle=90)
            u, v, w = rotation.rotate(forward)
            ax.quiver(
                x, y, z, u, v, w,
                length=quiver_length, normalize=True,
                color='C' + str(i)
            )

    ax.plot([0], [0], [0], marker="x")
    ax.view_init(45, 95)
    ax.set_xlabel('$X$', fontsize=15)
    ax.set_ylabel('$Y$', fontsize=15)
    ax.set_zlabel('$Z$', fontsize=15)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if zlim:
        ax.set_zlim(*zlim)

    if show:
        plt.show()

    img = fig2img(fig, fmt)
    plt.close(fig)
    return img


def vis_positions(positions, cross_latest=True, xlim=None, ylim=None,
                  fmt="torch", show=False, **kwargs):
    positions = numpyify(positions)
    if len(positions.shape) == 1:
        positions = np.expand_dims(positions, 0)
    if len(positions.shape) == 2:
        positions = np.expand_dims(positions, 0)
    if len(positions.shape) == 3 and positions.shape[-2] == 2:
        # besides positions also covariances provided
        # ignore for now
        positions = np.expand_dims(positions[:, 0, :], 0)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    for i in range(positions.shape[0]):
        xs = positions[i, :, 0]
        ys = positions[i, :, 1]
        ax.scatter(xs, ys)

        if cross_latest:
            if len(xs) > 0:
                ax.plot(xs[-1], ys[-1], 'r', marker='x',
                        mew=4, markersize=12)

    if show:
        plt.show()

    img = fig2img(fig, fmt)
    plt.close(fig)
    return img


def vis_radars(sequence_adc_samples, max_length=None, fmt="torch",
               show=False, **kwargs):
    """ Visualize a sequence of raw adc samples as intensities.
    """
    sequence_adc_samples = numpyify(sequence_adc_samples)

    if len(sequence_adc_samples.shape) == 6:
        # add batch dimension
        sequence_adc_samples = np.expand_dims(sequence_adc_samples, 0)

    batch = []
    for i in range(sequence_adc_samples.shape[0]):
        sequence = []
        for j in range(sequence_adc_samples.shape[1]):
            adc_samples = sequence_adc_samples[i][j]
            img = vis_radar(adc_samples, **kwargs)
            sequence.append(img)
        s = np.stack(sequence, axis=0)
        batch.append(s)
    b = np.stack(batch, axis=0)
    return vis_images(b, max_length, fmt, show)


def vis_radar(adc_samples, fmt="torch", show=False, **kwargs):
    intensities = []
    for i in range(adc_samples.shape[1]):
        for j in range(adc_samples.shape[2]):
            intensities.append(adc_samples[:, i, j, :, 0])
            intensities.append(adc_samples[:, i, j, :, 1])

    intensities = np.stack(intensities)
    return vis_intensities(intensities, max_length=2,
                           fmt=fmt, show=show, **kwargs)


def fig2img(fig, fmt="torch"):
    """ Render a Matplotlib Figure into an image array.

    :param format: 'numpy' or 'torch'
    """
    fig.canvas.draw()

    # TODO prevent this from initializing the alpha channel
    s, (width, height) = fig.canvas.print_to_buffer()
    buf = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    # we don't need the alpha channel
    buf = buf[:, :, 0:3]

    img = None
    if fmt == "torch":
        buf = buf.swapaxes(2, 0).swapaxes(1, 2)
        buf = buf / 255.0
        img = torch.as_tensor(buf)
    elif fmt == "numpy":
        img = buf
    else:
        raise Exception("Unsupported format: {}".format(fmt))

    return img


def show_grid(imgs, fmt="torch", show=False, **kwargs):
    """ Show a dictionary of images as a grid in matplotlib.
    """
    num_panes = len(imgs.keys())
    nrows = int(math.sqrt(num_panes))
    ncols = math.ceil(num_panes / nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    title = kwargs.get("title", None)
    if title is not None:
        fig.suptitle(title, fontsize=10)
    n_ax = 0
    for key in imgs.keys():
        if isinstance(axes, np.ndarray):
            if axes.ndim == 2:
                ax = axes[int(n_ax / axes.shape[1]), n_ax % axes.shape[1]]
            else:
                ax = axes[n_ax]
        else:
            ax = axes
        ax.axis("off")
        ax.set_title(key, {"fontsize": 8})
        to_show = np.transpose(imgs[key], (1, 2, 0))
        to_show = numpyify(to_show)
        ax.imshow(to_show)
        n_ax += 1

    # clear unused panes
    if n_ax < nrows * ncols:
        while n_ax < nrows * ncols:
            ax = axes[int(n_ax / axes.shape[1]), n_ax % axes.shape[1]]
            ax.axis("off")
            n_ax += 1

    if show:
        plt.show()

    img = fig2img(fig, fmt)
    plt.close(fig)
    return img


def make_grid(arr, nrow=8, padding=2, pad_value=0):
    """ Make a grid of images, similar to torchvision.make_grid
    """
    # we use the numpy methods since torch tensors also implement
    # the np.ndarray interface and we don't care about gradients here
    # this way make_grid will also work on h5 datasets and ndarrays
    # we will still respect the torch CHW convention to not break any earlier
    # code
    # ! NOTE that we don't support some optional torchvision make_grid
    # arguments

    # first get the input to a 4d array in BCHW format
    if len(arr.shape) == 2:
        # single gray_scale image
        arr = np.expand_dims(arr, axis=0)  # arr.shape has changed!
    if len(arr.shape) == 3:
        # single rgb image
        if arr.shape[0] == 1:
            new_shape = (3, arr.shape[1:])
            arr = np.reshape(arr, new_shape)
        arr = np.expand_dims(arr, axis=0)  # arr.shape has changed

    if len(arr.shape) == 4 and arr.shape[1] == 1:
        new_shape = (arr.shape[0], 3, *arr.shape[2:])
        arr = np.concatenate([arr, arr, arr], axis=1)
        arr = np.reshape(arr, new_shape)

    nmaps = arr.shape[0]
    xmaps = min(nrow, nmaps)
    if xmaps == 0:
        return np.zeros([3, 1, 1])
    ymaps = int(np.ceil(nmaps / xmaps))
    height, width = arr.shape[2] + padding, arr.shape[3] + padding
    num_channels = arr.shape[1]
    # create final image grid array in CHW format
    grid = np.full(
        (num_channels, height * ymaps + padding, width * xmaps + padding),
        pad_value,
        dtype=np.float32,
    )
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            y_start = y * height + padding
            y_stop = y_start + (height - padding)
            x_start = x * width + padding
            x_stop = x_start + (width - padding)
            # remember grid is CHW mpt BCHW
            arr = numpyify(arr)
            grid[:, y_start:y_stop, x_start:x_stop] = arr[k]
            k += 1
    # finally convert grid to the same type is the input type
    if isinstance(arr, torch.Tensor):
        grid = torch.as_tensor(grid)
    return grid


def numpyify(arr):
    """ Convert to numpy array if torch Tensor.
    """
    try:
        return arr.detach().cpu().numpy()
    except AttributeError:
        return arr
