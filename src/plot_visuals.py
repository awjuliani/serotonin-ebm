import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO
import PIL.Image as Image
import imageio


# Style settings
style_map = {
    "a": ("Neural Activity", "Energy", "terrain"),
    "b": ("Synaptic Weight", "Energy", "plasma"),
    "c": ("Belief", "Precision", "magma"),
}


def _create_figure_and_axes(dim=2, title="", close=True):
    """Helper function to create a figure and axis."""
    fig = plt.figure(figsize=(12, 10))
    if dim == 3:
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    else:
        ax = fig.add_subplot(111)
    ax.set_title(title, size=18)
    if close:
        plt.close()
    return fig, ax


def plot_3d(x, y, z, point, title="", close=True, plot_style="a"):
    fig, ax = _create_figure_and_axes(dim=3, title=title, close=close)

    dim_name, val_name, cmap = style_map.get(plot_style, style_map["a"])

    ax.plot_surface(x, y, z, cmap=cmap, zorder=1, alpha=0.9, vmin=0, vmax=1)

    ax.set_xlabel(f"{dim_name} A", size=14)
    ax.set_ylabel(f"{dim_name} B", size=14)
    ax.set_zlabel(val_name, size=14)
    ax.set_zlim(0, 1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    return fig


def plot_2d(x, y, z, points, title, close=True, plot_style="a"):
    fig, ax = _create_figure_and_axes(dim=2, title=title, close=close)
    dim_name, val_name, cmap = style_map.get(plot_style, style_map["a"])

    cf = ax.contourf(x, y, z, cmap=cmap, vmin=0, vmax=1, levels=50)

    if points is not None:
        half_size = int(x.shape[0] / 2)
        points = (1 + points - half_size) / half_size
        # add small noise so points don't overlap
        points += np.random.normal(0, 0.005, points.shape)
        # clip points to be between -1 and 1
        points = np.clip(points, -1, 1)
        ax.scatter(
            points[:, 1],
            points[:, 0],
            color="black",
            edgecolor="white",
            s=100,
            zorder=2,
            alpha=0.9,
        )

    # add colorbar
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(val_name, size=14)

    ax.set_xlabel(f"{dim_name} A", size=14)
    ax.set_ylabel(f"{dim_name} B", size=14)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    return fig


def fig2img(fig, img_1=None):
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="jpeg", dpi=60)
    buf.seek(0)
    img = Image.open(buf)

    if img_1 is not None:
        # concat image with img_1 along x axis
        img_1 = Image.fromarray(img_1)
        img = Image.concat((img_1, img))

    return np.asarray(img)


def plot_figure(
    plot_type, x, y, z_use, point, title, plot_style, output_type, to_img=False
):
    if plot_type == "3d":
        fig = plot_3d(
            x, y, z_use, point, title, close=output_type == "gif", plot_style=plot_style
        )
    elif plot_type == "2d":
        fig = plot_2d(
            x, y, z_use, point, title, close=output_type == "gif", plot_style=plot_style
        )
    else:
        raise ValueError("plot_type must be 1d, 2d, or 3d")

    return fig2img(fig) if to_img else fig


def generate_images(e_mods, e_star, x, y, points, plot_params, show_all=False):
    images = []
    for i in range(0, len(e_mods), plot_params.plot_freq):
        title = f"Step {i}" if plot_params.output_type == "gif" else ""
        if plot_params.show_points:
            use_point = points[i]
        else:
            use_point = None
        img = plot_figure(
            plot_params.plot_type,
            x,
            y,
            e_mods[i],
            use_point,
            title,
            plot_params.plot_style,
            plot_params.output_type,
            to_img=plot_params.output_type == "gif",
        )
        if show_all:
            img_s = plot_figure(
                plot_params.plot_type,
                x,
                y,
                e_star,
                None,
                title,
                plot_params.plot_style,
                plot_params.output_type,
                to_img=plot_params.output_type == "gif",
            )
            img_e = plot_figure(
                plot_params.plot_type,
                x,
                y,
                e_mods[0],
                None,
                title,
                plot_params.plot_style,
                plot_params.output_type,
                to_img=plot_params.output_type == "gif",
            )
            # create a combined image with the two side by side
            img = np.concatenate((img_e, img, img_s), axis=1)
        images.append(img)
    return images


def generate_combined_images(matrices, x, y, points, plot_style, output_type):
    """Generate combined images based on type (2d or 3d)"""
    types = ["2d", "2d"]
    images = [
        generate_images(matrices, x, y, t, points, plot_style, output_type)
        for t in types
    ]
    return [np.concatenate((i3, i2), axis=1) for i3, i2 in zip(*images)]


def generate_gif(e_mods, x, y, zs, e_star, plot_params, exp_name, plt_name=""):
    images = generate_images(
        e_mods,
        e_star,
        x,
        y,
        zs,
        plot_params,
    )
    if not os.path.exists("./output/figures/sims"):
        os.makedirs("./output/figures/sims")
    imageio.mimsave(
        f"./output/figures/sims/sim_{exp_name}{plt_name}.gif", images, fps=5
    )
    plt.close()


def generate_pdf(e_mods, e_star, x, y, points, plot_params, plt_name="", timestep=0):
    image = generate_images(
        e_mods[timestep : timestep + 1],
        e_star,
        x,
        y,
        points,
        plot_params,
    )[0]
    if not os.path.exists("./output/figures/sims"):
        os.makedirs("./output/figures/sims")
    # save figure using pyplot
    image.savefig(
        f"./output/figures/sims/sim_{plt_name}_{timestep}.pdf", bbox_inches="tight"
    )
    plt.close()
