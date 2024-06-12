using LaTeXStrings
import PyPlot
const plt = PyPlot

function new_figure(plot_type="full"; library=:pyplot, layout=:single)
    if layout == :spect2D
        size = (3.25, 3.175)
    elseif layout == :equal
        size = (3.175, 3.175)
    elseif plot_type == "full" || plot_type == "2D"
        size = (3.175, 2.25)
    elseif plot_type == "half"
        size = (1.56, 1.4)
    elseif plot_type == "double"
        size = (6.35, 4.50)
    end
        
    fig = plt.figure(; figsize = size)
    if layout == :single
        ax = fig.add_subplot(111)
    
        for axis in ["top", "bottom", "left", "right"]
            ax.spines[axis].set_linewidth(0.1)
            ax.spines[axis].set_color("gray")
        end
        if plot_type == "2D"
            ax.tick_params(width=0.1, direction="out", color="gray")
        else
            ax.tick_params(width=0.1, direction="in", color="gray")
        end
    
        fig, ax
    elseif layout == :spect2D
        gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[4, 1], height_ratios=[1, 4], left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[2, 1])
        ax_top = fig.add_subplot(gs[1, 1], sharex=ax)
        ax_right = fig.add_subplot(gs[2, 2], sharey=ax)
        for axis in ["top", "bottom", "left", "right"]
            ax.spines[axis].set_linewidth(0.1)
            ax.spines[axis].set_color("gray")
            ax_top.spines[axis].set_linewidth(0.1)
            ax_top.spines[axis].set_color("gray")
            ax_right.spines[axis].set_linewidth(0.1)
            ax_right.spines[axis].set_color("gray")
        end
        if plot_type == "2D"
            ax.tick_params(width=0.1, direction="out", color="gray")
        else
            ax.tick_params(width=0.1, direction="in", color="gray")
        end
        ax_top.tick_params(width=0.1, direction="in", color="gray")
        ax_right.tick_params(width=0.1, direction="in", color="gray")
        ax_top.tick_params(axis="x", labelbottom=false)
        ax_right.tick_params(axis="x", labelbottom=false)
        ax_top.tick_params(axis="y", labelleft=false)
        ax_right.tick_params(axis="y", labelleft=false)
        fig, ax, ax_top, ax_right
    end
end

function contourf(ax, x, y, z; levels=500, cmap = "viridis", vmin::Union{Nothing, Float64}=nothing, vmax::Union{Nothing, Float64}=nothing, colorbar=true)
    if isnothing(vmax) && isnothing(vmin)
        cnt = ax.contourf(x, y, z; levels, cmap)
    elseif isnothing(vmax) && !isnothing(vmin)
        cnt = ax.contourf(x, y, z; levels, cmap, vmin)
    elseif !isnothing(vmax) && isnothing(vmin)
        cnt = ax.contourf(x, y, z; levels, cmap, vmax)
    elseif !isnothing(vmax) && !isnothing(vmin)
        cnt = ax.contourf(x, y, z; levels, cmap, vmin, vmax)
    end
    for c in cnt.collections
        c.set_edgecolor("face")
    end
    if colorbar
        cbar = plt.colorbar(cnt, drawedges=false)
        cbar.outline.set_linewidth(0.1)
        cbar.outline.set_edgecolor("gray")
        cbar.ax.tick_params(width=0.1, direction="out", color="gray")
        cbar.solids.set_edgecolor("face")
        cnt, cbar
    else
        cnt
    end
end