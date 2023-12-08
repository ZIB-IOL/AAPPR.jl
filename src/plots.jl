using Plots
using LaTeXStrings
using StatsBase
using Interpolations

"""
    find_interval(x_min, x_max)

Returns the interval that contains the range of x values. This is used to determine the x-axis and y_axis tick marks.
"""
function find_interval(x_min, x_max)
    # Determine the nearest powers of 10 to x_min and x_max
    a = Int64(floor(log10(x_min)))
    b = Int64(ceil(log10(x_max)))
    
    # Return the interval, discretized nicely.
    return [10.0^i for i in a:b]
end


"""
    get_markers(x_data::AbstractVector, y_data::AbstractVector, n_markers::Int64, x_scale::Symbol, y_scale::Symbol)

Returns the x and y coordinates of the markers that will be plotted on the graph. This takes into account the scaling of the axes.
"""
function get_markers(x_data::AbstractVector, y_data::AbstractVector, n_markers::Int64, x_scale::Symbol, y_scale::Symbol)
    if x_scale == :log10
        x_marker = exp10.(range(log10(x_data[1]), log10(x_data[end]), length=n_markers))
        if y_scale == :log10
            itp = LinearInterpolation(log10.(x_data), log10.(y_data))
            y_marker = exp10.(itp(log10.(x_marker)))
        elseif y_scale == :identity
            itp = LinearInterpolation(log10.(x_data), y_data)
            y_marker = itp(log10.(x_marker))
        else 
            error("y_scale must be :log10 or :identity")
        end
    elseif x_scale == :identity
        x_marker = range(x_data[1], x_data[end], length=n_markers)
        if y_scale == :log10
            itp = LinearInterpolation(x_data, log10.(y_data))
            y_marker = exp10.(itp(x_marker))
        elseif y_scale == :identity
            itp = LinearInterpolation(x_data, y_data)
            y_marker = itp(x_marker)
        else 
            error("y_scale must be :log10 or :identity")
        end
    else
        error("x_scale must be :log10 or :identity")
    end
    return x_marker, y_marker
end


"""
plotting_function(
    x_data,
    y_data,
    labels,
    x_axis_label,
    y_axis_label;
    y_stddev=nothing,
    fill_alpha=0.2,
    save_figure::Bool=true,
    relative_path_to_figures_directory::String="figures/",
    figure_name::String="figure",
    x_axis_scale::Symbol=:log10,
    y_axis_scale::Symbol=:log10,
    x_plot_size::Int64=400,
    y_plot_size::Int64=300,
    dots_per_inch::Int64=800,
    legend_location::Symbol=:bottomleft,
    plot_font::String="Computer Modern",
    guide_font_size=10,
    tick_font_size::Int64=8,
    legend_font_size::Int64=10,
    line_width::Int64=2,
    marker_size::Int64=4,
    n_markers::Union{Int64, Nothing}=nothing,
    thickness_scaling::Float64=1.,
    tick_digits_rounding::Int64=2
    )

A function that handles all the plotting parameters to create a good-looking plot.
"""
function plotting_function(
    x_data,
    y_data,
    labels,
    x_axis_label,
    y_axis_label;
    y_stddev=nothing,
    fill_alpha=0.2,
    save_figure::Bool=true,
    relative_path_to_figures_directory::String="figures/",
    figure_name::String="figure",
    x_axis_scale::Symbol=:log10,
    y_axis_scale::Symbol=:log10,
    x_plot_size::Int64=400,
    y_plot_size::Int64=300,
    dots_per_inch::Int64=800,
    legend_location::Symbol=:bottomleft,
    plot_font::String="Computer Modern",
    guide_font_size=10,
    tick_font_size::Int64=8,
    legend_font_size::Int64=10,
    line_width::Int64=2,
    marker_size::Int64=4,
    n_markers::Union{Int64, Nothing}=nothing,
    thickness_scaling::Float64=1.,
    tick_digits_rounding::Int64=2
    )
    n = length(x_data)
    if y_stddev === nothing
        y_stddev = [zeros(length(x_data[i])) for i in 1:n]
    end
    # set up the color palette to be the right length such that cycling through the colors will be the same as cycling through the data
    colors = [:red, :blue, :green, :orange, :purple, :brown, :pink, :black, :gray, :olive, :cyan][1:n]
    markers = [:circle :square :diamond :star5 :hexagon :triangle :star4 :pentagon :star6 :cross :star7][:, 1:n]
    # plot the data
    plot()
    for i in 1:n
        plot!(
            x_data[i],
            y_data[i],
            label=nothing,
            xlabel=x_axis_label,
            ylabel=y_axis_label,
            xscale=x_axis_scale,
            yscale=y_axis_scale,
            size=(x_plot_size, y_plot_size),
            dpi=dots_per_inch,
            legend=legend_location,
            framestyle=:box,
            palette=colors,
            font=plot_font,
            guidefont=font(Int64(round(guide_font_size/thickness_scaling)), plot_font),
            tickfont=font(Int64(round(tick_font_size/thickness_scaling)), plot_font),
            legendfont=font(pointsize=Int64(round(legend_font_size/thickness_scaling)), family=plot_font),
            linewidth=Int64(round(line_width/thickness_scaling));
            ribbon=y_stddev[i],
            fillalpha=fill_alpha
            )
    end

    if n_markers !== nothing
        # add nicely spaced markers
        x_markers = []
        y_markers = []
        for i in eachindex(x_data)
            (x_marker, y_marker) = get_markers(x_data[i], y_data[i], n_markers, x_axis_scale, y_axis_scale)
            push!(x_markers, x_marker)
            push!(y_markers, y_marker)
        end
    else
        x_markers = x_data
        y_markers = y_data
    end

        scatter!(
            x_markers,
            y_markers,
            xscale=x_axis_scale,
            yscale=y_axis_scale,
            palette=colors,
            label=nothing,
            markerstrokecolor=:black,
            markerstrokestyle=:solid,
            marker=markers,
            markersize=Int64(round(marker_size/thickness_scaling)),
            )

    # make legend
    x_short = [[x_data[i][1]] for i in 1:length(x_data)]
    y_short = [[y_data[i][1]] for i in 1:length(y_data)]
    plot!(
        x_short,
        y_short,
        xscale=x_axis_scale,
        yscale=y_axis_scale,
        palette=colors,
        label=labels,
        markerstrokecolor=:black,
        markerstrokestyle=:solid,
        marker=markers,
        markersize=Int64(round(marker_size/thickness_scaling))
    )


    # if x_axis_scale == :log10
    #     x_ticks = find_interval(minimum(map(minimum, x_data)), maximum(map(maximum, x_data)))
    # else
    #     x_min = minimum(map(minimum, x_data))
    #     x_max = maximum(map(maximum, x_data))
    #     x_ticks = round.(collect(range(x_min, x_max, 5)); digits=tick_digits_rounding)
    # end
    # y_min = minimum(map(minimum, y_data))
    # y_max = maximum(map(maximum, y_data))
    # println("y_min = ", y_min)
    # println("y_max = ", y_max)
    # if y_axis_scale == :log10
    #     y_ticks = find_interval(y_min, y_max)
    #     println("y_ticks = ", y_ticks)
    # else
    #     y_ticks = round.(collect(range(y_min, y_max, 5)); digits=tick_digits_rounding)
    # end
    # plot!(
    #     xticks=x_ticks,
    #     yticks=y_ticks,
    #     )

    # save figure
    if save_figure
        savefig(relative_path_to_figures_directory*figure_name)
    end
    plot!()
end



