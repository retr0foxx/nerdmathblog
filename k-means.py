import matplotlib;
# matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt
import numpy as np
import random;
from matplotlib.animation import FuncAnimation;
from scipy.spatial import ConvexHull;

def generate_data_in_circle(center, radius, n_data):
    rad_n_theta = np.random.rand(n_data, 2);
    result = np.zeros((n_data, 2), dtype=np.float64);
    for i in range(n_data):
        rad, theta = rad_n_theta[i];
        rad *= radius;
        theta *= 2 * np.pi;
        result[i][0] = np.cos(theta) * rad;
        result[i][1] = np.sin(theta) * rad;
        result[i] += center;

    return result;

def max_pow2_region(powed):
    return int(powed ** 2);

def pow2_region_rad(powed):
    return 1/(2 * powed);

def pow2_regions(powed, nth):
    g_x = nth % powed;
    g_y = (nth - g_x) / powed;
    return np.array([g_x, g_y]) * 1/powed + 1/(2 * powed);

K_clusters = 10;
K_nearest_square = int(np.power(np.ceil(np.sqrt(K_clusters)), 2)) - 1;

# Generate some random data points and centroids
np.random.seed(0)
# data_points = np.concatenate((generate_data_in_circle((.5, .5), .2, 50), generate_data_in_circle((.7, .7), .2, 60)), axis=0) # np.random.rand(100, 2)  # Example 100 data points in 2D
data_points = np.zeros((50 * K_clusters, 2), dtype=np.float64);

centroids = np.random.rand(K_clusters, 2);

def rd2():
    global data_points;
    picked = random.sample(range(K_nearest_square ** 2), k=K_clusters)
    for k in range(K_clusters):
        data_points[50 * k : 50 * (k + 1), :] = generate_data_in_circle(pow2_regions(K_nearest_square, picked[k]), pow2_region_rad(K_nearest_square), 50);
        #centroids[k] = pow2_regions(K_nearest_square, picked[k]);

def rd1():
    global data_points;
    data_points = np.random.rand(50 * K_clusters, 2);

rd1();

np.random.shuffle(data_points);

clustered_data_points = np.zeros((K_clusters, 50, 2));
for k in range(K_clusters):
    print(data_points[k * 50 : (k + 1) * 50, :].shape, clustered_data_points[k].shape);
    clustered_data_points[k] = data_points[k * 50 : (k + 1) * 50, :];

# Assign colors for each cluster (you can adjust colors as needed)
colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # yellow-green
    '#17becf'   # cyan
]

path_clusters = [];
path_centroids = [];

# Plot the data points with different colors based on cluster
plt.figure(figsize=(8, 6))
plt.xlim(-.1, 1.1);
plt.ylim(-.1, 1.1);
for i in range(K_clusters):
    color = colors[i];
    points_in_cluster = clustered_data_points[i];
    path_clusters.append(plt.scatter(points_in_cluster[:, 0], points_in_cluster[:, 1], color=color, label=f'Cluster {i+1}'))

def reassign_vis_clusters(path_list, clusters):
    for i in range(len(clusters)):
        path_list[i].set_offsets(clusters[i]);

# Plot the centroids with custom marker style
for i, centroid in enumerate(centroids):
    path_centroids.append(plt.scatter(centroid[0], centroid[1], color=colors[i], marker='s', s=100, edgecolor='gold', linewidth=2,
                label=f'Centroid {i+1}'));

#for i in range(max_pow2_region(K_nearest_square)):
#    d = pow2_regions(K_nearest_square, i);
#    plt.scatter(d[0], d[1], color="black", marker="^", s=100);

# fig, ax = plt.subplots()

s = 0;
t = False;

new_centroids = np.zeros((K_clusters, 2));

my_lines = [];
main_ax = plt.gca();

def update_plot(frame):
    global clustered_data_points, s;
    global new_centroids, centroids;
    global my_lines, main_ax;

    if (t == False):
        return;

    if (s == 0):
        clusters = [list() for i in range(K_clusters)];
        for i in range(len(data_points)):
            cdp = data_points[i];
            min_dist_idx = -1;
            last_dist = np.inf;
            for l in range(K_clusters):
                curcen = centroids[l];
                curnorm = np.linalg.norm(curcen - cdp);
                if (curnorm < last_dist):
                    last_dist = curnorm;
                    min_dist_idx = l;

            clusters[min_dist_idx].append(cdp);
        
        for i in range(len(clusters)):
            clusters[i] = np.array(clusters[i]).reshape((len(clusters[i]), 2));
        reassign_vis_clusters(path_clusters, clusters);

        clustered_data_points = clusters;
    
        chulls = [];
        for line in my_lines:
            for al in line:
                al.remove();

        my_lines = [];
        for i in range(len(clusters)):
            cluster = clusters[i];
            color = colors[i];
            if (len(cluster) < 3):
                continue;

            v = ConvexHull(cluster).vertices;
            r = cluster[v];
            for i in range(len(r)):
                a, b = r[i], r[(i + 1) % len(r)];
                my_lines.append(main_ax.plot([a[0], b[0]], [a[1], b[1]], linestyle="--", linewidth=2, color=color));
    
        print(chulls);
    elif (s == 1):
        for i in range(len(centroids)):
            new_centroids[i] = np.sum(clustered_data_points[i], axis=0) / len(clustered_data_points[i]) if len(clustered_data_points[i]) != 0 else np.zeros(2);
            #path_centroids[i].set_offsets(centroids[i]);

        #print("old:");
        #print(centroids);
        #print("new:");
        #print(new_centroids);
    elif (s == 2):
        for i in range(len(centroids)):
            path_centroids[i].set_offsets(new_centroids[i]);
    
        centroids = new_centroids;

    s = (s + 1) % 3;
    #s = (s + 1) % 3 if s < 1 else 1;
    # Update the plot title

fig, ax = plt.gcf(), plt.gca();

# Create animation
ani = FuncAnimation(plt.gcf(), update_plot, frames=100, interval=100)  # 100 frames, 1000 ms (1 second) interval


# Customize plot labels and legend
plt.title('K-means Clustering Example')
plt.xlabel('X')
plt.ylabel('Y')
# plt.legend()

# Show plot
plt.grid(True)

def do_convex_hull(points):
    if (len(points) == 0):
        return np.zeros(2);

    vertices = [points[0]];
    current_points = points[1:];
    prev_rot = -1;
    while True:
        best_point_index = -1;
        best_point_rot = 0;
        best_point_dist = 0;
        for i in range(len(current_points)):
            curp = current_points[i];
            curp_rel = curp - vertices[-1];
            dist = np.linalg.norm(curp_rel);
            y_sign = 1 if curp_rel[1] >= 0 else -1;
            cosinv = np.arccos(curp_rel[0]/dist);
            if (y_sign  == -1):
                cosinv = 2 * np.pi - cosinv;
        
            rot = cosinv;
            if (len(vertices) > 1):
                if (rot - prev_rot):
                    pass;

        if (best_point_index == -1):
            break;

from matplotlib.widgets import Button

# Function to be called when the button is clicked
def on_button_click(event):
    global centroids, s;
    s = 0;
    centroids = np.random.rand(K_clusters, 2);

def on_button_click2(event):
    global data_points, s;
    s = 0;
    rd1();

def on_button_click3(event):
    global data_points, s;
    s = 0;
    rd2();

def on_button_click4(event):
    global s, t;
    t = not t;
    s = 0;
    button4.label.set_text("start" if t == False else "stop");

# Create a button widget
"""
button = Button(ax, "Click me", color='lightblue', hovercolor='skyblue', 
                activecolor='lightgreen', size=8, 
                **{button_position})"""

button_ax = fig.add_axes((.1, .1, .1, .1));
button_ax2 = fig.add_axes((.8, .1, .1, .1));
button_ax3 = fig.add_axes((.1, .8, .1, .1));
button_ax4 = fig.add_axes((.8, .8, .1, .1));

button = Button(button_ax, "rand centers", color="lightblue");
button2 = Button(button_ax2, "rand data", color="lightblue");
button3 = Button(button_ax3, "rand data2", color="lightblue");
button4 = Button(button_ax4, "start", color="lightblue");

button.on_clicked(on_button_click)  # Connect button
button2.on_clicked(on_button_click2);
button3.on_clicked(on_button_click3);
button4.on_clicked(on_button_click4);

plt.show()