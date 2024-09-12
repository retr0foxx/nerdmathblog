from manim import *
import numpy as np;

class ThreeDExample(ThreeDScene):
    def construct(self):

        transformation_matrix = np.array([
            [0, 0],
            [1, 2],
            [3, 3]
        ]);
        transformation_matrix_col = np.swapaxes(transformation_matrix, 0, 1);

        axes = ThreeDAxes();

        x_label = axes.get_x_axis_label(Tex("x"));
        y_label = axes.get_y_axis_label(Tex("y")).shift(UP * 1.8);

        dot = Dot3D((2, 2, 2));
        self.play(FadeIn(axes), FadeIn(dot), FadeIn(x_label), FadeIn(y_label));
        
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES, zoom=1, run_time=1.5);

        self.begin_ambient_camera_rotation(rate=0.15);

        # cube = Cube(side_length=2, fill_opacity=.5);
        # self.play(FadeIn(cube), cube.animate.move_to((1, 1, 1)));

        arrows = [];
        for vec in transformation_matrix_col:
            arrows.append(Arrow3D(ORIGIN, vec));
        
        self.play(FadeIn(arrows[0]), FadeIn(arrows[1]));

        grid_plane = NumberPlane();
        