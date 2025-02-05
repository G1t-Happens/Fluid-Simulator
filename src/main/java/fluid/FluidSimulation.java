package fluid;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * FluidSimulator – A complex fluid simulator (Stable Fluids)
 * implemented in a single class as an example of modern Java best practices.
 * <p>
 * Features:
 * - Interactive fluid dynamics simulation via mouse (Left-click: Add density/momentum, Right-click: Reset)
 * - High resolution (e.g., N = 256) with parallel computation (Jacobi iteration)
 * - Swing-based visualization of the density field as a grayscale image
 * <p>
 * This example demonstrates:
 * - Use of final, constants, and meaningful names
 * - Parallel processing using Java 8 Streams
 * - Clear separation of simulation logic, rendering, and event handling
 */
public class FluidSimulation extends JPanel implements MouseListener, MouseMotionListener {

    private static final Logger LOGGER = Logger.getLogger(FluidSimulation.class.getName());

    // --- Simulation Configuration ---
    private static final int N = 256;
    private static final int SIZE = N + 2;
    private static final double DT = 0.1;
    private static final double DIFFUSION = 0.0001;
    private static final double VISCOSITY = 0.0001;
    private static final int LINEAR_SOLVER_ITERATIONS = 20;
    private static final int FORCE = 500;
    private static final int DENSITY_AMOUNT = 100;

    // --- Simulation Fields (Size: SIZE x SIZE) ---
    private double[][] density;
    private double[][] densityPrev;
    private double[][] velocityX;
    private double[][] velocityY;
    private double[][] velocityXPrev;
    private double[][] velocityYPrev;

    // --- For Mouse Interactions (to compute motion vector) ---
    private int prevMouseX = -1;
    private int prevMouseY = -1;

    /**
     * Constructor: Initializes simulation fields, adds MouseListener,
     * and starts a timer for regular simulation updates.
     */
    public FluidSimulation() {
        setPreferredSize(new Dimension(600, 600));

        // Initialize arrays
        density = new double[SIZE][SIZE];
        densityPrev = new double[SIZE][SIZE];
        velocityX = new double[SIZE][SIZE];
        velocityY = new double[SIZE][SIZE];
        velocityXPrev = new double[SIZE][SIZE];
        velocityYPrev = new double[SIZE][SIZE];

        addMouseListener(this);
        addMouseMotionListener(this);

        // Start the timer (~60 FPS)
        int delay = 1000 / 60;
        Timer timer = new Timer(delay, _ -> {
            step();
            repaint();
        });
        timer.start();
    }

    /**
     * Updates the simulation step:
     * - Adds sources (from mouse interactions)
     * - Diffusion, advection, and projection to maintain incompressibility
     * - Updates velocity and density fields
     */
    private void step() {
        // Add sources (from mouse interactions) to the fields
        addSource(velocityX, velocityXPrev);
        addSource(velocityY, velocityYPrev);
        addSource(density, densityPrev);

        double[][] temp;
        // --- Update velocity field ---
        temp = velocityXPrev;
        velocityXPrev = velocityX;
        velocityX = temp;
        diffuse(1, velocityX, velocityXPrev, VISCOSITY);

        temp = velocityYPrev;
        velocityYPrev = velocityY;
        velocityY = temp;
        diffuse(2, velocityY, velocityYPrev, VISCOSITY);

        project(velocityX, velocityY, velocityXPrev, velocityYPrev);

        temp = velocityXPrev;
        velocityXPrev = velocityX;
        velocityX = temp;
        temp = velocityYPrev;
        velocityYPrev = velocityY;
        velocityY = temp;
        advect(1, velocityX, velocityXPrev, velocityXPrev, velocityYPrev);
        advect(2, velocityY, velocityYPrev, velocityXPrev, velocityYPrev);
        project(velocityX, velocityY, velocityXPrev, velocityYPrev);

        // --- Update density field ---
        temp = densityPrev;
        densityPrev = density;
        density = temp;
        diffuse(0, density, densityPrev, DIFFUSION);
        temp = densityPrev;
        densityPrev = density;
        density = temp;
        advect(0, density, densityPrev, velocityX, velocityY);

        // Reset source fields
        clearArray(velocityXPrev);
        clearArray(velocityYPrev);
        clearArray(densityPrev);
    }

    /**
     * Adds the source to each field point: x[i][j] += DT * s[i][j].
     * The operation is performed in parallel across the rows.
     *
     * @param x Target array
     * @param s Source array
     */
    private void addSource(final double[][] x, final double[][] s) {
        IntStream.range(0, SIZE).parallel().forEach(i -> {
            for (int j = 0; j < SIZE; j++) {
                x[i][j] += DT * s[i][j];
            }
        });
    }

    /**
     * Diffusion step: Solves the implicit linear equation system
     * for the diffused field.
     *
     * @param b    Type of the field (0 = Scalar, 1 = x-component, 2 = y-component)
     * @param x    Target array
     * @param x0   Source array
     * @param diff Diffusion rate
     */
    private void diffuse(final int b, final double[][] x, final double[][] x0, final double diff) {
        double a = DT * diff * N * N;
        linSolve(b, x, x0, a, 1 + 4 * a);
    }

    /**
     * Solves the linear equation system using a parallelized Jacobi iteration.
     * This function is used in the diffusion and projection steps to maintain stability.
     *
     * @param b  Type of the field (0 = Scalar, 1 = X-Velocity, 2 = Y-Velocity)
     * @param x  Target array (resulting field)
     * @param x0 Source array (previous state)
     * @param a  Coefficient (computed from diffusion or projection parameters)
     * @param c  Normalization factor (ensures numerical stability)
     */
    private void linSolve(final int b, final double[][] x, final double[][] x0, final double a, final double c) {
        double[][] xNew = new double[SIZE][SIZE];
        for (int k = 0; k < LINEAR_SOLVER_ITERATIONS; k++) {
            IntStream.rangeClosed(1, N).parallel().forEach(i -> {
                for (int j = 1; j <= N; j++) {
                    xNew[i][j] = (x0[i][j] + a * (x[i - 1][j] + x[i + 1][j] + x[i][j - 1] + x[i][j + 1])) / c;
                }
            });
            setBnd(b, xNew);
            IntStream.rangeClosed(1, N).parallel().forEach(i -> System.arraycopy(xNew[i], 1, x[i], 1, N));
            setBnd(b, x);
        }
    }

    /**
     * Advection step: Transports the field d along the velocity fields u and v.
     * The method clamps the interpolated values to ensure they remain within valid boundaries.
     *
     * @param b  Type of the field
     * @param d  Target array
     * @param d0 Source array
     * @param u  Velocity field in the x-direction
     * @param v  Velocity field in the y-direction
     */
    private void advect(final int b, final double[][] d, final double[][] d0, final double[][] u, final double[][] v) {
        double dt0 = DT * N;
        IntStream.rangeClosed(1, N).parallel().forEach(i -> {
            for (int j = 1; j <= N; j++) {
                double x = i - dt0 * u[i][j];
                double y = j - dt0 * v[i][j];
                x = Math.clamp(x, 0.5, N + 0.5);
                y = Math.clamp(y, 0.5, N + 0.5);
                int i0 = (int) x;
                int i1 = i0 + 1;
                int j0 = (int) y;
                int j1 = j0 + 1;
                double s1 = x - i0;
                double s0 = 1 - s1;
                double t1 = y - j0;
                double t0 = 1 - t1;
                d[i][j] = s0 * (t0 * d0[i0][j0] + t1 * d0[i0][j1]) + s1 * (t0 * d0[i1][j0] + t1 * d0[i1][j1]);
            }
        });
        setBnd(b, d);
    }

    /**
     * Projection step: Enforces the incompressibility of the velocity field.
     *
     * @param u   Velocity field in the x-direction
     * @param v   Velocity field in the y-direction
     * @param p   Auxiliary field p
     * @param div Divergence field
     */
    private void project(final double[][] u, final double[][] v, final double[][] p, final double[][] div) {
        IntStream.rangeClosed(1, N).parallel().forEach(i -> {
            for (int j = 1; j <= N; j++) {
                div[i][j] = -0.5 * (u[i + 1][j] - u[i - 1][j] + v[i][j + 1] - v[i][j - 1]) / N;
                p[i][j] = 0;
            }
        });
        setBnd(0, div);
        setBnd(0, p);
        linSolve(0, p, div, 1, 4);
        IntStream.rangeClosed(1, N).parallel().forEach(i -> {
            for (int j = 1; j <= N; j++) {
                u[i][j] -= 0.5 * N * (p[i + 1][j] - p[i - 1][j]);
                v[i][j] -= 0.5 * N * (p[i][j + 1] - p[i][j - 1]);
            }
        });
        setBnd(1, u);
        setBnd(2, v);
    }

    /**
     * Sets the boundary conditions for a field.
     *
     * @param b Field type:
     *          0 = Scalar field (e.g., density),
     *          1 = X-component of velocity (negates at vertical boundaries),
     *          2 = Y-component of velocity (negates at horizontal boundaries).
     * @param x The field for which the boundary conditions should be set.
     */
    private void setBnd(final int b, final double[][] x) {
        for (int i = 1; i <= N; i++) {
            x[0][i] = (b == 1) ? -x[1][i] : x[1][i];
            x[N + 1][i] = (b == 1) ? -x[N][i] : x[N][i];
            x[i][0] = (b == 2) ? -x[i][1] : x[i][1];
            x[i][N + 1] = (b == 2) ? -x[i][N] : x[i][N];
        }
        x[0][0] = 0.5 * (x[1][0] + x[0][1]);
        x[0][N + 1] = 0.5 * (x[1][N + 1] + x[0][N]);
        x[N + 1][0] = 0.5 * (x[N][0] + x[N + 1][1]);
        x[N + 1][N + 1] = 0.5 * (x[N][N + 1] + x[N + 1][N]);
    }

    /**
     * Sets all elements of a 2D field to 0.
     * The operation is performed in parallel across the rows.
     *
     * @param arr The array to be cleared
     */
    private void clearArray(final double[][] arr) {
        IntStream.range(0, SIZE).parallel().forEach(i -> Arrays.fill(arr[i], 0));
    }

    /**
     * Overrides paintComponent to render the density field as a grayscale image.
     *
     * @param g The Graphics object
     */
    @Override
    protected void paintComponent(final Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        int panelWidth = getWidth();
        int panelHeight = getHeight();
        double cellWidth = (double) panelWidth / N;
        double cellHeight = (double) panelHeight / N;

        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                float d = (float) density[i][j];
                d = Math.clamp(d, 0, 1);
                int c = (int) (d * 255);
                g2d.setColor(new Color(c, c, c));
                g2d.fillRect((int) ((i - 1) * cellWidth), (int) ((j - 1) * cellHeight), (int) Math.ceil(cellWidth), (int) Math.ceil(cellHeight));
            }
        }
    }

    /**
     * Converts mouse coordinates to grid coordinates.
     *
     * @param x X-coordinate of the mouse
     * @param y Y-coordinate of the mouse
     * @return An array with [i, j] grid indices
     */
    private int[] toGridCoordinates(final int x, final int y) {
        int i = (int) ((x / (double) getWidth()) * N) + 1;
        int j = (int) ((y / (double) getHeight()) * N) + 1;
        i = Math.clamp(i, 1, N);
        j = Math.clamp(j, 1, N);
        return new int[]{i, j};
    }


    // --- Mouse events: Left click: Add density/momentum, Right click: Reset ---
    @Override
    public void mousePressed(final MouseEvent e) {
        if (SwingUtilities.isRightMouseButton(e)) {
            resetSimulation();
        } else {
            prevMouseX = e.getX();
            prevMouseY = e.getY();
        }
    }

    @Override
    public void mouseDragged(final MouseEvent e) {
        int x = e.getX();
        int y = e.getY();
        int[] gridCoords = toGridCoordinates(x, y);
        int i = gridCoords[0];
        int j = gridCoords[1];

        if (prevMouseX != -1 && prevMouseY != -1) {
            int dx = x - prevMouseX;
            int dy = y - prevMouseY;
            velocityXPrev[i][j] += dx * FORCE;
            velocityYPrev[i][j] += dy * FORCE;
        }
        densityPrev[i][j] += DENSITY_AMOUNT;
        prevMouseX = x;
        prevMouseY = y;
    }

    @Override
    public void mouseReleased(final MouseEvent e) {
        prevMouseX = -1;
        prevMouseY = -1;
    }

    @Override
    public void mouseMoved(final MouseEvent e) {
        // Intentionally left blank.
        // No action is required for mouse move events in this simulation.
    }

    @Override
    public void mouseClicked(final MouseEvent e) {
        // Intentionally left blank.
        // No action is required for mouse move events in this simulation.
    }

    @Override
    public void mouseEntered(final MouseEvent e) {
        // Intentionally left blank.
        // No action is required for mouse move events in this simulation.
    }

    @Override
    public void mouseExited(final MouseEvent e) {
        // Intentionally left blank.
        // No action is required for mouse move events in this simulation.
    }

    /**
     * Resets all simulation fields to their initial state.
     * This clears density, velocity, and previous states.
     */
    private void resetSimulation() {
        clearArray(density);
        clearArray(densityPrev);
        clearArray(velocityX);
        clearArray(velocityY);
        clearArray(velocityXPrev);
        clearArray(velocityYPrev);
    }

    /**
     * Main method: Starts the application in a JFrame.
     *
     * @param args Command-line arguments (not used)
     */
    public static void main(final String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ex) {
            LOGGER.log(Level.SEVERE, "Error setting Look and Feel", ex);
        }
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Fluid Simulator");
            frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            FluidSimulation simulator = new FluidSimulation();
            frame.add(simulator);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
