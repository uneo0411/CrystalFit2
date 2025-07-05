import os
import sys
import numpy as np
import csv
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import imageio

# ウィンドウサイズ (録画品質向上のため 1080p)
window_width = 1920
window_height = 1080

def get_nonexistent_filename(filepath):
    if not os.path.exists(filepath):
        return filepath
    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = f"{base}({counter}){ext}"
    while os.path.exists(new_filepath):
        counter += 1
        new_filepath = f"{base}({counter}){ext}"
    return new_filepath

model_size = 5.0
L = 80
grid = np.zeros((L, L, L), dtype=np.int32)
p_class = [
    0.75,  # p6 : 0 個
    0.75,  # p5 : 1 個
    0.75,  # p4 : 2 個
    0.003,  # p3 : 3 個
    0.003,  # p2 : 4 個
    0.003, # p1 : 5 個
    0.003 # p0 : 6 個
]
reaction_counts = []
reaction_rates = []
simulation_finished = False
step_count = 0
captured_frames = []

def update_simulation():
    global grid, reaction_counts, reaction_rates, simulation_finished, step_count
    new_reactions = 0
    new_grid = grid.copy()
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if grid[i,j,k] == 0:
                    cnt = sum([
                        i>0     and grid[i-1,j,k]==0,
                        i<L-1   and grid[i+1,j,k]==0,
                        j>0     and grid[i,j-1,k]==0,
                        j<L-1   and grid[i,j+1,k]==0,
                        k>0     and grid[i,j,k-1]==0,
                        k<L-1   and grid[i,j,k+1]==0
                    ])
                    if np.random.rand() < p_class[cnt]:
                        new_grid[i,j,k] = 1
                        new_reactions += 1
    grid[:] = new_grid
    reaction_counts.append(new_reactions)
    total = L**3
    rate = (np.sum(grid)/total)*100
    reaction_rates.append(rate)
    step_count += 1
    if step_count >= 5 and rate >= 99.5:
        simulation_finished = True
    return rate

def capture_frame():
    glFlush()
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0,0,window_width,window_height,GL_RGBA,GL_UNSIGNED_BYTE)
    img = np.frombuffer(data, dtype=np.uint8).reshape(window_height,window_width,4)
    return np.flipud(img)

def drawText(x, y, text, color=(0,0,0)):
    glColor3f(*color)
    glRasterPos2f(x, y)
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(ch))

def drawVerticalText(x, y, text, angle, color=(0,0,0), scale=0.1):
    height = 119.05
    width = sum(glutStrokeWidth(GLUT_STROKE_ROMAN, ord(ch)) for ch in text)
    glPushMatrix()
    glTranslatef(x, y, 0)
    glRotatef(angle, 0, 0, 1)
    glScalef(scale, scale, scale)
    glTranslatef(-width/2, -height/2, 0)
    glColor3f(*color)
    for ch in text:
        glutStrokeCharacter(GLUT_STROKE_ROMAN, ord(ch))
    glPopMatrix()

def drawCube(x, y, z, size, color):
    glColor4f(*color)
    glPushMatrix()
    glTranslatef(x, y, z)
    glBegin(GL_QUADS)
    # 前面
    glVertex3f(0,0,size);       glVertex3f(size,0,size)
    glVertex3f(size,size,size); glVertex3f(0,size,size)
    # 後面
    glVertex3f(0,0,0);          glVertex3f(0,size,0)
    glVertex3f(size,size,0);    glVertex3f(size,0,0)
    # 左面
    glVertex3f(0,0,0);          glVertex3f(0,0,size)
    glVertex3f(0,size,size);    glVertex3f(0,size,0)
    # 右面
    glVertex3f(size,0,0);       glVertex3f(size,size,0)
    glVertex3f(size,size,size); glVertex3f(size,0,size)
    # 上面
    glVertex3f(0,size,0);       glVertex3f(0,size,size)
    glVertex3f(size,size,size); glVertex3f(size,size,0)
    # 底面
    glVertex3f(0,0,0);          glVertex3f(size,0,0)
    glVertex3f(size,0,size);    glVertex3f(0,0,size)
    glEnd()
    glPopMatrix()

def draw3DModel():
    cell = model_size / L
    for i in range(L):
        for j in range(L):
            for k in range(L):
                col = (1,0,0,0.5) if grid[i,j,k]==0 else (0,0,1,0.03)
                drawCube(i*cell, j*cell, k*cell, cell, col)

def drawOverlay():
    # エラークリア
    while glGetError() != GL_NO_ERROR: pass

    rw = window_width // 2
    rh = window_height

    # ── 修正：正方形サイズを計算 ──
    margin = 20
    size = int(min((rw - margin) * 0.7, (rh - margin) * 0.7))
    overlay_w = overlay_h = size
    ox = (rw - size) // 2
    oy = (rh - size) // 2

    tick = 5
    steps = max(step_count, 1)
    max_c = max(reaction_counts) if reaction_counts else L**3
    max_r = max(reaction_rates) if reaction_rates else 100
    n = 10

    # 投影設定
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, rw, 0, rh, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

    # 外枠＆軸線
    glLineWidth(2); glColor3f(0,0,0)
    glBegin(GL_LINES)
    glVertex2f(ox, oy);                          glVertex2f(ox+overlay_w, oy)
    glVertex2f(ox, oy);                          glVertex2f(ox, oy+overlay_h)
    glVertex2f(ox+overlay_w, oy);                glVertex2f(ox+overlay_w, oy+overlay_h)
    glVertex2f(ox, oy+overlay_h);                glVertex2f(ox+overlay_w, oy+overlay_h)
    glEnd()

    # 横軸 内向き目盛と値
    glBegin(GL_LINES)
    for i in range(n):
        x = ox + i/(n-1)*overlay_w
        glVertex2f(x, oy); glVertex2f(x, oy+tick)
    glEnd()
    for i in range(n):
        val = int(round(i/(n-1)*(steps-1)))
        x = ox + i/(n-1)*overlay_w
        drawText(x-5, oy-20, str(val))

    # 左縦軸 内向き目盛と値
    glBegin(GL_LINES)
    for i in range(n):
        y = oy + i/(n-1)*overlay_h
        glVertex2f(ox, y); glVertex2f(ox+tick, y)
    glEnd()
    for i in range(n):
        val = int(round(i*(max_c/(n-1))))
        y = oy + i/(n-1)*overlay_h
        drawText(ox-50, y-5, str(val))

    # 右縦軸 内向き目盛と値
    glBegin(GL_LINES)
    for i in range(n):
        y = oy + i/(n-1)*overlay_h
        glVertex2f(ox+overlay_w, y); glVertex2f(ox+overlay_w-tick, y)
    glEnd()
    for i in range(n):
        val = int(round(i*(max_r/(n-1))))
        y = oy + i/(n-1)*overlay_h
        drawText(ox+overlay_w+15, y-5, str(val))

    # データ線
    if step_count > 1:
        glColor3f(1,1,0)
        glBegin(GL_LINE_STRIP)
        for i, c in enumerate(reaction_counts):
            x = ox + i/(step_count-1)*overlay_w
            y = oy + c/max_c*overlay_h
            glVertex2f(x, y)
        glEnd()
        glColor3f(0,1,0)
        glBegin(GL_LINE_STRIP)
        for i, r in enumerate(reaction_rates):
            x = ox + i/(step_count-1)*overlay_w
            y = oy + r/max_r*overlay_h
            glVertex2f(x, y)
        glEnd()

    # ラベル
    drawText(ox + overlay_w/2 - 40, oy - 30, "Time Step")
    drawVerticalText(ox - 60, oy + overlay_h/2, "Newly Reacted Cells", 90)
    drawVerticalText(ox + overlay_w + 60, oy + overlay_h/2, "Reaction Rate (%)", -90)

    # 行列復帰
    glPopMatrix()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # 左ビュー（3Dモデル）
    glViewport(0, 0, window_width//2, window_height)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    center = np.array([model_size/2]*3)
    fov = 40.0
    fov_rad = np.radians(fov)
    d = (model_size/2)/np.tan(fov_rad/2)*1.5
    near = d/10; far = d + 2*(np.sqrt(3)/2*model_size)*1.05
    aspect = (window_width/2)/window_height
    gluPerspective(fov, aspect, near, far)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    eye = center + np.array([d,d,d])
    gluLookAt(*eye, *center, 0,1,0)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    draw3DModel()
    # 右ビュー（グラフ）
    glViewport(window_width//2, 0, window_width//2, window_height)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    glOrtho(0, window_width//2, 0, window_height, -1, 1)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    drawOverlay()
    captured_frames.append(capture_frame())
    glutSwapBuffers()

def reshape(w, h):
    global window_width, window_height
    window_width, window_height = w, h
    glViewport(0, 0, w, h)

def timer_callback(value):
    if not simulation_finished:
        rate = update_simulation()
        glutPostRedisplay()
        glutTimerFunc(100, timer_callback, 0)
        if step_count % 10 == 0:
            print(f"Step {step_count}: Reaction rate = {rate:.2f}%")
    else:
        print("Simulation finished")
        # CSV 保存
        csv_path = get_nonexistent_filename("reaction_data.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Step","New Cells","Rate (%)"])
            for i, (c, r) in enumerate(zip(reaction_counts, reaction_rates), 1):
                w.writerow([i, c, r])
        print(f"Data saved: {csv_path}")
        # 動画保存
        vid = get_nonexistent_filename("crystal_simulation.mp4")
        try:
            with imageio.get_writer(
                vid, fps=30, codec='libx264',
                ffmpeg_params=['-crf','18','-preset','slow','-pix_fmt','yuv420p']
            ) as writer:
                for f in captured_frames:
                    writer.append_data(f)
            print(f"Video saved: {vid}")
        except Exception as e:
            print("Error saving video:", e)
        try: glutLeaveMainLoop()
        except: sys.exit()

def keyboard(key, x, y):
    if key == b'\x1b':
        try: glutLeaveMainLoop()
        except: sys.exit()

def init():
    glClearColor(1,1,1,1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_LINE_SMOOTH); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_POLYGON_SMOOTH); glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

if __name__ == '__main__':
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE)
    glutInitWindowSize(window_width, window_height)
    glutCreateWindow(b"Crystal Simulation (PyOpenGL)")
    init()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutTimerFunc(100, timer_callback, 0)
    glutMainLoop()
