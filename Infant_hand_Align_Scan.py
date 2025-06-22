import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
import datetime
import os
import webbrowser
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

fingers = {
    "Thumb": [1, 2, 4],
    "Index": [5, 6, 8],
    "Middle": [9, 10, 12],
    "Ring": [13, 14, 16],
    "Pinky": [17, 18, 20]
}

angle_history = [{finger: deque(maxlen=50) for finger in fingers} for _ in range(2)]
saved_filename = None

class HandTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Hand Tracking App")

        self.cap = cv2.VideoCapture(0)

        # ========== Layout ==========
        self.fig_cam = Figure(figsize=(5, 4))
        self.ax_cam = self.fig_cam.add_subplot(111)
        self.ax_cam.axis('off')
        self.ax_cam.set_title("Camera Feed")

        self.fig_3d = Figure(figsize=(5, 4))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax_3d.set_title("3D Hand Structure")

        self.fig_corr1 = Figure(figsize=(5, 3))
        self.ax_corr1 = self.fig_corr1.add_subplot(111)
        self.ax_corr1.set_title("Correlation - Hand 1")

        self.fig_corr2 = Figure(figsize=(5, 3))
        self.ax_corr2 = self.fig_corr2.add_subplot(111)
        self.ax_corr2.set_title("Correlation - Hand 2")

        self.canvas_cam = FigureCanvasTkAgg(self.fig_cam, master=self.root)
        self.canvas_cam.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)

        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.root)
        self.canvas_3d.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)

        self.canvas_corr1 = FigureCanvasTkAgg(self.fig_corr1, master=self.root)
        self.canvas_corr1.get_tk_widget().grid(row=1, column=0, padx=5, pady=5)

        self.canvas_corr2 = FigureCanvasTkAgg(self.fig_corr2, master=self.root)
        self.canvas_corr2.get_tk_widget().grid(row=1, column=1, padx=5, pady=5)

        self.cam_im = self.ax_cam.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

        self.create_menu()

        self.update_loop()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Scan", command=self.save_scan)
        file_menu.add_command(label="Share Scan", command=self.share_scan)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.quit_app)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def get_angle(self, a, b, c):
        def vector(p1, p2):
            return (p2[0] - p1[0], p2[1] - p1[1])
        ba = vector(b, a)
        bc = vector(b, c)
        dot = ba[0]*bc[0] + ba[1]*bc[1]
        mag = math.hypot(*ba) * math.hypot(*bc)
        if mag == 0: return 0
        cos_angle = max(-1, min(1, dot / mag))
        return math.degrees(math.acos(cos_angle))

    def process_hand(self, frame, landmarks, hand_idx):
        h, w, _ = frame.shape
        lm_2d = [(int(p.x * w), int(p.y * h)) for p in landmarks.landmark]

        for name, (a, b, c) in fingers.items():
            angle = self.get_angle(lm_2d[a], lm_2d[b], lm_2d[c])
            angle_history[hand_idx][name].append(angle)
            y_pos = 30 + list(fingers).index(name)*20 + hand_idx * 140
            cv2.putText(frame, f"Hand {hand_idx+1} {name}: {int(angle)}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        return [(p.x, p.y, p.z) for p in landmarks.landmark]

    def plot_correlation(self, ax, data, hand_idx):
        ax.clear()
        corr = np.corrcoef(data)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(fingers.keys(), rotation=45)
        ax.set_yticklabels(fingers.keys())
        ax.set_title(f"Correlation - Hand {hand_idx+1}")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", color="white" if abs(corr[i,j]) > 0.5 else "black")
        return im

    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_loop)
            return

        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_3d_points_list = []

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if i < 2:
                    pts = self.process_hand(frame, hand_landmarks, i)
                    hand_3d_points_list.append(pts)

        # Camera update
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.cam_im.set_data(frame_rgb)

        # 3D update
        self.ax_3d.cla()
        self.ax_3d.set_xlim(0, 1)
        self.ax_3d.set_ylim(0, 1)
        self.ax_3d.set_zlim(-0.5, 0.5)
        self.ax_3d.set_title("3D Hand Structure")
        colors = ['blue', 'green']
        for idx, hand_pts in enumerate(hand_3d_points_list):
            xs, ys, zs = zip(*hand_pts)
            self.ax_3d.scatter(xs, ys, zs, color=colors[idx])
            for a, b in mp_hands.HAND_CONNECTIONS:
                x = [hand_pts[a][0], hand_pts[b][0]]
                y = [hand_pts[a][1], hand_pts[b][1]]
                z = [hand_pts[a][2], hand_pts[b][2]]
                self.ax_3d.plot(x, y, z, color=colors[idx])

        # Correlation updates
        if len(hand_3d_points_list) > 0 and all(len(v) == 50 for v in angle_history[0].values()):
            data = np.array([list(angle_history[0][f]) for f in fingers])
            self.plot_correlation(self.ax_corr1, data, 0)
        if len(hand_3d_points_list) > 1 and all(len(v) == 50 for v in angle_history[1].values()):
            data = np.array([list(angle_history[1][f]) for f in fingers])
            self.plot_correlation(self.ax_corr2, data, 1)

        # Draw canvases
        self.canvas_cam.draw()
        self.canvas_3d.draw()
        self.canvas_corr1.draw()
        self.canvas_corr2.draw()

        self.root.after(30, self.update_loop)

    def save_scan(self):
        global saved_filename
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"hand_scan_{now}.png"
        self.fig_cam.savefig(f"cam_{saved_filename}")
        self.fig_corr1.savefig(f"corr1_{saved_filename}")
        self.fig_corr2.savefig(f"corr2_{saved_filename}")
        self.fig_3d.savefig(f"3d_{saved_filename}")
        messagebox.showinfo("Saved", "Scan images saved successfully.")

    def share_scan(self):
        if not saved_filename or not os.path.exists(f"cam_{saved_filename}"):
            messagebox.showwarning("Warning", "Save the scan before sharing.")
            return
        webbrowser.open(f"https://api.whatsapp.com/send?text=Check%20out%20my%20hand%20scan!%20{saved_filename}")
        webbrowser.open(f"mailto:?subject=Hand Scan&body=Attached is my hand scan.%0A%0A{saved_filename}")

    def quit_app(self):
        self.cap.release()
        hands.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandTrackerApp(root)
    root.mainloop()
