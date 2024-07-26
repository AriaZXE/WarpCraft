import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import pyperclip
import math

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class PointSelectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Warpacraft")
        self.geometry("1200x800")  
        
        self.label = ctk.CTkLabel(self, text="Select an image")
        self.label.pack(pady=5)
        
        self.select_button = ctk.CTkButton(self, text="Choose Image", command=self.select_image)
        self.select_button.pack(pady=5) 
        
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)
        
        self.canvas_frame = ctk.CTkFrame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='#2c2c2c')
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        self.right_frame = ctk.CTkFrame(self.main_frame, width=220, height=360)
        self.right_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.copy_button = ctk.CTkButton(self, text="Copy Code", command=self.copy_code)
        self.copy_button.pack(pady=5)  
        self.copy_button.configure(state="disabled")
        
        self.reset_button = ctk.CTkButton(self, text="Restart", command=self.reset)
        self.reset_button.pack(pady=5)  
        self.reset_button.configure(state="disabled")
        
        self.points = []
        self.point_ids = []
        self.coord_labels = []
        self.line_ids = []
        self.image = None
        self.img_tk = None
        self.dragging_point = None

        self.warped_img_label = None
        self.angle_label = None
        self.checkmark_label = None

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.display_image()
            self.canvas.bind("<Button-1>", self.on_click)
            self.canvas.bind("<B1-Motion>", self.on_drag)
            self.points = []
            self.point_ids = []
            self.coord_labels = []
            self.line_ids = []
            self.copy_button.configure(state="disabled")
            self.reset_button.configure(state="disabled")

    def display_image(self):
        img = Image.fromarray(self.image)
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def on_click(self, event):
        if len(self.points) < 4:
            x, y = event.x, event.y
            self.points.append((x, y))
            point_id = self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='red', tags="point")
            self.point_ids.append(point_id)
            self.draw_lines(x, y)
            self.add_coord_label(x, y)
            
            if len(self.points) == 4:
                self.show_coordinates()
                self.copy_button.configure(state="normal")
                self.reset_button.configure(state="normal")
                self.show_warped_image()

    def on_drag(self, event):
        if len(self.points) == 4:
            nearest_point, nearest_id = self.find_nearest_point(event.x, event.y)
            if nearest_point is not None:
                self.dragging_point = nearest_point
                self.canvas.coords(nearest_id, event.x-5, event.y-5, event.x+5, event.y+5)
                self.points[nearest_point] = (event.x, event.y)
                self.update_canvas()
                self.show_coordinates()
                self.update_coord_label(nearest_point, event.x, event.y)
                self.show_warped_image()

    def find_nearest_point(self, x, y):
        min_dist = float('inf')
        nearest_point = None
        nearest_id = None
        for i, (px, py) in enumerate(self.points):
            dist = np.sqrt((px - x)**2 + (py - y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_point = i
                nearest_id = self.point_ids[i]
        return nearest_point, nearest_id

    def draw_lines(self, x, y):
        line_id1 = self.canvas.create_line(x, 0, x, self.image.shape[0], fill='blue', dash=(4, 2), tags="line")
        line_id2 = self.canvas.create_line(0, y, self.image.shape[1], y, fill='blue', dash=(4, 2), tags="line")
        self.line_ids.append(line_id1)
        self.line_ids.append(line_id2)

    def update_canvas(self):
        self.canvas.delete("line")
        for (x, y) in self.points:
            self.draw_lines(x, y)

    def add_coord_label(self, x, y):
        coord_label = self.canvas.create_text(x + 10, y, text=f"({x}, {y})", fill="white", anchor=tk.NW, tags="coord_label")
        self.coord_labels.append(coord_label)

    def update_coord_label(self, point_index, x, y):
        self.canvas.coords(self.coord_labels[point_index], x + 10, y)
        self.canvas.itemconfig(self.coord_labels[point_index], text=f"({x}, {y})")

    def show_coordinates(self):
        coords = np.int32(self.points)
        self.coords_str = f"src = np.float32({coords.tolist()})"
        self.label.configure(text=self.coords_str)
        print(self.coords_str)

    def show_warped_image(self):
        if len(self.points) == 4:
            src = np.float32(self.points)
            dst = np.float32([[0, 0], [200, 0], [200, 300], [0, 300]])
            M = cv2.getPerspectiveTransform(src, dst)
            warped_img = cv2.warpPerspective(self.image, M, (200, 300), flags=cv2.INTER_LINEAR)
            warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)

            angle = self.detect_and_show_lines(warped_img)

            if self.warped_img_label is None:
                self.warped_img_label = ctk.CTkLabel(self.right_frame)
                self.warped_img_label.pack(padx=10, pady=10)
                
                self.angle_label = ctk.CTkLabel(self.right_frame, text="")
                self.angle_label.pack(pady=5)
                
                self.checkmark_label = ctk.CTkLabel(self.right_frame, text="")
                self.checkmark_label.pack(pady=5)
            
            img = Image.fromarray(warped_img)
            self.warped_img_tk = ImageTk.PhotoImage(img)
            self.warped_img_label.configure(image=self.warped_img_tk)
            self.angle_label.configure(text=f"Angle: {angle:.1f} degrees")

            if angle == 0.0:
                self.checkmark_label.configure(text="✔️", fg_color="green")
            else:
                self.checkmark_label.configure(text="")

    def detect_and_show_lines(self, warped_img):
        gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is not None and len(lines) >= 2:
            line1 = lines[0][0]
            line2 = lines[1][0]
            rho1, theta1 = line1
            rho2, theta2 = line2
            
            angle = self.calculate_angle(theta1, theta2)
            print(f"Angle between lines: {angle:.1f} degrees")
            return angle
        return 0.0

    def calculate_angle(self, theta1, theta2):
        angle = abs(theta1 - theta2)
        angle = min(angle, np.pi - angle)
        angle = np.degrees(angle)
        return angle

    def copy_code(self):
        pyperclip.copy(self.coords_str)
        self.label.configure(text="Code copied to clipboard!")
    
    def reset(self):
        self.canvas.delete("all")
        self.image = None
        self.img_tk = None
        self.points = []
        self.point_ids = []
        self.coord_ids = []
        self.line_ids = []
        self.copy_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self.label.configure(text="Select an image")
        self.code_textbox.configure(state="normal")
        self.code_textbox.delete(1.0, tk.END)
        self.code_textbox.configure(state="disabled")
        
        if self.warped_img_label:
            self.warped_img_label.destroy()
            self.warped_img_label = None
        
        if self.angle_label:
            self.angle_label.destroy()
            self.angle_label = None
        
        if self.checkmark_label:
            self.checkmark_label.destroy()
            self.checkmark_label = None

if __name__ == "__main__":
    app = PointSelectionApp()
    app.mainloop()
