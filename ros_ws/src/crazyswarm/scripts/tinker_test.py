import tkinter as tk

root = tk.Tk()
canvas_width = 400
canvas_height = 400

w = tk.Canvas(root, width=canvas_width, height=canvas_height)
w.pack()



root.mainloop()
print "done"
