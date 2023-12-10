import tkinter as tk
from tkinter import ttk

def main():
    root = tk.Tk()
    root.title("Live GUI Customization")

    style = ttk.Style()

    # Apply initial styling
    style.configure("Custom.TLabel", background='#F0F0F0', font=('Arial', 12, 'bold'), foreground='#333333')

    label = ttk.Label(root, text="Hello, Tkinter!", style="Custom.TLabel")
    label.pack(pady=20)
    # Create input field for image path
    label_path = tk.Label(root, text="Image Path:")
    label_path.pack()
    root.mainloop()

if __name__ == "__main__":
    main()
