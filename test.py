import tkinter as tk

def create_rounded_button(canvas, x, y, width, height, radius, **kwargs):
    """Create a button with rounded corners on the canvas."""
    # Top-left corner
    canvas.create_arc(x, y, x + 2*radius, y + 2*radius, start=90, extent=90, style=tk.ARC, **kwargs)
    # Top-right corner
    canvas.create_arc(x + width - 2*radius, y, x + width, y + 2*radius, start=0, extent=90, style=tk.ARC, **kwargs)
    # Bottom-left corner
    canvas.create_arc(x, y + height - 2*radius, x + 2*radius, y + height, start=180, extent=90, style=tk.ARC, **kwargs)
    # Bottom-right corner
    canvas.create_arc(x + width - 2*radius, y + height - 2*radius, x + width, y + height, start=270, extent=90, style=tk.ARC, **kwargs)
    # Center rectangle
    canvas.create_rectangle(x + radius, y, x + width - radius, y + height, fill=kwargs.get('fill', ''))
    
def main():
    root = tk.Tk()
    root.title("Rounded Button Corners Example")

    canvas = tk.Canvas(root, width=200, height=50, highlightthickness=0)
    canvas.pack()

    # Set the radius for rounded corners
    radius = 10

    # Create a button with rounded corners on the canvas
    create_rounded_button(canvas, 0, 0, 200, 50, radius, outline="black", width=2, fill="blue", activefill="lightblue")

    # Create button text
    button_text = canvas.create_text(100, 25, text="Click Me", font=("Arial", 12, "bold"), fill="white")

    # Bind the button click event
    canvas.tag_bind(button_text, "<Button-1>", lambda event: print("Button clicked"))

    root.mainloop()

if __name__ == "__main__":
    main()
