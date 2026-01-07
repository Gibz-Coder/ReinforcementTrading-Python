#!/usr/bin/env python3
"""
Quick test of the Golden Gibz Native App
"""

import tkinter as tk
from tkinter import messagebox

def test_gui():
    """Test basic GUI functionality"""
    root = tk.Tk()
    root.title("Golden Gibz Test")
    root.geometry("400x300")
    
    # Test label
    label = tk.Label(root, text="🤖 Golden Gibz Trading System", font=('Arial', 14, 'bold'))
    label.pack(pady=20)
    
    # Test button
    def show_message():
        messagebox.showinfo("Test", "GUI is working correctly!")
    
    button = tk.Button(root, text="Test GUI", command=show_message, width=20, height=2)
    button.pack(pady=10)
    
    # Status
    status = tk.Label(root, text="Status: Ready", fg="green")
    status.pack(pady=10)
    
    # Close button
    close_btn = tk.Button(root, text="Close", command=root.destroy, width=10)
    close_btn.pack(pady=10)
    
    print("✅ GUI test window opened successfully")
    root.mainloop()

if __name__ == "__main__":
    try:
        print("🧪 Testing Golden Gibz GUI components...")
        test_gui()
        print("✅ GUI test completed successfully")
    except Exception as e:
        print(f"❌ GUI test failed: {e}")