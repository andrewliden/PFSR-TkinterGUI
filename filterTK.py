import simpleFilter
import tkinter as tk
from tkinter import filedialog

#This is messy.  Sorry.
#I really haven't worked with TK before and this is sort of a
# franticly-constructed weekend project.
class filterApp(tk.Frame):
	def __init__(self, master=None):
		super().__init__(master)
		self.master = master
		self.pack()
		#Declare the default image input & output names
		self.input_img = "img.jpg"
		self.output_img = "result.jpg"
		#Make some widgets.
		self.create_widgets()
		#Assume some values for the input and output image names at first.
		self.master.title("Progressive-Face-Super-Resolution")
		self.master.geometry("500x200")
		
	def create_widgets(self):
		#Create the label explaining that this is the file
		#being opened
		self.open_label = tk.Label(self, text = "Input Image:")
		self.open_label.grid(row = 0, column = 0, columnspan = 2)
		#Create the label telling the user the current input filename.
		self.input_img_labeltext = tk.StringVar()
		self.input_img_labeltext.set(self.input_img)
		self.open_fname_label = tk.Label(self, textvariable = self.input_img_labeltext)
		self.open_fname_label.grid(row = 1, column = 0)
		#Create the button to specify the image to open.
		self.open = tk.Button(self)
		self.open["text"] = "Select input file"
		self.open["command"] = self.open_image
		self.open.grid(row = 1, column = 1)
		#Create the label explaining that this is the file
		#being saved
		self.save_label = tk.Label(self, text = "Output Image:")
		self.save_label.grid(row = 2, column = 0, columnspan = 2)
		#Create the label telling the user the current output filename.
		self.output_img_labeltext = tk.StringVar()
		self.output_img_labeltext.set(self.output_img)
		self.open_fname_label = tk.Label(self, textvariable = self.output_img_labeltext)
		self.open_fname_label.grid(row = 3, column = 0)
		#Create the button to specify the name of the saved image.
		self.save = tk.Button(self)
		self.save["text"] = "Select output file"
		self.save["command"] = self.save_image
		self.save.grid(row = 3, column = 1)
		
		#Create the button to run the filter.
		self.filter = tk.Button(self)
		self.filter["text"] = "Generate image"
		self.filter["command"] = self.generate_image
		self.filter.grid(row = 4, column = 0, columnspan = 2)
	
	def generate_image(self):
		WORKERS = 4
		simpleFilter.testImage(self.input_img, WORKERS, self.output_img)
	
	def open_image(self):
		#Select the image
		self.input_img = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
		#Update the label
		self.input_img_labeltext.set(self.input_img)
		
	def save_image(self):
		self.output_img = filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
		#Update the label
		self.output_img_labeltext.set(self.output_img)
		
if __name__ == '__main__':
	root = tk.Tk()
	app = filterApp(master=root)
	app.mainloop()