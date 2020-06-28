import os
import pdfkit

res = os.system("ipython3 nbconvert --to html main.ipynb")
if res:
    print("ipynb to html failed")
pdfkit.from_file("main.html", "main.pdf")
os.system('rm main.html')
