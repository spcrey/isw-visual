pyinstaller --onefile visual_image_sequence.py^
 --hidden-import=tqdm --hidden-import=imageio --hidden-import=matplotlib.backends.backend_svg^
 --hidden-import=numpy --hidden-import=matplotlib --hidden-import=matplotlib.pyplot^
 --hidden-import=netCDF4 --hidden-import=matplotlib.colors --hidden-import=scipy.interpolate^
 --hidden-import=encodings --hidden-import=iswpkg^
 --distpath=bin --name="vis"

pyinstaller --onefile get_coord_data.py^
 --hidden-import=tqdm --hidden-import=imageio --hidden-import=matplotlib.backends.backend_svg^
 --hidden-import=numpy --hidden-import=matplotlib --hidden-import=matplotlib.pyplot^
 --hidden-import=netCDF4 --hidden-import=matplotlib.colors --hidden-import=scipy.interpolate^
 --hidden-import=encodings --hidden-import=iswpkg^
 --distpath=bin --name="gcd"
