import numpy as np
from astropy import wcs
from astropy.io import fits
from skimage.filters import rank
import skimage.morphology
import skimage.io
import pickle, os
from matplotlib import pyplot as plt

import re, sys, traceback, logging

def setup_logging_to_file(filename):
	logging.basicConfig( filename='./'+filename,
						 filemode='w',
						 level=logging.DEBUG,
						 format= '%(asctime)s - %(levelname)s - %(message)s',
					   )

def extract_function_name():
	"""Extracts failing function name from Traceback
	by Alex Martelli
	http://stackoverflow.com/questions/2380073/\
	how-to-identify-what-function-call-raise-an-exception-in-python
	"""
	tb = sys.exc_info()[-1]
	stk = traceback.extract_tb(tb, 1)
	fname = stk[0][3]
	return fname

def log_exception(e):
	logging.error(
	"Function {function_name} raised {exception_class} ({exception_docstring}): {exception_message}".format(
	function_name = extract_function_name(), #this is optional
	exception_class = e.__class__,
	exception_docstring = e.__doc__,
	exception_message = e.message))

log_name = "%s.log" % os.path.basename(__file__).split('.py')[0]
setup_logging_to_file(log_name)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info('Beginning %s' % os.path.basename(__file__))

def generate_central_wcs(crval, cdelt):
	# Create a new WCS object.  The number of axes must be set
	# from the start
	w = wcs.WCS(naxis=2)

	# Set up an "Airy's zenithal" projection
	# Vector properties may be set with Python lists, or Numpy arrays
	#CTYPE1  = projection
	#CRVAL1  = central position in degrees
	#CDELT1  = pixel demarcation
	#CRPIX1  = reference pixel
	#CUNIT1  = values of angle objects
	w.wcs.crpix = np.array([0, 0]).astype(int)
	w.wcs.cdelt = np.array(cdelt).astype(float)
	w.wcs.crval = np.array(crval).astype(float)
	w.wcs.ctype = ["RA---SIN", "DEC--SIN"]

	# Some pixel coordinates of interest.
	pixcrd = np.array([[-10, -10], [24, 38], [45, 98]], np.float_)

	# Convert pixel coordinates to world coordinates
	world = w.wcs_pix2world(pixcrd, 1)
	#print(world)

	# Convert the same coordinates back to pixel coordinates.
	pixcrd2 = w.wcs_world2pix(world, 1)
	#print(pixcrd2)

	# These should be the same as the original pixel coordinates, modulo
	# some floating-point error.
	assert np.max(np.abs(pixcrd - pixcrd2)) < 1e-6

	return w

def sharp_edge_type(image_data,erosion):
	'''
	This edge detection algorithm works better for those fits files where
	the flux density drops off sharply to zero. The algorithm shifts the image
	relative to itself to identify the areas that are zero when multiplied
	together.
	'''
	image_data = (image_data != 0)
	nregion = ~ image_data
	shift = 1
	edgex1 = (image_data ^ np.roll(nregion,shift=shift,axis=0))
	edgey1 = (image_data ^ np.roll(nregion,shift=shift,axis=1))
	selem = skimage.morphology.disk(erosion)
	edgex1 = skimage.morphology.erosion(edgex1, selem)
	edgey1 = skimage.morphology.erosion(edgey1, selem)
	del nregion
	x = np.where(edgex1==0)
	y = np.where(edgey1==0)
	x = np.array([x[0],x[1]])
	y = np.array([y[0],y[1]])
	z = np.unique(np.vstack([x.T,y.T]),axis=0).T
	z = np.array([z[1],z[0]])
	del x,y
	return z

def rough_edge_type(image_data,thick,erosion):
	'''
	This edge detection algorithm works better for those fits files where
	the flux density drops less slowly to zero.
	'''
	nregion1 = np.zeros_like(image_data, dtype=np.uint8)
	nregion1[ image_data == 0 ] = 0
	nregion1[ image_data != 0] = 1
	del image_data
	selem = skimage.morphology.rectangle(thick,thick)
	# create masks for the two kinds of edges
	edges = (skimage.filters.rank.minimum(nregion1, selem) == 0) & (skimage.filters.rank.maximum(nregion1, selem) == 1)
	selem = skimage.morphology.disk(erosion)
	edges = skimage.morphology.erosion(edges, selem)
	z = np.where(edges==True)
	z = np.array([z[1],z[0]])
	return z

def pull_FITS_edges(fitsfile_name, w, edgedetection, edgetype, plots):
	'''
	This function returns the plotting coordinates in terms of pixels values
	this is calculated using the w WCS object. This makes the plotting easier
	and allows full plots when no fitsimage is loaded. 
	'''
	fitsfile = fits.open(fitsfile_name)
	header = fitsfile['PRIMARY'].header
	fitswcs = wcs.WCS(header)
	if edgedetection == False:
		world = fitswcs.all_pix2world(
			[[0, 0], [header['NAXIS1'], 0],
			 [header['NAXIS1'], header['NAXIS2']], [0, header['NAXIS2']]], 1)
	elif edgedetection == True:
		image_data = np.nan_to_num(fitsfile['PRIMARY'].data, copy=True)
		if edgetype == 'sharp':
			z = sharp_edge_type(image_data,erosion=2)
		elif edgetype == 'rough':
			z = rough_edge_type(image_data,thick=4,erosion=2)
		if plots == True:
			fig = plt.figure(1,figsize=(8,8))
			ax = fig.add_subplot(111,rasterized=True)
			ax.set_xlim(0,header['NAXIS1'])
			ax.set_ylim(0,header['NAXIS2'])
			ax.imshow(np.log10(image_data),origin='lower',rasterized=True)
			ax.scatter(z[0],z[1],c='k',alpha=0.1)
			fig.savefig('%s_edge_detection.pdf' % fitsfile_name.split('/')[-1], bbox_inches='tight')
			plt.clf()
			#plt.show()
		world = fitswcs.all_pix2world(z.T,1)
	else:
		print('Please set edge detection')
		exit()

	#pix_new_wcs = w.all_world2pix(world, 1)
	pix_new_wcs = world
	fitsfile.close()
	pix_new_wcs = np.vstack([pix_new_wcs, pix_new_wcs[0]])
	return pix_new_wcs.T

#w = generate_central_wcs(crval=[-170.79166667,62.21611111],cdelt=[1e-5,1e-5])

fitsfiles = {}
fitsfiles['HST'] = [
    ['/Volumes/HD-LXU3/eMERGE_data/HST-3D/Images/goodsn_3dhst.v4.0.F125W_orig_sci.fits',True,'sharp',True]]

fits_edges_pixels= {}
for j in fitsfiles.keys():
	array = []
	for i in fitsfiles[j]:
		#print(i[0])
		array = array + [pull_FITS_edges(fitsfile_name=i[0],w='',edgedetection=i[1],edgetype=i[2],plots=i[3])]
	fits_edges_pixels[j] = array

os.system('rm fits_edges.pkl')
#f = open("fits_edges.pkl","wb",protocol=2) #P2 used so python 2 can read
pickle.dump(fits_edges_pixels,open( "fits_edges.pkl", "wb" ),protocol=2)
pickle.dump(w,open( "wcs.pkl", "wb" ),protocol=2)