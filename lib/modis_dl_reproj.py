import os
import sys
import subprocess
import shutil
from http.cookiejar import CookieJar
import urllib.request, urllib.error, urllib.parse
from html.parser import HTMLParser
import queue
import calendar

import gdal
## tqdm required for download progress bar - imported later


# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):


	def __init__(self):
		HTMLParser.__init__(self)
		self.in_link = False
		self.data_list = []
		self.hdf = 'hdf'
		self.xml = 'xml'
		self.file_name = ''

	# actions we want to take when encountering a starttag. 
	# check that the tag is 'a', and that there is a link
	def handle_starttag(self, tag, attrs):
		self.in_link = False
		self.file_name = ''
		
		if tag == 'a':
			for name, value in attrs:
				if name == 'href':
					if self.hdf in value and self.xml not in value:
						self.in_link = True
						self.lasttag = tag
						self.file_name = value


	# def handle_endtag(self, tag):
	# 	print "Encountered an end tag :", tag

	def handle_data(self, data):
		if self.lasttag == 'a' and self.in_link:
			if self.file_name == data:
				self.data_list.append(data)

'''
needy:
	Controls whether or not the script asks for user confirmation
	 at various steps (confirm number of downloads, redownload of corrupted files, etc)
	When set to False, script assumes the answer to every question is 'yes'.
'''
def modis_dl_reproject(product, collection, date, download_root, needy=True):

	# Select the correct nasa server based on the product requested
	if product == 'MOD09GA':
		nasa_server = 'MOLT'
	elif product == 'MYD09GA':
		nasa_server = 'MOLA'

	# The url of the file we wish to retrieve
	#url = "https://e4ftl01.cr.usgs.gov/MOLT/MOD09GA.006/2014.06.25/"
	#myd = 'https://e4ftl01.cr.usgs.gov/MOLA/MYD09GA.006/'

	url = "https://e4ftl01.cr.usgs.gov/{}/{}.{}/{}/".format(nasa_server,product,collection,date)

	# ADD: Test to make sure they entered a valid product / collection / date

	# Initiate the username / password manager, and the cookie jar for this 
	# session. 
	cookie_jar = initiate_urllib()

	## ------------------------------------------------------------------------
	##  Find list of files (https) to download
	## ------------------------------------------------------------------------

	# Read the html of the http server for the desired modis product
	try:
		html_body = read_html(url)
	except urllib.error.HTTPError:
		print(("Could not reach url: {}".format(url)))
		return

	# instantiate the parser and fed it some HTML text
	parser = MyHTMLParser()
	parser.feed(html_body)

	# Get a list of the available data on the http server
	data_list = parser.data_list

	# Calculate the granules needed to cover the arctic ocean
	granule_list = calc_arctic_granules()

	# Compare the available files with the ones required. Goes through the
	# whole list of needed granules, and finds the full image name that matches
	# each granule ID. Once the right image is found, stopiteration. 
	download_list = []
	for granule in granule_list:
		download_list.append(next((file_ for file_ in data_list if granule in file_),None))

	## ------------------------------------------------------------------------
	###  Download requested granules from NASA server
	## ------------------------------------------------------------------------

	question = "There are [{}] files to download. Continue? [y/n] ".format(len(download_list))
	answer = query_yes_no(question,needy)
	if answer is True:

		# Create the download path based on the images to be downloaded
		download_path = os.path.join(download_root, 'MODIS/{}/{}/originals'.format(product,date))
		if not os.path.isdir(download_path):
			os.makedirs(download_path)

		# Submit the download list to be downloaded
		batch_job(url,download_list,download_path)
	else:
		print("Goodbye.")
		quit()
		# print "Normally I'd quit here."

	# Check that download succeeded
	pass_ = False
	while pass_ is False: 
		pass_, failed_dl = verify_downloads(download_list,download_path)
		if pass_:
			infostring = gdal.Info(os.path.join(download_path,download_list[0]))
			question = "File verification successful. Continue to mosaic and reproject? [y/n] "
			if query_yes_no(question,needy) is False: quit()
		else:
			for dl in failed_dl:
				print(("Failed download: " + dl))
			question = "Would you like to redownload [{}] failed images? [y/n]".format(len(failed_dl))
			answer = query_yes_no(question,needy)
			if answer is True:
				for dl in failed_dl:
					os.remove(os.path.join(download_path,dl))
				batch_job(url,failed_dl,download_path)
			else:
				print("Good day.")
				quit()

	## ------------------------------------------------------------------------
	##   Process downloaded files
	## ------------------------------------------------------------------------
	# Reproject to Arctic Polar Stereographic, mosaic all tiles together,
	# and convert to a geotiff. This is all done using the gdal_warp utility

	# Find the names of all of the subdatasets contained within the downloaded
	# hdf files. In the future, maybe add a way for the user to select the
	# datasets they're actually interested in. 
	sample_im = download_list[0]	# name of one of the downloaded images
	# Read the metadata in the sample image

	# list of subdatasets, read from the metadata
	sds_list = parse_gdalinfo(infostring)	

	# We want to put the mosaic image in the parent directory of the downloads
	dst_path = os.path.split(download_path)[0]
	# output image name is "product_date_sds"
	dst_image_base = sample_im.split('.')[0]+'.'+sample_im.split('.')[1]

	# list of bands we are interested in reprojecting and mosaicing for now
	# Experiementing with queues (maybe useful for multithreading later?)
	reproj_queue = queue.Queue()
	reproj_list = [ 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04',
					'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07', 'state_1km' ]
	
	# reproj_list = [ 'state_1km']

	for band in reproj_list:
		reproj_queue.put(band)
	# Reproject and mosaic each band individually
	# We 'probably' dont want to mosaic _all_ of the bands

	while not reproj_queue.empty():
		band = reproj_queue.get()

		sds = (next(s for s in sds_list if band in s))

		# Parse the band name out of the full sds name
		band_name = sds.split('"')[-1]
		# Replace ':' with '_' in the output name
		dst_band_name = band_name.replace(':','_')
		# Create the full output path
		dst_image = os.path.join(dst_path,dst_image_base+dst_band_name+'.tif')
		# Reproject, mosaic, and convert to geotiff

		print(("Reprojecting and mosaicing " + band_name))
		reproject_mosaic(download_list,band_name,dst_image)

	print("Removing raw files in {}".format(download_path))
	shutil.rmtree(download_path)

#### END modis_dl_reproject()


# Define function for batch downloading
def batch_job(url, files, download_path):
	
	# See if tqdm is installed on this system. If it is, use
	# that to display a progress bar. 
	try:	
		from tqdm import tqdm
		progress_bar = True
	except ImportError:
		print("Install tqdm to display progress bar.")
		progress_bar = False

	existing_files = os.listdir(download_path)
	i = 0
	num_files = len(files)
	for dat in files:
		i += 1
		# Check to see if the file has already been downloaded to this folder
		if dat in existing_files:
			print(("already exists: {}  [{}/{}]".format(dat,i,num_files)))
			continue

		print(("downloading: {}  [{}/{}]".format(dat,i,num_files)))

		# Request the resource at the specified url
		request = urllib.request.Request(url+dat)
		response = urllib.request.urlopen(request)

		if progress_bar:
			# Get the header information from the url response
			meta = dict(response.info())
			# Filesize for the progress bar
			# file_size = int(meta.getheaders("Content-Length")[0])
			file_size = int(meta["Content-Length"][0])
			# Create a tqdm progressbar handle, in human readable units
			pbar = tqdm(total=file_size,unit='B',unit_scale=True)

		# Downloading 8kb blocks at a time. 
		block_sz = 8192
		with open(os.path.join(download_path,dat), 'wb') as handle:
			while True:
				# Request the next block size for download
				buffer_ = response.read(block_sz)

				# If theres nothing left to download, break the loop
				if not buffer_:
					break

				# write the buffer to the file on disk
				handle.write(buffer_)

				# Manually update the progress bar
				if progress_bar:
					pbar.update(len(buffer_))

		# close the connections
		if progress_bar: pbar.close()
		response.close()

	print(("Files downloaded to: ", download_path))

#### END batch_job()


# Function that will call the gdal_merge.py script on all of the images in input_list
# input_list: raw .hdf files downloaded from NASA servers
# subdataset: the subdataset (band) you wish to extract from the hdf file 
def reproject_mosaic(src_images, band_name, dst_image):

	base_path = os.path.split(dst_image)[0]

	if os.path.isfile(dst_image):
		print(("already exists: {}".format(dst_image)))
		os.remove(dst_image)
		# return

	# QA Band is unsigned bit, others are signed. 
	if 'state_1km' in band_name:
		out_res = '1000 1000'
		nodata = 65535
		outtype = 'UInt16'
	else:
		out_res = '500 500'
		nodata = -28672
		outtype = 'Int16'

	# For some reason, gdalwarp called on multiple images introduces a bunch of artifacts. 
	#  Unclear why at this point.
	# Current solution: gdalwarp on each tile individually, then stitch them together with
	#  gdal_merge. I do not know why this produces different outputs. 
	
	# Make a temporary folder to hold the reprojected tiles. This will be deleted
	# after the tiles have been merged into a single image. 
	temp_folder = os.path.join(base_path,'temp')
	if not os.path.isdir(temp_folder):
		os.mkdir(temp_folder)

	# Turn the input list into a single string with a space in between each entry
	single_string = ''
	procs = []
	for image in src_images:
		
		sds_name = 'HDF4_EOS:EOS_GRID:"'+base_path+'/originals/'+image+'"'+band_name

		# image name is in this format: "MOD09GA.A2014176.h17v02.006.2015288102054.hdf"
		tile_name = image.split('.')[2] + '.tif'
		tile_path = os.path.join(temp_folder,tile_name)

		# Saving the output names as a single string for later use in gdal_merge
		single_string = single_string + ' ' + tile_path
		# Call the gdal warp command on the current tile
		cmd = 'gdalwarp \
			-t_srs EPSG:3413 -of GTiff -ot {}\
			-srcnodata {} {} "{}"'.format(outtype, nodata, sds_name, tile_path)

		# Open a subprocess to reproject this current tile
		procs.append(subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
							 stdout=subprocess.PIPE, stderr=subprocess.STDOUT))


		## Uncomment to output result
		# output = p.stdout.read()
		# print output

	# Wait for reprojections to finish
	for p in procs:
		p.wait()

	# Tile all of the reprojected images together to create a single output
	cmd = 'gdal_merge.py -of GTiff -ps {} -ot {} -n {} -o {} {}'.format(out_res, outtype, nodata, dst_image, single_string)
	p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, 
						 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

	# Wait for stitching to finish before removing the temporary folder. 
	p.wait()
	shutil.rmtree(temp_folder)

	# Testing raw output of individual tiles
	# cmd = 'gdal_translate -of GTiff {} {}'.format(sds_list[12], dst_image)

	# cmd = 'gdalwarp -t_srs EPSG:3995 -of GTiff -ot Int16 -r cubicspline \
	# 		-srcnodata -28672 {} "{}"'.format(single_string,dst_image)

#### END reproject_mosaic()


# Create a list of all of the MODIS granules that cover the arctic ocean
# In the future, this could be refined to allow a user to input a custom
# list of granules. 
# Summer Arctic sea ice is covered by these tiles:
# H15:20, v0
# H11:23, v1
# H9:17, v2, H26, v2
# Could just make this a square, where the user enters the TL and BR tiles
# and we download everything. The drawback here would be downloading a bunch
# of useless tiles of russia (v2, h18:25), and having to filter out the empty
# ones (eg, v0,h12)
def calc_arctic_granules():
	granule_list = []

	v = 0
	for h in range(14,22):
		granule = 'h{0:02d}v{1:02d}'.format(h,v)
		granule_list.append(granule)

	v = 1
	for h in range(11,25):
		granule = 'h{0:02d}v{1:02d}'.format(h,v)
		granule_list.append(granule)

	v = 2
	for h in range(9,14):
		granule = 'h{0:02d}v{1:02d}'.format(h,v)
		granule_list.append(granule)
	# h = 26
	# granule = 'h{0:02d}v{1:02d}'.format(h,v)
	# granule_list.append(granule)

	return granule_list

#### END calc_arctic_granules()


# Reads and returns the html of the MODIS http server "https://e4ftl01.cr.usgs.gov/MOLT/"
# with the given product, collection, and date. 
def read_html(url):

	# Create and submit the request. There are a wide range of exceptions that
	# can be thrown here, including HTTPError and URLError. These should be
	# caught and handled.
	request = urllib.request.Request(url)
	response = urllib.request.urlopen(request)

	# Return the result
	body = response.read()
	return str(body)

#### END read_html()


# initiate the password manager and cookie jar for the session
def initiate_urllib():

	# url = "http://e4ftl01.cr.usgs.gov/MOLA/MYD17A3H.006/2009.01.01/MYD17A3H.A2009001.h12v05.006.2015198130546.hdf.xml"
	# The user credentials that will be used to authenticate access to the data
	username = " "
	password = " "

	# Create a password manager to deal with the 401 reponse that is returned from
	# Earthdata Login
	password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
	password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)


	# Create a cookie jar for storing cookies. This is used to store and return
	# the session cookie given to use by the data server (otherwise it will just
	# keep sending us back to Earthdata Login to authenticate).  Ideally, we
	# should use a file based cookie jar to preserve cookies between runs. This
	# will make it much more efficient.
	cookie_jar = CookieJar()


	# Install all the handlers.
	opener = urllib.request.build_opener(
		urllib.request.HTTPBasicAuthHandler(password_manager),
		# urllib2.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
		# urllib2.HTTPSHandler(debuglevel=1),   # details of the requests/responses
		urllib.request.HTTPCookieProcessor(cookie_jar))
	urllib.request.install_opener(opener)

	return cookie_jar

#### END initiate_urllib()


# Extracts the subdataset names from the string returned by gdal.Info
def parse_gdalinfo(infostring):
	meta_data = infostring.split('\n')
	# subdatasets are labeled by "SUBDATASET_N"
	# Each SDS has two entries: NAME and DESC. We are only 
	# intereseted in NAME here. 
	sds_list = [s for s in meta_data if 'SUBDATASET' in s]
	sds_names = [sds for sds in sds_list if 'NAME' in sds]

	return sds_names

#### END parse_gdalinfo()


# Test the requested list of images to make sure they can all
#  be opened. Returns a list of failed downloads if any are
#  encountered. 
def verify_downloads(requested_list, download_folder):
	print("Verifying images...")
	# Assime things went well until we find otherwise
	pass_ = True
	failed_dl = []	# images that failed
	for dl in requested_list:
		try:
			infostring = gdal.Info(os.path.join(download_folder,dl))
		except SystemError:
			pass_ = False
			failed_dl.append

	return pass_, failed_dl

#### END verify_downloads()
	

# Asks the user a yes or no question
# Adapted from https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
# http://code.activestate.com/recipes/577058/
def query_yes_no(question,needy):
	# flag to assume the answer is yes
	if needy == False:
		sys.stdout.write(question)
		print("yes")
		return True

	valid = {'yes': True, 'y': True, 'no': False, 'n': False}
	while True:
		sys.stdout.write(question)
		choice = input().lower()
		if choice in valid:
			return valid[choice]
		else:
			sys.stdout.write("Please respond with 'yes' or 'no'"
							 "(or 'y' or 'n').\n")

#### END query_yes_no

def main():
	# Get these from user input in the future
	product = 'MOD09GA'		#MYD09GA or MOD09GA
	collection = '006'

	cal = calendar.Calendar()
	
	i=0
	for year in range(2015,2018):
		year = 2016
		# March through September
		for month in range(5,10):
			month = 7
			# for day in cal.itermonthdays(year,month):
			for day in [13,14,19,21]:
				# for day in range(20,31):
				#day = 13
				if month == 5 and day < 15:
					continue
				if month == 9 and day > 18:
					continue
				#if day != 0 and month != 7:
				date = '{0:04d}.{1:02d}.{2:02d}'.format(year,month,day)
				# print "{}.{}.{}".format(year,month,day)
				modis_dl_reproject(product,collection,date,'/media/sequoia/',needy=False)
				i+=1
				#quit()
	print(i)


if __name__ == '__main__':
	main()
