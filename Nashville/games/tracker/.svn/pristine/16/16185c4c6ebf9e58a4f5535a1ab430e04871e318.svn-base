#
# Configuration...
#
DEBUG = False
from vicon_proxy import ViconProxy
import time

#
# Func to get data now (low-level)...
#
def _gnow(vp):

    # To get the most "current" Vicon data
    # Read buffer until we can't anymore

    lastobj = None
    keepgoing = True
    while keepgoing:
        obj = vp.get_json()
        if obj is None:
		if lastobj:
			if DEBUG: print "Returning last obj"
			return lastobj
		else:
			if DEBUG: print "Don't have first yet..."
			continue
	else:
		if DEBUG: print "Got something!"
		lastobj = obj

#
# Func to get data now (low-level) with timeout...
#
def _gnow_withtimeout(vp,timeout):

    lasttime = time.time()

    # To get the most "current" Vicon data
    # Read buffer until we can't anymore

    lastobj = None
    keepgoing = True
    while keepgoing:

	# check any timeout first...
	if (timeout!=None) and ( ( time.time() - lasttime ) > timeout ):
		if DEBUG: print "Timeout..."
		return None
	
	# get vicon data the old way...	
        obj = vp.get_json()
        if obj is None:
		if lastobj:
			if DEBUG: print "Returning last obj"
			return lastobj
		else:
			if DEBUG: print "Don't have first yet..."
			continue
	else:
		if DEBUG: print "Got something!"
		lastobj = obj

#
# Func to get the most recent data from vicon if queued by OS via get_json()...
#
def get_now(vp):
	first_data = _gnow(vp)
	data = _gnow(vp)
	return data

#
# Func to get the most recent data from vicon if queued by OS via get_json()...
#
def get_now_ars(vp,timeout=None):
	first_data = _gnow_withtimeout(vp,timeout)
	data = _gnow_withtimeout(vp,timeout)
	return data

#
# Unit test...
#
if __name__ == "__main__":

	vp = ViconProxy(port=9999)

	while True:
		data = get_now_ars(vp,0.1)
		if data:
			print data['objs'][0]['t']
		else:
			print "no data"
		time.sleep(1)
