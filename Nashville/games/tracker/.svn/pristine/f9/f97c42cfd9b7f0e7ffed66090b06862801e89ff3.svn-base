#
# Configuration...
#

# The default bind address to receive all broadcast LAN traffic...
HOST = "0.0.0.0"

# The default Vicon JSON broadcast port...
PORT = 6667

# The default max packet size (TODO: may need to revisit this!)...
MAX_READSIZE = 100000

# The default read buffer size (TODO: may need to revisit this!)...
DEFAULT_READ_BUFFER_SIZE = 65508

# The default socket blocking type...
BLOCK = 0

# Max times to read in unit test...
MAX_READS = 10000000

# Set verbose debugging...
DEBUG = False

#
# Code...
#
import sys
import json
from socket import *
import select
import time

#
# The main interface class..
#
class ViconProxy:
	
	sock = None
	counter = 0
	good_data_counter = 0
	last_good_data = None
	last_consec_read_count = 0
	block = 0
	
	#
	# Constructor...
	# 
	def __init__(self, host = HOST, port = PORT, block = BLOCK ):
		self.sock = socket( AF_INET, SOCK_DGRAM )
		self.sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1) 
		self.sock.setsockopt(SOL_SOCKET, SO_RCVBUF, DEFAULT_READ_BUFFER_SIZE)
		self.sock.setblocking( block ) # NOTE: we no longer do this because we use select()...
		print "addr->", host, port
		self.sock.bind( ( host, port) )
		self.block = block

	#
	# Get data via select() method...
	#
	def get_json2(self):

		# reset the dequeue counter...
		self.last_consec_read_count = 0

		# dequeue the contents of the socket until empty...
		while True:
			inputs = [ self.sock ]

			# via select(), see if there is a packet to read...
			rready, wready, eready = select.select( inputs, [], [], 0 )

			if rready != []: 
				# read the packet (this better not block or return EWOULDBLOCK )...
				self.last_good_data, addr = self.sock.recvfrom( MAX_READSIZE )
				self.good_data_counter += 1
				self.last_consec_read_count +=1

				# continue reading until nothing left to read...
				continue
			else:
				break
		
		if not self.last_good_data:
			return None
		else:
			return json.loads( self.last_good_data )
	
	#
	# Get data via recvfrom() on socket, you should use get_json2()...
	#
	def get_json(self):
		socket_error = False
		self.counter += 1

		# Read the most recent packet off the socket...
		data = None
		while True:
			try:
				thisdata = None
				if DEBUG: print "INFO: ABOUT TO READ"
				thisdata, addr = self.sock.recvfrom( MAX_READSIZE )
				if DEBUG: print "INFO: AFTER READ!"
				data = thisdata
			except error, msg:
				# Check for special non-block error code...
				if (str(msg)=="[Errno 35] Resource temporarily unavailable"):
					# means there is no or no-more data on the socket...
					if DEBUG: print "INFO: socket might block, thisdata->", thisdata
					break
				else:
					print "ERROR: %s: A socket error occurred->" % (sys.argv[0]), msg
					socket_error = True
					break
			if self.block: # TODO: maybe we should try to also read all packets in block mode too... 
				break


	
		# If we read the max, then its possible the packet was bigger than expected...
		if data and len(data)== MAX_READSIZE:
			print "WARNING: %s: Packet was unexpectedly big!" % sys.argv[0]
			return None
		# Else, return the data packet...
		elif data:
			obj = json.loads(data)
			self.last_data = obj
			return obj
		else:
			return None

	#
	# Print a json object...
	#
	def print_json(self,obj):
		frame_no = obj['fno']
		num_objs = len( obj['objs'] )
		print "frame_no=%d, num_objects=%d" % (frame_no, num_objs)
	
		# Print the packet in a readable way...
		objs = obj['objs']
		for o in objs:
			if type(o)==type([]): # squidball/raw mode...
				print o
			else: # normal/object mode...
				print "obj=", o
				print "name=", o['name'], o['t'] #, o['r'] #,o['ct']
				if o.has_key('mks') and o['mks']!=None:
					mks = o['mks']
					for mk in mks:
						print "\tmk=", mk['name'], mk['t']

	#
	# Get last position of object by name...
	#
	def get_coords(self, name):
		if not self.last_data:
			return None
		objs = self.last_data['objs']
		for o in objs:
			if name == o['name']:
				print "FOUND IT", o['name'], o['t']
				return [ o['t'], o['r'], o['lt'] ]
		return None
	
	#
	# Shut'er down...
	#
	def close(self):
		if self.sock:
			self.sock.close()
			self.sock = None	



#
# Test code...
#

if __name__=="__main__":

	print "INFO: %s: Starting unit test." % (sys.argv[0])

	# Possible print just a specific object (by name)...
	obj_to_get = None
	if len(sys.argv)==2:
		obj_to_get = sys.argv[1]
	
	# Initialize a proxy object...
	vp = ViconProxy()

	# Read a bunch of times in a loop...
	counter = 0
	while (counter < MAX_READS ):
		counter += 1
			
		time.sleep(1)

		# Do a read...
		obj = vp.get_json()
		if obj:
			print "INFO: last consec reads->%d" % ( vp.last_consec_read_count )

			# If got something, print the objects/markers (make it "pretty" please)...
			vp.print_json(obj)
			
			# Possibly print a specific object...
			if (obj_to_get):
				print vp.get_coords( sys.argv[1] )
		else:
			print "INFO: Nothing to read..."	

	# Shut'er down..
	vp.close()
	
	print "INFO: %s: Done." % (sys.argv[0])

