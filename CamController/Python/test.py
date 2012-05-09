import socket
import sys
import time
import os

#
# Configuration...
#

LISTENERS = [ ( "127.0.0.1", 9000 ), ( "127.0.0.1", 9001 ) ]


def send_cmd( addr, port, cmd, arg ):

	msg = "%d:" % cmd
	if arg:
		msg = msg + arg

	s = socket.socket( socket.AF_INET, socket.SOCK_DGRAM )

	s.sendto( msg, ( addr, port ) )

#
# send "prefix file path" cmd...
#
for listener in LISTENERS:

	addr = listener[0]
	port = listener[1]

	# form path string using address parts...
	path = "%s_%d_take%d" % (addr,port,1)
	send_cmd( addr, port, 2, path )

#
# send "image format" cmd...
#
for listener in LISTENERS:

	addr = listener[0]
	port = listener[1]

	# form path string using address parts...
	send_cmd( addr, port, 3, "3" )

#
# send "start" cmd...
#
for listener in LISTENERS:
	
	addr = listener[0]	
	port = listener[1]
	print "sending start to %s,%d" % ( addr, port )
	send_cmd( addr, port, 0, None )

# wait...
time.sleep(2)

#
# send "stop" cmd...
#
for listener in LISTENERS:
	
	addr = listener[0]	
	port = listener[1]
	print "sending stop to %s,%d" % ( addr, port )
	send_cmd( addr, port, 1, None )

