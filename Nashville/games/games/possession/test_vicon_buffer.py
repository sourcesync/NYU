from vicon_proxy import ViconProxy


def get_now(vp):
    # To get the most "current" Vicon data
    # Read buffer until we can't anymore

    last_obj = vp.get_json()
    #print last_obj
    #print last_obj['objs'][0]['t']

    go = True
    while go:
        #print "Read"
        obj = vp.get_json()

        if obj is None:
            go = False
        else:
            last_obj = obj
	    #print last_obj
            #print last_obj['objs'][0]['t']
        
    return last_obj



#vp = ViconProxy()

#print get_now()['objs'][0]['t']
