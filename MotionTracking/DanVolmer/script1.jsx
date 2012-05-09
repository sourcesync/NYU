{
	
	var composition = app.project.activeItem;
	var layers = composition.selectedLayers;
	//var properties = composition.selectedProperties;
	var numFrames = Math.floor(composition.duration / composition.frameDuration);
	
    //gw
	var layer = composition.selectedLayers[0];
     
     alert(layer);
     alert(layer.motionTracker.numProperties);
     /*
     alert(layer.motionTracker(1)(1));
     alert(layer.motionTracker("FC_1")("Track Point 1"));
     alert( layer.motionTracker("FC_1")("Track Point 1").attachPoint.matchName);
     alert( layer.motionTracker("FC_1")("Track Point 1").attachPoint.numKeys );
     alert( layer.motionTracker("FC_1")("Track Point 1").attachPoint.keyTime(1));
     var t = layer.motionTracker("FC_1")("Track Point 1").attachPoint.keyTime(2);
     var x = layer.motionTracker("FC_1")("Track Point 1").attachPoint.valueAtTime(t,true)[0];
     alert(x);
     var y = layer.motionTracker("FC_1")("Track Point 1").attachPoint.valueAtTime(t,true)[1];
     alert(y);
     */
     //gw
    
	var outputString = ''
	output('<?xml version="1.0" encoding="utf-8"?>');
	output('<data>');
	
	//buildFile();
     buildFile2();
	writeFile();
	
	function buildFile() {
		if (properties.length == 0) {
			alert("No properties selected.");
		}
		else {
			exportProperties();
		}
	}

    function buildFile2() {
		if (layers.length == 0) {
			alert("No layers selected.");
		}
		else {
			exportProperties2();
		}
	}
	
	function output(value) {
		outputString += value + "\r";
	}
	
	function writeFile() {
		output('</data>');
		var file = new File(Folder.desktop.absoluteURI + "/" + "export.txt");
		file.open("w","TEXT","????");
		file.write(outputString);
		file.close();
		file.execute();
	}
	
	function exportProperties() {
		var i = 0;
		var l = properties.length;
		// build properties list
		var x = 0;
		var y = properties.length;
		var source = [];
		var sourceCount = 0;
		var currentContainer;
		for (; x<y; ++x) {
			if (properties[x].valueAtTime != undefined) {
				// find parent
				var parentName;
				var pparentName;
				var nearestkey;
				var numkeys;
				if (properties[x].parentProperty != undefined) {
					//gw
					var parent;
					//gw
					parentName = properties[x].parentProperty.name;
					//gw
					parent = properties[x].parentProperty;
					pparent = parent.parentProperty;
					pparentName = pparent.name;
					numkeys = properties[x].numKeys;
					//gw
				}
				else {
					parentName = "Undefined Parent Source";
				}
				// build property
				//gw
				//output('	<source name="' + parentName + '">');
				output('	<source name="' + parentName + '" parent="' + pparentName + '" num="' + numkeys + '" >');
				//gw
				//gw exportProperty(properties[x]);
				exportProperty2(properties[x])
				output('	</source>');
			}
		}
	}

    function exportProperties2() {
        
        var llayer = composition.selectedLayers[0];
        
        //alert(llayer);
         var xx =1;
         for (xx=1;xx<=llayer.motionTracker.numProperties;xx++)
         {
             //alert(xx);
             var properties = llayer.motionTracker(xx)(1);
              //alert(properties);
             
             var parentName = llayer.motionTracker(xx).name;
             //alert(parentName);
             
             //var yy=1;
             //alert(properties.length);
             //for  (yy=1;yy< properties.length;yy++)
             {
                 output('	<source name="' + parentName + '" >');
                 //alert('1');
                exportProperty2(properties.attachPoint);
                //alert('2');
				output('	</source>');
                //alert('3');
              }
             
             
             continue;
		
        var i = 0;
		var l = properties.length;
		// build properties list
		var x = 0;
		var y = properties.length;
		var source = [];
		var sourceCount = 0;
		var currentContainer;
        
             
		for (; x<y; ++x) {
			if (properties[x].valueAtTime != undefined) {
				// find parent
				var parentName;
				var pparentName;
				var nearestkey;
				var numkeys;
				if (properties[x].parentProperty != undefined) {
					//gw
					var parent;
					//gw
					parentName = properties[x].parentProperty.name;
					//gw
					parent = properties[x].parentProperty;
					pparent = parent.parentProperty;
					pparentName = pparent.name;
					numkeys = properties[x].numKeys;
					//gw
				}
				else {
					parentName = "Undefined Parent Source";
				}
				// build property
				//gw
				//output('	<source name="' + parentName + '">');
				output('	<source name="' + parentName + '" parent="' + pparentName + '" num="' + numkeys + '" >');
				//gw
				//gw exportProperty(properties[x]);
				exportProperty2(properties[x])
				output('	</source>');
			}
		}
    
        }
	}
	
	function exportProperty(prop) {
		var val = prop.valueAtTime(0 * composition.frameDuration, true);
		if (val.length > 1) {
			exportMultiValue(prop);
		}
		else {
			exportSingleValue(prop);
		}
	}
	function exportProperty2(prop) {
		var val = prop.valueAtTime(0 * composition.frameDuration, true);
		if (val.length > 1) {
             alert('m ' + prop.name);
			exportMultiValue2(prop);
		}
		else {
             alert('s '+prop.name);
			exportSingleValue2(prop);
		}
	}
	
	function exportSingleValue(prop) {
		var i = 0;
		var l = numFrames;
		var str = '		<property name="' + prop.name + '">\n';
		for (; i<l; ++i) {
			var time = i * composition.frameDuration;
			var val = prop.valueAtTime(time, true);
			str += '			<keyframe time="' + time + '" frame="' + i + '" value="' + val + '"/>\n';
		}
		str += "		</property>";
		output(str);
	}

	function exportSingleValue2(prop) {
		//var i = 0;
		var i = 1;
		var numKeys = prop.numKeys;
		//var l = numFrame;
		var str = '		<property name="' + prop.name + '">\n';
		
		
		for (; i<=numKeys; ++i) {
			//var time = i * composition.frameDuration;
			var time = (i-1) * composition.frameDuration;
			var key = prop.key(i);
			//var val = prop.valueAtTime(time, true);
			var val = key.value;
			str += '			<keyframe time="' + time + '" frame="' + i + '" value="' + val + '"/>\n';
		}
		str += "		</property>";
		output(str);
	}
	
	function exportMultiValue(prop) {
		var i = 0;
		var l = numFrames;
		var str = '		<property name="' + prop.name + '">\n';
		for (; i<l; ++i) {
			var time = i * composition.frameDuration;
			var val = prop.valueAtTime(time, true);
			if (val.length > 0) {
				str += '			<keyframe time="' + time + '" frame="' + i + '"';
				var x = 0;
				var y = val.length;
				for (; x<y; ++x) {
					str += ' value' + x + '="' + val[x] + '"';
				}
				str += '/>\n';
			}
		}
		str += "		</property>";
		output(str);
	}
	function exportMultiValue2(prop) {
		//var i = 0;
		var i = 1;
		//var l = numFrames;
		var l = prop.numKeys;
		var str = '		<property name="' + prop.name + '">\n';
		//for (; i<l; ++i) {
		for (; i<=l; ++i) {
			//var time = i * composition.frameDuration;
			var time = prop.keyTime(i);
			//var val = prop.valueAtTime(time, true);
			var val = prop.valueAtTime(time,true);
             //var v = prop.key(i).value;
			if (val.length > 0) 
			{
                  var fno = parseInt(time / composition.frameDuration);
				str += '			<keyframe time="' + time + '" frame="' + fno + '"';
				var x = 0;
				var y = val.length;
				for (; x<y; ++x) {
					str += ' value' + x + '="' + val[x] + '"';
				}
				str += '/>\n';
			}
		}
		str += "		</property>";
		output(str);
	}
	
	function outputObject(obj) {
		var out = "";
		for (var s in obj) {
			out += s + "\n";
		}
		alert(out);
	}

}