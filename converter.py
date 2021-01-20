#converter.py
import commands

#output = commands.getoutput('convert')
op1 = commands.getoutput('ls')
#loops through every Ey file
#print output
for i in range (1,500):
	Ey = 'E.%d' %i
	FileNameEy = Ey + '.pgm'
	NewFileNameEy = Ey+'.tiff'
	output = commands.getoutput('convert '+ FileNameEy +' ' + NewFileNameEy)
	print output



#print 'hi'
#print op1
#reads in the text 
