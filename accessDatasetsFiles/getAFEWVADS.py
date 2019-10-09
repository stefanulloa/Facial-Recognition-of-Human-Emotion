#generate files for images, V-A values and landmark points for AFEW-VA dataset

import os
import os.path as path
import zipfile
import json

currentDir = os.getcwd()

#create the data folder that will hold all the data files
dataFolder = path.join(path.abspath(path.join(currentDir,'..')), 'data/dataFolder')
if not os.path.exists(dataFolder):
    os.makedirs(dataFolder)

#get 'data' path
currentDirLevelUp = path.abspath(path.join(currentDir,".."))
for files in os.listdir(currentDirLevelUp):
    if files == 'data':
        dataDir = path.join(currentDirLevelUp,files)

zipFilesDir = []

#get list of zip files paths
for file in os.listdir(dataDir):
    if file == 'AFEW_VA':
        subDataDir = path.join(dataDir,file)
        for root, directories, files in os.walk(subDataDir):
            for sub_files in files:
                zipFilesDir.append(path.join(subDataDir,sub_files))

print(zipFilesDir)

#folders for images, valence-arousal and landmarks
for zipFiles in zipFilesDir:
    topID = os.path.splitext(os.path.basename(zipFiles))[0] #corresponds to zip file ID

    imgFolder = path.join(dataFolder, topID+'/img')
    if not os.path.exists(imgFolder):
        os.makedirs(imgFolder)

    lmFolder = path.join(dataFolder, topID+'/annot')
    if not os.path.exists(lmFolder):
        os.makedirs(lmFolder)

    vaFolder = path.join(dataFolder, topID+'/annot2')
    if not os.path.exists(vaFolder):
        os.makedirs(vaFolder)

#extract valence-arousal and landmark points from json files, assign new unique ID to them so that all of them can be in the same folder
for zipFiles in zipFilesDir:
    topID = os.path.splitext(os.path.basename(zipFiles))[0] #corresponds to zip file ID
    with zipfile.ZipFile(zipFiles) as zf:
        for zip_inf in zf.infolist():
            if zip_inf.filename.endswith('.json'):
                with zf.open(zip_inf.filename) as js:
                    data = json.load(js) #data is of type dict
                    videoID = os.path.dirname(zip_inf.filename)
                    for frameKey in data['frames']:

                        #valence-arousal data
                        vaPath = dataFolder+'/'+topID+'/'+videoID+'/annot2/'+topID+'_'+videoID+'_'+frameKey+'.aro'
                        if not os.path.exists(os.path.dirname(vaPath)):
                            os.makedirs(os.path.dirname(vaPath))
                        arousal = data['frames'][frameKey]['arousal']
                        valence = data['frames'][frameKey]['valence']


                        with open(vaPath,"w") as file:
                            file.write(str(arousal) + " " + str(valence) + "\n")


                        #landmark data
                        pointsPath = dataFolder+'/'+topID+'/'+videoID+'/annot/'+topID+'_'+videoID+'_'+frameKey+'.pts'
                        if not os.path.exists(os.path.dirname(pointsPath)):
                            os.makedirs(os.path.dirname(pointsPath))


                        with open(pointsPath, "w") as file:
                            pointsSize = len(data['frames'][frameKey]['landmarks'])
                            file.write('version 1\nn_points: '+str(pointsSize)+'\n{\n')
                            for x_landmark, y_landmark in data['frames'][frameKey]['landmarks']:
                                file.write(str(x_landmark) + " " + str(y_landmark) + "\n")
                            file.write('}')


#extract image files, assign new unique ID to them so that all of them can be in the same folder
for zipFiles in zipFilesDir:
    topID = os.path.splitext(os.path.basename(zipFiles))[0] #corresponds to zip file ID
    with zipfile.ZipFile(zipFiles) as zf:
        for zip_inf in zf.infolist():
            if zip_inf.filename.endswith('.png'):
                videoID, frameID = os.path.split(zip_inf.filename)
                fileRename = topID + '_' + videoID + '_' + frameID
                finalPath = path.join(dataFolder,topID+'/'+videoID+'/img')
                zip_inf.filename = fileRename #this change of filename allows to get only the file with no upper structures
                zf.extract(zip_inf, finalPath) #uncomment to process file transfer
