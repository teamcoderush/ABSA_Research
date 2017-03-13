import urllib.request
import zipfile
import os

# Obtains dataset from Google Drive and extract it.
print('Downloading File...')
urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=0B4eVVdMl2Fr0ektkNENaUmxaWlk','temp.zip')
print('Unzipping file...')
zip_ref = zipfile.ZipFile('temp.zip', 'r')
zip_ref.extractall('.')
zip_ref.close()
print('Removing the temperory file...')
print('Dataset downloaded successfully')
os.remove('temp.zip')
