# Artificial-Eyeliner
Script to apply artificial eyeliner

## Demo:
<table>
  <tr>
    <td><h4>Before</h4></td>
    <td><h4>After</h4></td>
  </tr>
  <tr>
    <td><img src="https://github.com/kaushil24/Artificial-Eyeliner/blob/master/Media/Sample%20Image.jpg" height="250" width="250"></td>
    <td><img src="https://github.com/kaushil24/Artificial-Eyeliner/blob/master/Media/Output%20Image.png" height="250" width="250"></td>
  </tr>
   <tr>
    <td><img src="https://github.com/kaushil24/Artificial-Eyeliner/blob/master/Media/sample%20gif.gif" height="250" width="250"></td>
    <td><img src="https://github.com/kaushil24/Artificial-Eyeliner/blob/master/Media/output%20gif.gif" height="250" width="250"></td>
  </tr>
 </table>
 
 ## CLI Usage
 ```python eyeliner.py [-i image] [-v video] [-d dat] [-t thickness] [-c color] [-s save]```
 * ```-i ```: Location of image you want to apply eyeliner on
 * ```-v ```: Location of video you want to apply eyeliner on.
 * ```-v ```: Live eyeliner of webcam video if ```webcam``` is given
 * ```-t ```: Whole interger number to set thickness of eyeliner. Default = ```2```. Recommended number value between 0-5
 * ```-d```: Path to your ```shape_predictor_68_face_landmarks.dat``` file. Default value is the root unless you have the ```shape_predictor_68_face_landmarks.dat``` file stored at some other location you need not use this argument.
 * ```-c ```: Change color of the eyeliner. Use ```-c 255 255 255```. Defaule = ```0 0 0```.
 * ```-s ```: Location and file name you want to save the output to. **NOTE** The program automatically adds extension while saving the file. **NOTE**: If a file with same name already exists, it will overwrite that file.
 
  

# Version Info:
Python - 3.6
Numpy - 1.17.4
Dlib - 19.18.0
cv2 - 4.1.2
matplotlib - 3.1.2
skimage - 0.16.2
scipy - 1.3.3
imutils - 0.5.3
PIL - 6.2.1
