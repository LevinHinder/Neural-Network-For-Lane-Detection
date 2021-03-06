# Neural-Network-For-Lane-Detection

## Manual
<ol>
  <li>Download the newest <i>programme.py</i> file from the <a href="https://github.com/LevinHinder/Neural-Network-For-Lane-Detection/releases">release</a>.</li>
  <li>Download the newest <a href="https://www.python.org/downloads/">Python Interpreter</a>. Make sure you can pip install.</li>
  <li>Run <i>programme.py</i> in the Python console. The programme will automaticly install all necessary libraries by itself or, if needed, update them. It will also automaticly install the model. The first startup process may take a view minutes.</li>
  <li>Once the programme has loaded it will show you <code>path:</code>. Write there the full path to your video file you want to process.<br>E.g. <code>path: C:\Users\Levin\Downloads\test.mp4</code></li>
  <li>Wait until the programme has finished. As soon as it's done, it will save the new video file in the directory where <i>programme.py</i> is saved. It is named <i>original_file_name</i>_output.mp4</li>
  <li>To exit the programme write <code>exit</code> as path.</li>
  <li>To uninstall the programme write <code>uninstall</code> as path.</li>
</ol>


## Error Messages
<code>ERROR: Failed to install required libraries</code><br>
Start the programme again. If the programme still crashes after the second restart, install the libraries specified in the console manually.<br><br>
<code>ERROR: Automatic folder to save and load model not found</code><br>
Specify a complete storage path where the model is to be saved. Remember this path as you will need it for future uses of the programme.<br><br>
<code>ERROR: Download failed</code><br>
Start the programme again. If the programme still crashes after the second restart, download the <a href="https://drive.google.com/file/d/1tWqXcNXYtYrVTY2XL9iETIyFOwfU9kSr/view?usp=sharing">v.1.3.0.model</a> file. Save the file in any folder. Finally, enter the folder path in the console when prompted. IMPORTANT: do not change the name of the .model file.<br><br>
<code>ERROR: Model not found</code><br>
Check the storage path of the model you specified. Also check the name of the downloaded file.<br>


## Hardware Specefications
<ul>
  <li>Memory: >5GB</li>
  <li>RAM: >5GB</li>
</ul>
The programme is heavily RAM dependent and will automaticly adapt to the amount of RAM availiable. Therefore more than 8 GB of RAM is recommendable.


## Software Specefications
<ul>
  <li>Python 3.8, 3.9</li>
</ul>
For my testruns I used following library versions.
<ul>
  <li>dill 0.3.4</li>
  <li>numpy 1.21.2</li>
  <li>opencv-python 4.5.3.56</li>
  <li>psutil 5.8.0</li>
  <li>requests 2.26.0</li>
</ul>
There should be no dependencies with regard to the operating system. However testruns were succesful on Windows, MacOS and Linux.


## Dataset
For this project I used the <a href="https://github.com/SoulmateB/CurveLanes">CurveLanes Dataset</a>. With the programme <i>data preparation.py</i> I wrote a little scipt to convert the datapoints into a more uniform way. To do so, I converted all the information from the .JSON files into pictures with binary segmentation.


## License

    MIT License

    Copyright (c) 2022 Levin Hinder

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
