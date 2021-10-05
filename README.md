# Neural-Network-For-Lane-Detection

### Manual
<ol>
  <li>Download the newest <a href="https://github.com/LevinHinder/Neural-Network-For-Lane-Detection/releases">release</a>.</li>
  <li>Donwload the newest <a href="https://drive.google.com/file/d/1WrDlZdjC6sFBnJ5mOpW7xdxqZTTB-qny/view?usp=sharing">model</a>.</li>
  <li>Download the newest <a href="https://www.python.org/downloads/">Python Interpreter</a>. Make sure you can pip install.</li>
  <li>Merge both <i>lane_detection.model</i> and <i>program.py</i> into the same folder.</li>
  <li>Run <i>program.py</i> in the Python console. The program will automaticly install all necessary libraries by itself or, if needed, update them. The startup process can take a view minutes and will only work if both <i>program.py</i> and <i>lane_detection.model</i> are in the same folder.</li>
  <li>Once the program has loaded it will show you <code>path:</code>. Write there the full path to your video file you want to process.<br>E.g. <code>path: C:\Users\Levin\Downloads\test.mp4</code></li>
  <li>Wait until the program has finished. As soon as it's done, it will save the new video file in the directory where <i>program.py</i> is saved. It is named <i>original_file_name</i>_output.mp4</li>
  <li>To exit the program write <code>exit</code> as path.</li>
</ol>


### Hardware Specefications
<ul>
  <li>Memory: >5GB</li>
  <li>RAM: >5GB</li>
</ul>

The program is heavily RAM dependent and will automaticly adapt to the amount of RAM availiable. Therefore more than 8 GB of RAM is desirable.


### Software Specefications
<ul>
  <li>Python 3.8 or higher should work.</li>
</ul>

There should be no dependencies with regard to the operating system.




For this project I used the <a href="https://github.com/SoulmateB/CurveLanes">CurveLanes Dataset</a>.

### License
<span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/InteractiveResource" property="dct:title" rel="dct:type">Neural Network For Lane Detection</span> by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">Levin Hinder</span> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
