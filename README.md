This repo containing the code of simulation and real data for HDFP paper

To run the code, it is suggested to use `docker`, you can pull my image via 


```bash

docker pull huaqingjin/hdf:orthbasis

docker run -it --name test huaqingjin/hdf:orthbasis /bin/bash

```


- The python scripts for running my results are in `python_scripts` folder

You may run it with 

```bash

python simu_settingns.py --cs 0 --setting n1

```

Refer to `python_scripts` readme file for details.


- The notebooks for analyzing results are in `notebooks` folder.

- Main python code are in `mypkg` code 
    
    - `mypkg/optimization/opt.py` contains the may class for `HDFP`, you may refer to this file on how to use it 

- `matlab_scripts` folder contains the code for get PSD from raw time series as well as the code for Bootstrap method (credit to Prof Kaijie Xue) 

- A clean version of my HDFP implentation is ongoing.
